from typing import List
from scipy.ndimage import uniform_filter1d
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class HintMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count_frame", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",default=torch.tensor(0),dist_reduce_fx="sum")
        self.add_state("count_hint", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_batch", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []

        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # Distance
        self.add_state("Distance", default=torch.tensor(0), dist_reduce_fx="sum")

        # chached batches
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_genmotions = torch.cat(self.recmotion_embeddings,axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings,axis=0).cpu()[shuffle_idx, :]

        # # Compute r-precision with gt
        # assert count_seq > self.R_size
        # top_k_mat = torch.zeros((self.top_k, ))
        # for i in range(count_seq // self.R_size):
        #     # [bs=32, 1*256]
        #     group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
        #     # [bs=32, 1*256]
        #     group_motions = all_gtmotions[i * self.R_size:(i + 1) *
        #                                   self.R_size]
        #     # [bs=32, 32]
        #     dist_mat = euclidean_distance_matrix(group_texts,
        #                                          group_motions).nan_to_num()
        #     # match score
        #     self.gt_Matching_Score += dist_mat.trace()
        #     argsmax = torch.argsort(dist_mat, dim=1)
        #     top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        # metrics["gt_Matching_Score"] = self.gt_Matching_Score / R_count
        # for k in range(self.top_k):
        #     metrics[f"gt_R{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # metrics["gt_Diversity"] = calculate_diversity_np(
        #     all_gtmotions, self.diversity_times)

        metrics["Distance"] = self.Distance / self.count_hint


        return {**metrics}

    def calculate_skating_ratio(self, motions):
        motions = motions.permute(0, 2, 3, 1)  # motions [bs, 22, 3, max_len]
        thresh_height = 0.05 # 10
        fps = 20.0
        thresh_vel = 0.50 # 20 cm /s
        avg_window = 5 # frames

        batch_size = motions.shape[0]
        # 10 left, 11 right foot. XZ plane, y up
        # motions [bs, 22, 3, max_len]
        verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
        verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
        # [bs, 2, max_len-1]
        vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

        verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
        # If feet touch ground in agjecent frames
        feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
        # skate velocity
        skate_vel = feet_contact * vel_avg

        # it must both skating in the current frame
        skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
        # and also skate in the windows of frames
        skating = np.logical_and(skating, (vel_avg > thresh_vel))

        # Both feet slide
        skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
        skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

        return skating_ratio, skate_vel

    def update(
        self,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
        recmotion: Tensor,
        gtmotion: Tensor,
        hint_masks: Tensor,
    ):
        self.count_frame += sum(lengths)
        self.count_seq += len(lengths)
        self.count_batch += 1
        self.count_hint += torch.sum(hint_masks)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        recmotion_embeddings = torch.flatten(recmotion_embeddings,
                                             start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
                                            start_dim=1).detach()

        # store all texts and motions
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)

        gtmotion = torch.mul(gtmotion.to(hint_masks.device), hint_masks)
        recmotion = torch.mul(recmotion.to(hint_masks.device), hint_masks) 
        distance = gtmotion - recmotion
        distance_2 = torch.pow(distance, 2)
        distance_sum = torch.sum(distance_2, dim=-1)
        distance_sum = torch.sqrt(distance_sum)
        self.Distance += torch.sum(distance_sum).to(torch.long)
