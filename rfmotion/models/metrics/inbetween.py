import torch

from scipy.ndimage import uniform_filter1d
from torchmetrics import Metric

from .utils import *
from rfmotion.models.tmr.tmr import get_sim_matrix
from rfmotion.models.tmr.metrics import all_contrastive_metrics_mot2mot


class InbetweenMetrics(Metric):

    def __init__(self,
                 diversity_times = 300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Inbetween'
        self.diversity_times = diversity_times

        self.add_state("count_batch", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_hint", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_frame", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("Distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("latent_motions_A",default=[],dist_reduce_fx="cat")
        self.add_state("latent_motions_B",default=[],dist_reduce_fx="cat")

    def compute(self, sanity_flag):
        if sanity_flag:
            return {}
        
        if not isinstance(self.latent_motions_B,list):
            all_gtmotions = self.latent_motions_A.cpu().numpy()
            all_genmotions = self.latent_motions_B.cpu().numpy()
        if isinstance(self.latent_motions_B,list):
            all_gtmotions = torch.cat(self.latent_motions_A).cpu().numpy()
            all_genmotions = torch.cat(self.latent_motions_B).cpu().numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)

        mf_metrics = {}
        mf_metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        mf_metrics["Distance"] = self.Distance / self.count_hint

        return mf_metrics
    
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

    
    def update(self, length_target, hint_mask, 
               target_motion_ref, target_motion_rst,
               target_emb_ref, target_emb_rst,):
        self.count_frame += sum(length_target)
        self.count_seq += len(length_target)
        self.count_batch += 1
        self.count_hint += torch.sum(hint_mask)

        target_motion_ref = torch.mul(target_motion_ref.to(hint_mask.device), hint_mask)
        target_motion_rst = torch.mul(target_motion_rst.to(hint_mask.device), hint_mask) 
        distance = target_motion_ref - target_motion_rst
        distance_2 = torch.pow(distance, 2)
        distance_sum = torch.sum(distance_2, dim=-1)
        distance_sum = torch.sqrt(distance_sum)
        self.Distance += torch.sum(distance_sum).to(torch.long)

        self.latent_motions_A.append(torch.flatten(target_emb_ref,start_dim=1).detach())
        self.latent_motions_B.append(torch.flatten(target_emb_rst,start_dim=1).detach())