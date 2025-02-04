from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *
from rfmotion.models.tmr.load_model import load_model_from_cfg, read_config
from rfmotion.utils.temos_utils import lengths_to_mask


class UncondSMPLPairedMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 TMR_path,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "fid, and diversity scores"
        self.diversity_times = diversity_times
        self.TMR_cfg = read_config(TMR_path)
        self.TMR = load_model_from_cfg(self.TMR_cfg, eval_mode=True)

        self.add_state("source_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("source_count_seq",default=torch.tensor(0),dist_reduce_fx="sum")

        self.add_state("target_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_count_seq",default=torch.tensor(0),dist_reduce_fx="sum")

        self.metrics = []

        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("FID_source", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("FID_target", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("FID_GT", default=torch.tensor(0.0), dist_reduce_fx="mean")
        # self.metrics.extend(["FID","FID_source","FID_target","FID_GT"])
        self.metrics.extend(["FID"])

        # Diversity
        self.add_state("Diversity",default=torch.tensor(0.0),dist_reduce_fx="sum")
        self.add_state("Diversity_source",default=torch.tensor(0.0),dist_reduce_fx="sum")
        self.add_state("Diversity_target",default=torch.tensor(0.0),dist_reduce_fx="sum")
        self.add_state("Diversity_GT",default=torch.tensor(0.0),dist_reduce_fx="sum")
        # self.metrics.extend(["Diversity", "Diversity_source","Diversity_target","Diversity_GT",])
        self.metrics.extend(["Diversity", "Diversity_GT",])

        # chached batches
        self.add_state("source_recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("source_gtmotion_embeddings", default=[], dist_reduce_fx=None)

        self.add_state("target_recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("target_gtmotion_embeddings", default=[], dist_reduce_fx=None)


    def compute(self, sanity_flag):
        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # ## SOURCE MOTION
        # # cat all embeddings
        # all_gtmotions = torch.cat(self.source_gtmotion_embeddings, axis=0).cpu().numpy()
        # all_genmotions = torch.cat(self.source_recmotion_embeddings, axis=0).cpu().numpy()

        # # Compute fid
        # mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        # metrics["FID_source"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        # # Compute diversity
        # metrics["Diversity_source"] = calculate_diversity_np(all_genmotions,self.diversity_times)


        # ## TARGET MOTION
        # # cat all embeddings
        # all_gtmotions = torch.cat(self.target_gtmotion_embeddings, axis=0).cpu().numpy()
        # all_genmotions = torch.cat(self.target_recmotion_embeddings, axis=0).cpu().numpy()

        # # Compute fid
        # mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        # metrics["FID_target"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        # # Compute diversity
        # metrics["Diversity_target"] = calculate_diversity_np(all_genmotions,self.diversity_times)

        # metrics["FID"] = (metrics["FID_source"]+metrics["FID_target"])/2
        # metrics["Diversity"] = (metrics["Diversity_source"]+metrics["Diversity_target"])/2

        # cat all embeddings
        all_gtmotions = torch.cat(self.source_gtmotion_embeddings + self.target_gtmotion_embeddings, axis=0).cpu().numpy()
        all_genmotions = torch.cat(self.source_recmotion_embeddings + self.target_recmotion_embeddings, axis=0).cpu().numpy()

        ## GENERATED MOTION
        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov) 
        # Compute diversity
        metrics["Diversity"] = calculate_diversity_np(all_genmotions,self.diversity_times)

        ## GROUNDTTURTH MOTION
        # Compute fid
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        # metrics["FID_GT"] = calculate_frechet_distance_np(gt_mu, gt_cov, gt_mu, gt_cov)
        # Compute diversity
        # metrics["Diversity_GT"] = calculate_diversity_np(all_gtmotions,self.diversity_times)

        return {**metrics}
    
    
    def update(
        self,
        length_source, source_recmotion, source_gtmotion,
        length_target, target_recmotion, target_gtmotion,
    ):
        ## source motion
        self.source_count_seq += len(length_source)
        masks_rec, masks_gt = lengths_to_mask(length_source, source_recmotion.device),  lengths_to_mask(length_source, source_gtmotion.device)

        motion_rec_dict = {'length': length_source, 'mask': masks_rec,'x': source_recmotion}
        motion_gt_dict = {'length': length_source, 'mask': masks_gt, 'x': source_gtmotion}
        latent_motion_A = self.TMR.encode(motion_rec_dict, sample_mean=True)
        latent_motion_B = self.TMR.encode(motion_gt_dict, sample_mean=True)

        self.source_recmotion_embeddings.append(torch.flatten(latent_motion_A,start_dim=1).detach())
        self.source_gtmotion_embeddings.append(torch.flatten(latent_motion_B,start_dim=1).detach())

        ## target motion
        self.target_count_seq += len(length_target)
        masks_rec, masks_gt = lengths_to_mask(length_target, target_recmotion.device),  lengths_to_mask(length_target, target_gtmotion.device)

        motion_rec_dict = {'length': length_target, 'mask': masks_rec,'x': target_recmotion}
        motion_gt_dict = {'length': length_target, 'mask': masks_gt, 'x': target_gtmotion}
        latent_motion_A = self.TMR.encode(motion_rec_dict, sample_mean=True)
        latent_motion_B = self.TMR.encode(motion_gt_dict, sample_mean=True)

        self.target_recmotion_embeddings.append(torch.flatten(latent_motion_A,start_dim=1).detach())
        self.target_gtmotion_embeddings.append(torch.flatten(latent_motion_B,start_dim=1).detach())

