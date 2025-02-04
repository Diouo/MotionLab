import torch

from torchmetrics import Metric
from .utils import *
from rfmotion.models.tmr.load_model import load_model_from_cfg, read_config
from rfmotion.models.tmr.tmr import get_sim_matrix
from rfmotion.utils.temos_utils import lengths_to_mask
from rfmotion.models.tmr.metrics import all_contrastive_metrics_mot2mot


class ReconMetrics(Metric):

    def __init__(self,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Recon'
        self.add_state("count_hint", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("Distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("latent_motions_A",default=[],dist_reduce_fx="cat")
        self.add_state("latent_motions_B",default=[],dist_reduce_fx="cat")

    def compute(self, sanity_flag):
        if sanity_flag:
            return {}
        
        if not isinstance(self.latent_motions_B,list):
            all_gtmotions = self.latent_motions_B.cpu().numpy()
            all_genmotions = self.latent_motions_A.cpu().numpy()
        if isinstance(self.latent_motions_B,list):
            all_gtmotions = torch.cat(self.latent_motions_B).cpu().numpy()
            all_genmotions = torch.cat(self.latent_motions_A).cpu().numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)

        mf_metrics = {}
        mf_metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        mf_metrics["Distance"] = self.Distance / self.count_hint

        return mf_metrics
    
    def update(self, 
               target_emb_ref, target_emb_rst,
               target_motion_ref, target_motion_rst, hint_mask,
               ):
        self.count_hint += torch.sum(hint_mask)

        target_motion_ref = torch.mul(target_motion_ref.to(hint_mask.device), hint_mask)
        target_motion_rst = torch.mul(target_motion_rst.to(hint_mask.device), hint_mask) 
        distance = target_motion_ref - target_motion_rst
        distance_2 = torch.pow(distance, 2)
        distance_sum = torch.sum(distance_2, dim=-1)
        distance_sum = torch.sqrt(distance_sum)
        self.Distance += torch.sum(distance_sum).to(torch.long)

        self.latent_motions_A.append(torch.flatten(target_emb_rst,start_dim=1).detach())
        self.latent_motions_B.append(torch.flatten(target_emb_ref,start_dim=1).detach())