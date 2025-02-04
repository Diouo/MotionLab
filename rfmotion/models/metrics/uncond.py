import torch

from torchmetrics import Metric
from .utils import *
from rfmotion.models.tmr.load_model import load_model_from_cfg, read_config
from rfmotion.models.tmr.tmr import get_sim_matrix
from rfmotion.utils.temos_utils import lengths_to_mask
from rfmotion.models.tmr.metrics import all_contrastive_metrics_mot2mot


class UncondMetrics(Metric):

    def __init__(self,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Uncond'
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
        return mf_metrics
    
    def update(self, target_emb_ref, target_emb_rst,):
        self.latent_motions_A.append(torch.flatten(target_emb_rst,start_dim=1).detach())
        self.latent_motions_B.append(torch.flatten(target_emb_ref,start_dim=1).detach())
