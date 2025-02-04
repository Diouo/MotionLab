import torch

from torchmetrics import Metric
from .utils import *
from rfmotion.models.tmr.load_model import load_model_from_cfg, read_config
from rfmotion.models.tmr.tmr import get_sim_matrix
from rfmotion.utils.temos_utils import lengths_to_mask
from rfmotion.models.tmr.metrics import all_contrastive_metrics_mot2mot


class SourceHintMetrics(Metric):

    def __init__(self,
                 diversity_times = 300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Source Text'
        self.diversity_times = diversity_times

        self.add_state("count_batch", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_hint", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("Distance", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("R1_G2T",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("R2_G2T",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("R3_G2T",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("AvgR_G2T",default=torch.tensor([0.0]),dist_reduce_fx="sum")

        self.add_state("R1_G2S",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("R2_G2S",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("R3_G2S",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("AvgR_G2S",default=torch.tensor([0.0]),dist_reduce_fx="sum")

        self.add_state("latent_motions_A",default=[],dist_reduce_fx="cat")
        self.add_state("latent_motions_B",default=[],dist_reduce_fx="cat")
        self.add_state("latent_motions_C",default=[],dist_reduce_fx="cat")

    def compute(self, sanity_flag):
        if sanity_flag:
            return {}
        
        count = self.count_batch
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
        mf_metrics["Diversity"] = calculate_diversity_np(all_genmotions,self.diversity_times)
        mf_metrics["Distance"] = self.Distance / self.count_hint

        mf_metrics["R1_G2T"] = self.R1_G2T / count 
        mf_metrics["R2_G2T"] = self.R2_G2T / count 
        mf_metrics["R3_G2T"] = self.R3_G2T / count 
        mf_metrics["AvgR_G2T"] = self.AvgR_G2T / count 

        mf_metrics["R1_G2S"] = self.R1_G2S / count 
        mf_metrics["R2_G2S"] = self.R2_G2S / count 
        mf_metrics["R3_G2S"] = self.R3_G2S / count 
        mf_metrics["AvgR_G2S"] = self.AvgR_G2S / count 

        return mf_metrics
    
    def update(self, 
               source_emb_ref, target_emb_ref, target_emb_rst,
               target_motion_ref, target_motion_rst, hint_mask,
               ):
        self.count_batch += 1
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
        self.latent_motions_C.append(torch.flatten(source_emb_ref,start_dim=1).detach())

        sim_matrix = get_sim_matrix(target_emb_rst, target_emb_ref).detach().cpu()
        sim_matrix, cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix, emb=None, threshold=None, return_cols=True)
        self.R1_G2T += sim_matrix['m2m/R01']
        self.R2_G2T += sim_matrix['m2m/R02']
        self.R3_G2T += sim_matrix['m2m/R03']
        self.AvgR_G2T += sim_matrix['m2m/AvgR']


        sim_matrix = get_sim_matrix(target_emb_rst, source_emb_ref).detach().cpu()
        sim_matrix, cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix, emb=None, threshold=None, return_cols=True)
        self.R1_G2S += sim_matrix['m2m/R01']
        self.R2_G2S += sim_matrix['m2m/R02']
        self.R3_G2S += sim_matrix['m2m/R03']
        self.AvgR_G2S += sim_matrix['m2m/AvgR']