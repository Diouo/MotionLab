import torch

from torchmetrics import Metric
from .utils import *
from rfmotion.models.tmr.load_model import load_model_from_cfg, read_config
from rfmotion.models.tmr.tmr import get_sim_matrix
from rfmotion.utils.temos_utils import lengths_to_mask
from rfmotion.models.tmr.metrics import all_contrastive_metrics_mot2mot


class MotionFixMetrics(Metric):

    def __init__(self,
                 TMR_path,
                 diversity_times = 300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Fix'
        self.diversity_times = diversity_times
        self.TMR_cfg = read_config(TMR_path)
        self.TMR = load_model_from_cfg(self.TMR_cfg, eval_mode=True)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

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
        
        count = self.count
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

        mf_metrics["R1_G2T"] = self.R1_G2T / count 
        mf_metrics["R2_G2T"] = self.R2_G2T / count 
        mf_metrics["R3_G2T"] = self.R3_G2T / count 
        mf_metrics["AvgR_G2T"] = self.AvgR_G2T / count 

        mf_metrics["R1_G2S"] = self.R1_G2S / count 
        mf_metrics["R2_G2S"] = self.R2_G2S / count 
        mf_metrics["R3_G2S"] = self.R3_G2S / count 
        mf_metrics["AvgR_G2S"] = self.AvgR_G2S / count 

        return mf_metrics
    
    def update(self, length_source, length_target, source_motion_ref, target_motion_ref, target_motion_rst,):
        self.count += 1
        masks_a = lengths_to_mask(length_target, target_motion_rst.device, max_len=target_motion_rst.shape[1]) 
        masks_b = lengths_to_mask(length_target, target_motion_ref.device, max_len=target_motion_ref.shape[1])
        masks_c = lengths_to_mask(length_source, source_motion_ref.device, max_len=source_motion_ref.shape[1])

        motion_a_dict = {'length': length_target, 'mask': masks_a,'x': target_motion_rst}
        motion_b_dict = {'length': length_target, 'mask': masks_b, 'x': target_motion_ref}
        motion_c_dict = {'length': length_source, 'mask': masks_c, 'x': source_motion_ref}
        # Encode the motion
        latent_motion_A = self.TMR.encode(motion_a_dict, sample_mean=True, modality='motion')
        latent_motion_B = self.TMR.encode(motion_b_dict, sample_mean=True, modality='motion')
        latent_motion_C = self.TMR.encode(motion_c_dict, sample_mean=True, modality='motion')

        self.latent_motions_A.append(torch.flatten(latent_motion_A,start_dim=1).detach())
        self.latent_motions_B.append(torch.flatten(latent_motion_B,start_dim=1).detach())
        self.latent_motions_C.append(torch.flatten(latent_motion_C,start_dim=1).detach())

        sim_matrix = get_sim_matrix(latent_motion_A, latent_motion_B).detach().cpu()
        sim_matrix, cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix, emb=None, threshold=None, return_cols=True)
        self.R1_G2T += sim_matrix['m2m/R01']
        self.R2_G2T += sim_matrix['m2m/R02']
        self.R3_G2T += sim_matrix['m2m/R03']
        self.AvgR_G2T += sim_matrix['m2m/AvgR']


        sim_matrix = get_sim_matrix(latent_motion_A, latent_motion_C).detach().cpu()
        sim_matrix, cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix, emb=None, threshold=None, return_cols=True)
        self.R1_G2S += sim_matrix['m2m/R01']
        self.R2_G2S += sim_matrix['m2m/R02']
        self.R3_G2S += sim_matrix['m2m/R03']
        self.AvgR_G2S += sim_matrix['m2m/AvgR']