import numpy as np
import torch
import time
import math
import itertools
import random

from torch.optim import AdamW
from torchmetrics import MetricCollection

from .base import BaseModel
from rfmotion.config import instantiate_from_config
from rfmotion.models.losses.rfmotion import RFMotionLosses
from rfmotion.models.modeltype.base import BaseModel
from rfmotion.models.operator.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from rfmotion.utils.temos_utils import lengths_to_mask, lengths_to_query_mask

class RFMOTION_SEPERATE(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()
        self.datamodule = datamodule
        self.cfg = cfg
    
        self.stage = cfg.TRAIN.STAGE
        self.is_vae = cfg.TRAIN.ABLATION.VAE
        self.debug = cfg.DEBUG
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.token_dim = cfg.model.token_dim
        self.ln_sampling = cfg.TRAIN.ABLATION.LN_Sampling
        self.warm_up = cfg.TRAIN.ABLATION.WARM_UP
        self.is_hint_guidance = cfg.TRAIN.ABLATION.HINT_GUIDANCE
        self.condition_type = cfg.model.condition_type

        if self.is_vae:
            if "mld" in cfg.model.motion_vae.target:
                self.vae_type = 'mld'
            elif "rfmotion" in cfg.model.motion_vae.target:
                if "native" in cfg.model.motion_vae.target:
                    self.vae_type = 'rfmotion_native'
                elif "seperate" in cfg.model.motion_vae.target:
                    self.vae_type = 'rfmotion_seperate'

            self.vae = instantiate_from_config(cfg.model.motion_vae)
            if self.stage == "diffusion":  # Don't train the motion encoder and decoder
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
        else:
            self.vae_type = 'no'
        
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.predict_type = "v_prediction"
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.model.noise_scheduler.params.num_train_timesteps)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.model.noise_scheduler.params.num_train_timesteps)
        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.get_style_encoder(cfg)
        self.get_content_encoder(cfg)
        self.get_style_test_dataset(cfg)
        self.configure_conditions_prob_and_scale(cfg, condition_type=self.condition_type)
        
        self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,params=self.parameters())
        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()
        self.get_t2m_evaluator(cfg)
        self._losses = MetricCollection({split: RFMotionLosses(vae=self.is_vae, mode="xyz", cfg=cfg, prediction_type=self.predict_type) for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

    def diffusion_reverse(self, stage,
                          target_motion, target_lengths, target_lengths_z,
                          source_motion=None, source_lengths=None, source_lengths_z=None,
                          text=None, text_lengths=None,
                          hint=None, hint_lengths=None, hint_masks=None,
                          content=None, content_lengths=None,
                          style=None, style_lengths=None,):
        noise = torch.randn_like(target_motion, device=target_motion.device,dtype=torch.float,)
        noisy_latents = noise.clone()
        num_inference_steps = self.cfg.model.scheduler.num_eval_steps if stage == "eval" else self.cfg.model.scheduler.num_demo_steps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=target_motion.device)
        timesteps = self.scheduler.timesteps.to(torch.int32)

        # reverse
        for i, t in enumerate(timesteps):
            if t==0:
                continue

            # expand the latents if we are doing classifier free guidance
            if self.condition_type == "none":
                latent_model_input = noisy_latents
                source_motion_input = source_motion
                source_lengths_input = source_lengths
                source_lengths_z_input = source_lengths_z
                target_lengths_input = target_lengths
                target_lengths_z_input = target_lengths_z
            elif self.condition_type == "text" or self.condition_type == "hint" or self.condition_type == "inbetween":
                latent_model_input = torch.cat([noisy_latents] *2)
                source_motion_input = None
                source_lengths_input = None
                source_lengths_z_input = None
                target_lengths_input = target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z
            elif self.condition_type == "style":
                latent_model_input = torch.cat([noisy_latents] *2)
                source_motion_input = None
                source_lengths_input = None
                source_lengths_z_input = None
                target_lengths_input = target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z
            elif self.condition_type == "source_text" or self.condition_type == "source_hint":
                latent_model_input = torch.cat([noisy_latents] *3)
                source_motion_input = torch.cat([source_motion] *3)
                source_lengths_input = [0] * len(source_lengths) + source_lengths + source_lengths
                source_lengths_z_input = [0] * len(source_lengths_z) + source_lengths_z + source_lengths_z
                target_lengths_input = target_lengths + target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z + target_lengths_z

            v_pred = self.denoiser(instructions=None,
                                   hidden_states=latent_model_input,
                                   timestep=t,
                                   text=text,
                                   text_lengths=text_lengths,
                                   hint=hint,
                                   hint_lengths=hint_lengths,
                                   style=style,
                                   style_lengths=style_lengths,
                                   content=content,
                                   content_lengths=content_lengths,
                                   source_motion=source_motion_input,
                                   source_lengths=source_lengths_input,
                                   source_lengths_z=source_lengths_z_input,
                                   target_lengths=target_lengths_input,
                                   target_lengths_z=target_lengths_z_input,
                                   return_dict=False,)[0]

            # perform CFG guidance
            if self.condition_type == "text":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.text_guidance_scale * (v_cond - v_uncond)
            elif self.condition_type == "hint":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.text_hint_guidance_scale * (v_cond - v_uncond)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif self.condition_type == "inbetween":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.inbetween_guidance_scale * (v_cond - v_uncond)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif self.condition_type=='source_text':
                v_pred_dropall, v_pred_with_source, v_pred_with_all = v_pred.chunk(3)
                v_pred = v_pred_dropall + self.source_text_guidance_scale_1 * (v_pred_with_source - v_pred_dropall) + self.source_text_guidance_scale_2 * (v_pred_with_all - v_pred_with_source)
            elif self.condition_type=='source_hint':
                v_pred_dropall, v_pred_with_source, v_pred_with_all = v_pred.chunk(3)
                v_pred = v_pred_dropall + self.source_hint_guidance_scale_1 * (v_pred_with_source - v_pred_dropall) + self.source_hint_guidance_scale_2 * (v_pred_with_all - v_pred_with_source)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif self.condition_type == "style":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.style_guidance_scale * (v_cond - v_uncond)
            noisy_latents = self.scheduler.step(v_pred, t, noisy_latents, return_dict=False)[0]

        return noisy_latents

    def diffusion_process(self, 
                          target_motion, target_lengths, target_lengths_z,
                          source_motion=None, source_lengths=None, source_lengths_z=None,
                          text=None, text_lengths=None,
                          style=None, style_lengths=None,
                          content=None, content_lengths=None,
                          hint=None, hint_lengths=None, hint_masks=None,):

        # prepocess
        bsz = target_motion.shape[0]
        timesteps = torch.randint(0,self.noise_scheduler.config.num_train_timesteps+1,(bsz, ),device=target_motion.device,dtype=torch.long,) 

        # mid motion between source_motion adn target_motion
        noise = torch.randn_like(target_motion, device=target_motion.device)
        noisy_latents = self.noise_scheduler.scale_noise(sample=target_motion.clone(), noise=noise, timestep=timesteps) 

        # predicted result
        v_pred = self.denoiser(instructions=None,
                               hidden_states=noisy_latents,
                               timestep=timesteps,
                               text=text,
                               text_lengths=text_lengths,
                               style=style,
                               style_lengths=style_lengths,
                               content = content,
                               content_lengths = content_lengths,
                               hint=hint,
                               hint_lengths=hint_lengths,
                               source_motion=source_motion,
                               source_lengths=source_lengths,
                               source_lengths_z=source_lengths_z,
                               target_lengths=target_lengths,
                               target_lengths_z=target_lengths_z,
                               return_dict=False,)[0]
        v_gt = noise-target_motion

        if not self.is_vae:
            target_mask = lengths_to_mask(target_lengths, target_motion.device, max_len=v_pred.shape[1]).unsqueeze(-1).expand(-1, -1, target_motion.shape[-1])
        elif self.vae_type == "rfmotion_native":
            target_mask = lengths_to_mask(target_lengths_z, target_motion.device, max_len=v_pred.shape[1]).unsqueeze(-1).expand(-1, -1, target_motion.shape[-1])
        elif self.vae_type == "rfmotion_seperate":
            _, target_mask = lengths_to_query_mask(target_lengths, target_lengths_z, target_motion.device, max_len=v_pred.shape[1])
            target_mask = target_mask.unsqueeze(-1).expand(-1, -1, target_motion.shape[-1])
        n_set = {"v_gt": torch.mul(v_gt, target_mask),"v_pred": torch.mul(v_pred, target_mask),}

        return n_set

    def get_motion_batch(self, batch):
        target_motion = batch["motion"].to(self.device)
        target_lengths = batch["length"]

        return target_motion, target_lengths

    def get_paired_motion_one(self, batch):
        if self.datamodule.name == "motionfix":
            if "body_transl_delta_pelv" in self.datamodule.load_feats and "body_joints_local_wo_z_rot" in self.datamodule.load_feats:
                source_motion = torch.cat((batch["body_transl_delta_pelv_source"], batch["body_orient_xy_source"], batch["z_orient_delta_source"], batch["body_pose_source"], batch["body_joints_local_wo_z_rot_source"]), 1)
                target_motion = torch.cat((batch["body_transl_delta_pelv_target"], batch["body_orient_xy_target"], batch["z_orient_delta_target"], batch["body_pose_target"], batch["body_joints_local_wo_z_rot_target"]), 1)
            elif "body_transl" in self.datamodule.load_feats:
                source_motion = torch.cat((batch["body_transl_source"], batch["body_orient_source"], batch["body_pose_source"]), 1)
                target_motion = torch.cat((batch["body_transl_target"], batch["body_orient_target"], batch["body_pose_target"]), 1)

            source_motion = source_motion.unsqueeze(0).to(self.device)
            target_motion = target_motion.unsqueeze(0).to(self.device)

            source_motion = self.datamodule.cat_inputs(self.datamodule.norm_inputs(self.datamodule.uncat_inputs(source_motion,self.datamodule.nfeats),self.datamodule.load_feats))[0]
            target_motion = self.datamodule.cat_inputs(self.datamodule.norm_inputs(self.datamodule.uncat_inputs(target_motion,self.datamodule.nfeats),self.datamodule.load_feats))[0]

            source_lengths = [batch['length_source']]
            target_lengths = [batch['length_target']]

        elif self.datamodule.name == "motionfix_retarget":
            source_motion = batch["source_motion"].to(self.device)
            target_motion = batch["target_motion"].to(self.device)  

            source_lengths = batch['length_source']
            target_lengths = batch['length_target']

        max_source_lengths = source_motion.shape[1]
        source_lengths = [i if i < max_source_lengths else max_source_lengths for i in source_lengths]

        max_target_lengths = target_motion.shape[1]
        target_lengths = [i if i < max_target_lengths else max_target_lengths for i in target_lengths]

        return source_motion, source_lengths, target_motion, target_lengths

    def get_paired_motion_batch(self, batch):
        if self.datamodule.name == "motionfix":
            if "body_transl_delta_pelv" in self.datamodule.load_feats:
                source_motion = torch.cat((batch["body_transl_delta_pelv_source"], batch["body_orient_xy_source"], batch["z_orient_delta_source"], batch["body_pose_source"], batch["body_joints_local_wo_z_rot_source"]), 2)
                target_motion = torch.cat((batch["body_transl_delta_pelv_target"], batch["body_orient_xy_target"], batch["z_orient_delta_target"], batch["body_pose_target"], batch["body_joints_local_wo_z_rot_target"]), 2)
            elif "body_transl" in self.datamodule.load_feats:
                source_motion = torch.cat((batch["body_transl_source"], batch["body_orient_source"], batch["body_pose_source"]), 2)
                target_motion = torch.cat((batch["body_transl_target"], batch["body_orient_target"], batch["body_pose_target"]), 2)
            source_motion = self.datamodule.cat_inputs(self.datamodule.norm_inputs(self.datamodule.uncat_inputs(source_motion,self.datamodule.nfeats),self.datamodule.load_feats))[0]
            target_motion = self.datamodule.cat_inputs(self.datamodule.norm_inputs(self.datamodule.uncat_inputs(target_motion,self.datamodule.nfeats),self.datamodule.load_feats))[0]
        elif self.datamodule.name == "motionfix_retarget":
            source_motion = batch["source_motion"].to(self.device)
            target_motion = batch["target_motion"].to(self.device)  

        source_lengths = batch['length_source']
        max_source_lengths = source_motion.shape[1]
        source_lengths = [i if i < max_source_lengths else max_source_lengths for i in source_lengths]

        target_lengths = batch['length_target']
        max_target_lengths = target_motion.shape[1]
        target_lengths = [i if i < max_target_lengths else max_target_lengths for i in target_lengths]

        return source_motion, source_lengths, target_motion, target_lengths

    def get_paired_motion_cat(self, batch):
        target_motion = batch["motion"].to(self.device)
        target_lengths = batch["length"]

        reference_motion = batch["reference_motion"].to(self.device)
        reference_lengths = batch["reference_length"].detach().cpu().tolist()

        # padding and aligning
        if target_motion.shape[1] < reference_motion.shape[1]:
            target_motion = torch.cat([target_motion, torch.zeros(target_motion.shape[0], reference_motion.shape[1] - target_motion.shape[1], target_motion.shape[2], device=target_motion.device)], dim=1)
        elif target_motion.shape[1] > reference_motion.shape[1]:
            reference_motion = torch.cat([reference_motion, torch.zeros(reference_motion.shape[0], target_motion.shape[1] - reference_motion.shape[1], reference_motion.shape[2], device=reference_motion.device)], dim=1)

        return target_motion, target_lengths, reference_motion, reference_lengths

    def encode_motion_into_latent(self, motion, lengths,):
        if self.vae_type in ["mld", "vposert", "actor"]:
            lengths_z = [self.cfg.train.ABLATION.VAE_LATENT_NUMS] * len(lengths)
            motion_z, dist_m = self.vae.encode(motion, lengths, lengths_z)
            motion_z = motion_z.permute(1, 0, 2)
        elif self.vae_type in ["rfmotion_native"]:
            lengths_z = [math.ceil(l / self.cfg.train.ABLATION.VAE_LATENT_NUMS) for l in lengths]
            motion_z, dist_m = self.vae.encode(motion, lengths, lengths_z)
            motion_z = motion_z.permute(1, 0, 2)
        elif self.vae_type == "no":
            motion_z = motion
            lengths_z = lengths
            dist_m = None

        return motion_z, lengths_z, dist_m

    def decode_latent_into_motion(self, motion_z, lengths, lengths_z,):
        if self.vae_type in ["mld", "vposert", "actor"]:
            motion = self.vae.decode(motion_z, lengths, lengths_z)
        elif self.vae_type in ["rfmotion_native"]:
            motion = self.vae.decode(motion_z, lengths, lengths_z)
        elif self.vae_type == "no":
            motion = motion_z

        return motion

    def hint_mask(self, target_motion, target_motion_lens, prompt=None, n_joints=22, hint_type=None):
        if hint_type == None:
            choice = np.random.rand(1)
            if choice < 0.5:
                hint_type = "inbetween"
            else:
                hint_type = "trajectory"

        if hint_type == 'inbetween':
            specify_joints = torch.arange(n_joints, dtype=torch.long, device=target_motion.device)
            # controllable_joints_lens = torch.randint(1, math.ceil(target_motion_lens/2), (1,), device=target_motion.device)
            controllable_joints_lens = torch.randint(20, 21, (1,), device=target_motion.device)
            choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)
            choose_seq = choose_seq[torch.randperm(target_motion_lens)][:controllable_joints_lens]
            choose_seq = choose_seq.sort()[0]
            
        elif hint_type == "trajectory" :
            specify_joints = []
            if 'head' in prompt:
                specify_joints.append(15)
            if "sit" in prompt or "stand" in prompt or 'hips' in prompt or 'pelvis' in prompt or "walk"  in prompt or "run"  in prompt or "jog"  in prompt:
                specify_joints.append(0)
            if 'shoulder' in prompt or 'shoulders' in prompt:
                specify_joints.append(16)
                specify_joints.append(17)
            if 'left shoulder' in prompt:
                specify_joints.append(16)
            if 'right shoulder' in prompt:
                specify_joints.append(17)
            if 'hands' in prompt or 'arms' in prompt:
                specify_joints.append(20)
                specify_joints.append(21)
            if 'left hand' in prompt or 'left arm' in prompt or "left wrist" in prompt:
                specify_joints.append(20)
            if 'right hand' in prompt or 'right arm' in prompt or "right wrist" in prompt:
                specify_joints.append(21)
            if 'elbows' in prompt:
                specify_joints.append(18)
                specify_joints.append(19)   
            if 'left elbow' in prompt:
                specify_joints.append(18)
            if 'right elbow' in prompt:
                specify_joints.append(19)
            if 'legs' in prompt :
                specify_joints.append(4)
                specify_joints.append(5)
            if 'left leg' in prompt:
                specify_joints.append(4)
            if 'right leg' in prompt:  
                specify_joints.append(5)
            if "feet" in prompt:
                specify_joints.append(10)
                specify_joints.append(11)
            if 'left foot' in prompt:
                specify_joints.append(10)
            if 'right foot' in prompt:
                specify_joints.append(11)

            if len(specify_joints) > 0:
                specify_joints = torch.tensor(specify_joints, dtype=torch.long, device=target_motion.device)
                choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)
            elif len(specify_joints) == 0:
                specify_joints = self.random_joints[torch.randperm(6,device=target_motion.device)][:1]
                controllable_joints_lens = torch.randint(1, target_motion_lens, (1,), device=target_motion.device)
                choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)
                choose_seq = choose_seq[torch.randperm(target_motion_lens)][:controllable_joints_lens]
                choose_seq = choose_seq.sort()[0]

        elif hint_type == "random_joints":
            specify_joints = self.random_joints[torch.randperm(6,device=target_motion.device)][:1]
            controllable_joints_lens = torch.randint(1, target_motion_lens, (1,), device=target_motion.device)
            choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)
            choose_seq = choose_seq[torch.randperm(target_motion_lens)][:controllable_joints_lens]
            choose_seq = choose_seq.sort()[0]

        elif hint_type == "pelvis":
            specify_joints = torch.tensor([0], dtype=torch.long, device=target_motion.device)
            choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)

        mask1 = torch.zeros((1, target_motion.shape[1], n_joints, 3), dtype=torch.bool, device=target_motion.device)
        mask1[:, choose_seq, :, :] = True
        mask2 = torch.zeros((1, target_motion.shape[1], n_joints, 3), dtype=torch.bool, device=target_motion.device)
        mask2[:, :, specify_joints, :] = True
        hint_mask = mask1 & mask2

        encoder_hidden_state = target_motion
        encoder_hidden_state = encoder_hidden_state - self.datamodule.mean_motion 
        encoder_hidden_state = encoder_hidden_state / self.datamodule.std_motion
        encoder_hidden_state = encoder_hidden_state * hint_mask
        encoder_hidden_state = encoder_hidden_state.reshape(1, -1, n_joints * 3)
        encoder_hidden_state_mask = torch.zeros((1, target_motion.shape[1]), dtype=torch.bool, device=target_motion.device)
        encoder_hidden_state_mask[:, choose_seq] = True

        return encoder_hidden_state, encoder_hidden_state_mask, hint_mask

    def hint_guidance(self, t, noisy_latents, v_pred, encoder_hidden_states, hint_masks):
        noisy_latents_ = noisy_latents.detach().clone()
        scale = torch.tensor([t/self.cfg.model.scheduler.num_inference_steps], device=v_pred.device).detach()
        _, _, encoder_hidden_state = encoder_hidden_states.detach().chunk(3)

        batch_size = encoder_hidden_state.shape[0]
        seq_len = encoder_hidden_state.shape[1]
        encoder_hidden_state = encoder_hidden_state.reshape(batch_size, seq_len, 22, 3)

        keyframes = torch.sum(hint_masks, dim=-1).bool()
        num_keyframes = torch.sum(keyframes, dim=1)
        max_keyframes = torch.max(num_keyframes, dim=1).values
        strength = 20/max_keyframes
        strength = strength.unsqueeze(-1).unsqueeze(-1)

        with torch.enable_grad():
            x = noisy_latents_ - v_pred.clone() * scale
            x.requires_grad_(True)
            x_ = self.datamodule.feat2motion(x)
            loss = torch.norm((x_ - encoder_hidden_state) * hint_masks, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            x.detach()

        x = x - grad * strength
        v_pred = (noisy_latents_ - x)/scale
            
        return v_pred

    def get_motion_eval_emb(self, feat, lengths):
        motion = self.datamodule.feats2joints(feat)
        motion_ = self.datamodule.renorm4t2m(feat)

        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feat.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motion_ = motion_[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,self.cfg.DATASET.HUMANML3D.UNIT_LEN,rounding_mode="floor")

        motion_mov = self.t2m_moveencoder(motion_[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        return motion, motion_emb
    
    def get_text_eval_emb(self, batch, is_MM):
        text = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()

        if is_MM:
            text = text * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]
        return text, text_emb

    def train_paired_vae(self, batch):
        source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
        source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
        target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)
        source_motion_rst = self.decode_latent_into_motion(source_motion_z, source_lengths, source_lengths_z,)
        target_motion_rst = self.decode_latent_into_motion(target_motion_z, target_lengths, target_lengths_z,)

        if source_dist_m is not None:  # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(source_dist_m.loc)
            scale_ref = torch.ones_like(source_dist_m.scale)
            source_dist_ref = torch.distributions.Normal(mu_ref, scale_ref)

        if target_dist_m is not None:  # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(target_dist_m.loc)
            scale_ref = torch.ones_like(target_dist_m.scale)
            target_dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        
        rs_set = {
            "source_m_ref": source_motion,
            "source_m_rst": source_motion_rst,
            "source_dist_m": source_dist_m,
            "source_dist_ref": source_dist_ref,

            "target_m_ref": target_motion, 
            "target_m_rst": target_motion_rst,
            "target_dist_m": target_dist_m,
            "target_dist_ref": target_dist_ref,
        }
        return rs_set
    
    def eval_paired_vae(self,batch):
        source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
        with torch.no_grad():
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)
            source_motion_rst = self.decode_latent_into_motion(source_motion_z, source_lengths, source_lengths_z,)
            target_motion_rst = self.decode_latent_into_motion(target_motion_z, target_lengths, target_lengths_z,)

        rs_set = {
            ## for MotionReconstruction
            "source_m_ref": self.datamodule.feat2motion(source_motion).detach(),
            "source_m_rst": self.datamodule.feat2motion(source_motion_rst).detach(),
            "target_m_ref": self.datamodule.feat2motion(target_motion).detach(),
            "target_m_rst": self.datamodule.feat2motion(target_motion_rst).detach(),
            
            ## for UnCondition
            "source_motion": self.datamodule.feat2joint(source_motion).detach(),
            "source_motion_rst": self.datamodule.feat2joint(source_motion_rst).detach(),
            "target_motion": self.datamodule.feat2joint(target_motion).detach(),
            "target_motion_rst": self.datamodule.feat2joint(target_motion_rst).detach(),
        }
        return rs_set

    def train_rectified_flow_text(self, batch):
        text = batch["text"]
        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        for i in range(len(target_motion)):
            choice = np.random.rand(1)
            if choice < self.none_guidance_prob:
                encoder_hidden_states.append('')
                encoder_hidden_states_lengths.append(77)
            else:
                encoder_hidden_states.append(text[i])
                encoder_hidden_states_lengths.append(77)
        encoder_hidden_states = self.text_encoder(encoder_hidden_states)

        # diffusion process
        n_set = self.diffusion_process(text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_text_MM(self,batch):
        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, True)
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)
            target_motion_z = target_motion_z.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            target_lengths = target_lengths * self.cfg.TEST.MM_NUM_REPEATS
            target_lengths_z = target_lengths_z * self.cfg.TEST.MM_NUM_REPEATS

        # text encode
        encoder_hidden_states_lengths = [77] * len(text) * 2
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval",
                                                        text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

        # get eval embeddings
        generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)

        rs_set = { 
                   "target_emb_rst":generated_motion_emb, 
                   "length_target":target_lengths,
                   }
        return rs_set
    
    def eval_rectified_flow_text_T2M(self,batch):
        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, False)
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [77] * len(text) * 2
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # start
            start = time.time()
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval",
                                                        text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.text_times.append(end - start)
            self.text_samples += target_motion_z.shape[0]

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)


        # get eval embeddings
        generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "length_target":target_lengths, "text_emb":text_emb,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb, 
                   }
        return rs_set

    def train_rectified_flow_hint(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]

        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        input_text = []
        text_lengths = []
        input_hint = []
        hint_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            choice = np.random.rand(1)
            if choice < self.none_guidance_prob:
                input_text.append('')
                text_lengths.append(77)

                input_hint.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                hint_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))
            elif choice < self.none_guidance_prob + self.hint_guidance_prob:
                input_text.append('')
                text_lengths.append(77)

                encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="trajectory")
                input_hint.append(encoder_hidden_state)
                hint_lengths.append(encoder_hidden_state_mask)
                hint_masks.append(hint_mask)

            elif choice < self.none_guidance_prob + self.hint_guidance_prob + self.text_guidance_prob:
                input_text.append(text[i])
                text_lengths.append(77)

                input_hint.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                hint_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))

            else:
                input_text.append(text[i])
                text_lengths.append(77)

                encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="trajectory")
                input_hint.append(encoder_hidden_state)
                hint_lengths.append(encoder_hidden_state_mask)
                hint_masks.append(hint_mask)

        input_text = self.text_encoder(input_text)
        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(text=input_text, text_lengths=text_lengths,
                                       hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, False)
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        input_text = [""] * len(text) + text
        text_lengths = [77] * len(text) * 2
        input_hint = [torch.zeros((len(text), seq_len, 66), device=target_motion.device)] 
        hint_lengths = [torch.zeros((len(text), seq_len), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="trajectory")
            input_hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_state_mask)
            hint_masks.append(hint_mask)

        input_text = self.text_encoder(input_text)
        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        with torch.no_grad():
            # start
            start = time.time()
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval",
                                                        text=input_text, text_lengths=text_lengths,
                                                        hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.text_hint_times.append(end - start)
            self.text_hint_samples += target_motion_z.shape[0]

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

        # get eval embeddings
        generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "length_target":target_lengths, "text_emb":text_emb,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb, 
                   "target_motion_ref":target_motion, "target_motion_rst":generated_motion, "hint_masks":hint_masks,
                   }
        return rs_set

    def train_rectified_flow_source_text(self, batch):
        text = batch["text"]

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # padding and aligning
        if max(source_lengths_z) > max(target_lengths_z):
            target_motion_z = torch.cat([target_motion_z, torch.zeros(target_motion_z.shape[0], max(source_lengths_z) - max(target_lengths_z), target_motion_z.shape[2], device=target_motion_z.device)], dim=1)
        elif max(source_lengths_z) < max(target_lengths_z):
            source_motion_z = torch.cat([source_motion_z, torch.zeros(source_motion_z.shape[0], max(target_lengths_z) - max(source_lengths_z), source_motion_z.shape[2], device=source_motion_z.device)], dim=1)

        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        for i in range(len(text)):
            choice = np.random.rand(1)
            if choice < (self.none_guidance_prob/2): # drop all, 2source
                encoder_hidden_states.append("")
                encoder_hidden_states_lengths.append(77)

                target_motion_z[i] = source_motion_z[i].clone()
                target_lengths[i] = source_lengths[i]
                target_lengths_z[i] = source_lengths_z[i]

                source_lengths[i] = 0
                source_lengths_z[i] = 0

            elif choice < self.none_guidance_prob: # drop all, 2target
                encoder_hidden_states.append("")
                encoder_hidden_states_lengths.append(77)

                source_lengths[i] = 0
                source_lengths_z[i] = 0

            elif choice < (self.none_guidance_prob + self.source_guidance_prob/2): # drop text, source2source
                encoder_hidden_states.append("")
                encoder_hidden_states_lengths.append(77)
                target_motion_z[i] = source_motion_z[i].clone()
                target_lengths[i] = source_lengths[i]
                target_lengths_z[i] = source_lengths_z[i]

            elif choice < (self.none_guidance_prob + self.source_guidance_prob): # drop text, target2target
                encoder_hidden_states.append("")
                encoder_hidden_states_lengths.append(77)
                source_motion_z[i] = target_motion_z[i].clone()
                source_lengths[i] = target_lengths[i]
                source_lengths_z[i] = target_lengths_z[i]

            else:
                encoder_hidden_states.append(text[i])
                encoder_hidden_states_lengths.append(77)
        encoder_hidden_states = self.text_encoder(encoder_hidden_states)

        # diffusion process
        n_set = self.diffusion_process(text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths, 
                                       source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_source_text(self,batch):
        text = batch["text"]

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [77] * len(text) * 3
        text = [""]*len(text)*2 + text
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # start
            start = time.time()
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="eval",
                                                     text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths, 
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.source_text_times.append(end - start)
            self.source_text_samples += target_motion_z.shape[0]

            # decode motion
            edited_motion = self.decode_latent_into_motion(edited_motion_z, target_lengths, target_lengths_z,)


        # # for MotionFixMetrics
        # rs_set = { "length_source":source_lengths, "length_target":target_lengths,
        #            "source_motion_ref":self.datamodule.feat2joint(source_motion).detach(), "target_motion_ref":self.datamodule.feat2joint(target_motion).detach(), "target_motion_rst":self.datamodule.feat2joint(edited_motion).detach(),}

        # for SourceTextMetrics
        edited_motion, edited_motion_emb = self.get_motion_eval_emb(edited_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)
        source_motion, source_motion_emb = self.get_motion_eval_emb(source_motion, source_lengths)
        rs_set = { "source_emb_ref":source_motion_emb,
                   "target_emb_ref":target_motion_emb, 
                   "target_emb_rst":edited_motion_emb,}

        return rs_set

    def train_rectified_flow_source_hint(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)

            # padding and aligning
            if source_motion.shape[1] > target_motion.shape[1]:
                target_motion = torch.cat([target_motion, torch.zeros(target_motion.shape[0], source_motion.shape[1] - target_motion.shape[1], target_motion.shape[2], device=target_motion.device)], dim=1)
            elif source_motion.shape[1] < target_motion.shape[1]:
                source_motion = torch.cat([source_motion, torch.zeros(source_motion.shape[0], target_motion.shape[1] - source_motion.shape[1], source_motion.shape[2], device=source_motion.device)], dim=1)

            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

            # padding and aligning
            if source_motion_z.shape[1] > target_motion_z.shape[1]:
                target_motion_z = torch.cat([target_motion_z, torch.zeros(target_motion_z.shape[0], source_motion_z.shape[1] - target_motion_z.shape[1], target_motion_z.shape[2], device=target_motion_z.device)], dim=1)
            elif source_motion_z.shape[1] < target_motion_z.shape[1]:
                source_motion_z = torch.cat([source_motion_z, torch.zeros(source_motion_z.shape[0], target_motion_z.shape[1] - source_motion_z.shape[1], source_motion_z.shape[2], device=source_motion_z.device)], dim=1)

        seq_len = target_motion.shape[1]
        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        hint_masks = []
        for i in range(len(text)):
            prompt = text[i]
            choice = np.random.rand(1)
            if choice < (self.none_guidance_prob/2): # drop all, 2source
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                target_motion_z[i] = source_motion_z[i].clone()
                target_lengths[i] = source_lengths[i]
                target_lengths_z[i] = source_lengths_z[i]
                source_lengths[i] = 0
                source_lengths_z[i] = 0
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))

            elif choice < self.none_guidance_prob: # drop all, 2target
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                source_lengths[i] = 0
                source_lengths_z[i] = 0
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))

            elif choice < (self.none_guidance_prob + self.source_guidance_prob/2): # drop hint, source2source
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                target_motion_z[i] = source_motion_z[i].clone()
                target_lengths[i] = source_lengths[i]
                target_lengths_z[i] = source_lengths_z[i]
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))

            elif choice < (self.none_guidance_prob + self.source_guidance_prob): # drop hint, target2target
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                source_motion_z[i] = target_motion_z[i].clone()
                source_lengths[i] = target_lengths[i]
                source_lengths_z[i] = target_lengths_z[i]
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))

            elif choice < (self.none_guidance_prob + self.source_guidance_prob + self.hint_guidance_prob/2): # drop source, 2maskedsource
                # encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feat2motion(source_motion[i].unsqueeze(0)).detach().to(source_motion.device), source_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(source_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_states.append(encoder_hidden_state)
                encoder_hidden_states_lengths.append(encoder_hidden_states_length)

                target_motion_z[i] = source_motion_z[i].clone()
                target_lengths[i] = source_lengths[i]
                target_lengths_z[i] = source_lengths_z[i]
                source_lengths[i] = 0
                source_lengths_z[i] = 0
                hint_masks.append(hint_mask)

            elif choice < (self.none_guidance_prob + self.source_guidance_prob + self.hint_guidance_prob): # drop source, 2maskedtarget
                # encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feat2motion(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_states.append(encoder_hidden_state)
                encoder_hidden_states_lengths.append(encoder_hidden_states_length)

                source_lengths[i] = 0
                source_lengths_z[i] = 0
                hint_masks.append(hint_mask)

            else: # with source and hint
                # encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feat2motion(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
                encoder_hidden_states.append(encoder_hidden_state)
                encoder_hidden_states_lengths.append(encoder_hidden_states_length)
                hint_masks.append(hint_mask)

        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                       source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_source_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text)*2, target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text)*2, target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            prompt = text[i]
            # encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feat2motion(target_motion[i].unsqueeze(0)).to(target_motion.device).detach(), target_lengths[i], prompt, hint_type='trajectory')
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).to(target_motion.device).detach(), target_lengths[i], prompt, hint_type='trajectory')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        with torch.no_grad():
            # start
            start = time.time()
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="eval",
                                                     hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            
            # end time
            end = time.time()
            self.source_hint_times.append(end - start)
            self.source_hint_samples += target_motion_z.shape[0]

            # decode motion
            edited_motion = self.decode_latent_into_motion(edited_motion_z, target_lengths, target_lengths_z,)


        # for MotionFixHintMetrics
        # rs_set = { 
        #            "length_source":source_lengths, "length_target":target_lengths,
        #            "source_joint_ref":self.datamodule.feat2joint(source_motion).detach(), "target_joint_ref":self.datamodule.feat2joint(target_motion).detach(), "target_joint_rst":self.datamodule.feat2joint(edited_motion).detach(),
        #            "target_motion_ref":self.datamodule.feat2motion(target_motion).detach(), "target_motion_rst":self.datamodule.feat2motion(edited_motion).detach(), "hint_masks": hint_masks,
        #            }

        # for SourceHintMetrics
        edited_motion, edited_motion_emb = self.get_motion_eval_emb(edited_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)
        source_motion, source_motion_emb = self.get_motion_eval_emb(source_motion, source_lengths)
        rs_set = { 
                   "target_emb_ref":target_motion_emb, "target_emb_rst":edited_motion_emb, "source_emb_ref":source_motion_emb,
                   "target_motion_ref":target_motion, "target_motion_rst":edited_motion, "hint_masks":hint_masks,
                   }
        return rs_set

    def train_rectified_flow_inbetween(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        text = batch["text"]
        text_lengths = [77] * len(text)
        seq_len = target_motion.shape[1]
        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            choice = np.random.rand(1)
            if choice < 0.1:
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))
                text_lengths[i] = 0
            elif choice < 0.2:
                encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
                encoder_hidden_states.append(encoder_hidden_state)
                encoder_hidden_states_lengths.append(encoder_hidden_states_length)
                hint_masks.append(hint_mask)
                text_lengths[i] = 0
            elif choice < 0.3:
                encoder_hidden_states.append(torch.zeros((1, seq_len, 66), device=target_motion.device))
                encoder_hidden_states_lengths.append(torch.zeros((1, seq_len), device=target_motion.device))
                hint_masks.append(torch.zeros((1, seq_len, 22, 3), dtype=torch.bool, device=target_motion.device))
            else:
                encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
                encoder_hidden_states.append(encoder_hidden_state)
                encoder_hidden_states_lengths.append(encoder_hidden_states_length)
                hint_masks.append(hint_mask)

        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)
        text = self.text_encoder(text)

        # diffusion process
        n_set = self.diffusion_process(text=text, text_lengths=text_lengths,
                                       hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}
    
    def eval_rectified_flow_inbetween(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, False)
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        input_text = [""]*len(text) + text
        input_text = self.text_encoder(input_text)
        text_lengths = [0] * len(text) + [77] * len(text)

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        with torch.no_grad():
            # start
            start = time.time()
            # inbetween motion
            inbetween_motion_z = self.diffusion_reverse(stage="eval",
                                                        text=input_text, text_lengths=text_lengths,
                                                        hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.text_inbetween_times.append(end - start)
            self.text_inbetween_samples += target_motion_z.shape[0]

            # decode motion
            inbetween_motion = self.decode_latent_into_motion(inbetween_motion_z, target_lengths, target_lengths_z,)

        # get eval embeddings
        inbetween_motion, inbetween_motion_emb = self.get_motion_eval_emb(inbetween_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "length_target":target_lengths, "hint_masks": hint_masks, "text_emb":text_emb,
                   "target_motion_ref":target_motion, "target_motion_rst":inbetween_motion,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":inbetween_motion_emb, 
                   }
        return rs_set
    
    def train_rectified_flow_style(self, batch):
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        self.datamodule.mean = self.datamodule.mean.to(self.device)
        self.datamodule.std = self.datamodule.std.to(self.device)

        with torch.no_grad():
            motion, lengths = self.get_motion_batch(batch) 
            
            motion_for_hint = motion.clone()
            motion_for_hint = self.datamodule.feats2joints(motion_for_hint).reshape(motion_for_hint.shape[0], motion_for_hint.shape[1], 66)
            hint = torch.zeros(motion.shape[0], motion.shape[1], 66, device=motion.device)
            hint[..., :3] = motion_for_hint[..., :3].clone()
            hint = hint - self.datamodule.mean_motion.reshape(-1)
            hint = hint / self.datamodule.std_motion.reshape(-1)
            hint_lengths = lengths_to_mask(lengths, device=motion.device)

            content = motion.clone()
            content[..., :3] = 0
            content, _ = self.content_encoder.encode(content, lengths)
            content = content.permute(1,0,2)
            content_lengths = [7] * len(lengths)

            motion_seq = motion.clone() * self.datamodule.std + self.datamodule.mean
            motion_seq[...,:3]= 0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
            style = self.style_encoder.encoder({'x': motion_seq,
                                                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                                                'mask': lengths_to_mask(lengths, device=motion_seq.device)})["mu"].unsqueeze(1)
            style_lengths = [1] * len(lengths)

            for i in range(len(style)):
                choice = np.random.rand(1)
                if choice < self.none_guidance_prob:
                    hint_lengths[i, :] = False
                    style_lengths[i] = 0
                    content_lengths[i] = 0
                elif choice < self.none_guidance_prob + self.drop_hint_guidance_prob:
                    hint_lengths[i, :] = False
                elif choice < self.none_guidance_prob + self.drop_hint_guidance_prob + self.drop_style_guidance_prob:
                    style_lengths[i] = 0
                elif choice < self.none_guidance_prob + self.drop_hint_guidance_prob + self.drop_style_guidance_prob + self.drop_content_guidance_prob:
                    content_lengths[i] = 0
                

        # diffusion process
        n_set = self.diffusion_process(
                                       style=style, style_lengths=style_lengths,
                                       content=content, content_lengths = content_lengths,
                                       hint = hint, hint_lengths = hint_lengths,
                                       target_motion=motion, target_lengths=lengths, target_lengths_z=lengths,)
        return {**n_set}
    
    def eval_rectified_flow_style(self, batch):
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        self.datamodule.mean = self.datamodule.mean.to(self.device)
        self.datamodule.std = self.datamodule.std.to(self.device)

        # pick up 32 test samples
        content_samples = list(range(self.test_content_feats.shape[0]))
        style_samples = list(range(self.test_style_feats.shape[0]))
        all_combinations = list(itertools.product(content_samples, style_samples))
        selected_combinations = random.sample(all_combinations, 32)
        content_samples = [int(x[0]) for x in selected_combinations]
        style_samples = [int(x[1]) for x in selected_combinations]
        content_motion = self.test_content_feats[content_samples].clone().to(self.device)
        content_lengths = [self.test_content_lengths[i] for i in content_samples]
        style_motion = self.test_style_feats[style_samples].clone().to(self.device)
        style_lengths = [self.test_style_lengths[i] for i in style_samples]

        # motion encode
        with torch.no_grad():
            motion_for_hint = content_motion.clone()
            motion_for_hint = self.datamodule.feats2joints(motion_for_hint).reshape(motion_for_hint.shape[0], motion_for_hint.shape[1], 66)
            hint = torch.zeros(content_motion.shape[0], content_motion.shape[1], 66, device=content_motion.device)
            hint[..., :3] = motion_for_hint[..., :3].clone()
            hint = hint - self.datamodule.mean_motion.reshape(-1)
            hint = hint / self.datamodule.std_motion.reshape(-1)
            hint = hint.repeat(2,1,1) # for CFG
            hint_lengths = lengths_to_mask(content_lengths, device=content_motion.device, max_len=content_motion.shape[1]).repeat(2,1)

            content = content_motion.clone()
            content[..., :3] = 0
            content, _ = self.content_encoder.encode(content, content_lengths)
            content = content.permute(1,0,2).repeat(2,1,1)

            motion_seq = style_motion * self.datamodule.std + self.datamodule.mean
            motion_seq[...,:3]= 0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
            style = self.style_encoder.encoder({'x': motion_seq,
                                                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                                                'mask': lengths_to_mask(style_lengths, device=motion_seq.device)})["mu"].unsqueeze(1)
            style = style.repeat(2,1,1)

            # start
            start = time.time()
            # generate motion
            generated_motion = self.diffusion_reverse(stage="eval",
                                                        style=style, style_lengths=[0] * len(style_lengths) + [1] * len(style_lengths),
                                                        content=content, content_lengths = [7] * len(content_lengths) * 2,
                                                        hint = hint, hint_lengths = hint_lengths,
                                                        target_motion=content_motion, target_lengths=content_lengths, target_lengths_z=content_lengths,)
            # end time
            end = time.time()
            self.style_times.append(end - start)
            self.style_samples += content_motion.shape[0]

        # get eval embeddings
        generated_motion_ = generated_motion.clone()
        generated_motion_ = generated_motion_ * self.datamodule.std + self.datamodule.mean
        generated_motion_ = generated_motion_.unsqueeze(-1).permute(0,2,3,1)
        style_rst = self.style_encoder.encoder({'x': generated_motion_,
                                        'y': torch.zeros(generated_motion_.shape[0], dtype=int, device=generated_motion_.device),
                                        'mask': lengths_to_mask(content_lengths, device=generated_motion_.device)})["mu"].unsqueeze(1)
        style_ref = style.chunk(2)[0]

        content_motion, content_ref = self.get_motion_eval_emb(content_motion, content_lengths)
        generated_motion, content_rst = self.get_motion_eval_emb(generated_motion, content_lengths)

        hint_masks = torch.zeros((content_motion.shape[0], content_motion.shape[1], 22, 3), dtype=torch.bool, device=content_motion.device)
        for i in range(hint_masks.shape[0]):
            hint_masks[i, :content_lengths[i], :, :] = True

        rs_set = {"content_lengths":content_lengths,
                  "content_ref":content_ref, "content_rst":content_rst,
                  "style_ref":style_ref, "style_rst":style_rst,
                  "target_motion_rst":generated_motion, "target_motion_ref": content_motion, "hint_masks": hint_masks,
                   }
        return rs_set

    def demo_text(self,batch):
        text = batch["text"]
        
        # motion encode
        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [77] * len(text) * 2
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="demo",
                                                        text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

        rs_set = { 
                   "length_target":target_lengths,
                   "target_motion": self.datamodule.feats2joints(target_motion)[0].detach().cpu(),
                   "generated_motion":self.datamodule.feats2joints(generated_motion)[0],
                   }
        return rs_set

    def demo_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]
        
        # motion encode
        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        input_text = [""] * len(text) + text
        text_lengths = [77] * len(text) * 2

        # hint encode
        hint=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        hint_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type='trajectory')
            hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)
        hint = torch.cat(hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)
        input_text = self.text_encoder(input_text)

        with torch.no_grad():
            # edit motion
            generated_motion_z = self.diffusion_reverse(stage="demo",
                                                        text=input_text, text_lengths=text_lengths,
                                                        hint=hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

        rs_set = { 
                   "length_target":target_lengths, "hint_masks": hint_masks[0].detach().cpu(),
                   "target_motion": self.datamodule.feats2joints(target_motion)[0].detach().cpu(),
                   "generated_motion":self.datamodule.feats2joints(generated_motion)[0].detach().cpu(),
                   }
        return rs_set

    def demo_source_text(self, batch):
        # the batch is 1
        text = batch["text"]
        if type(text) == str:
            text = [text]
        elif type(text) == list:
            text = text

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_one(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [77] * len(text) * 3
        text = [""]*len(text)*2 + text
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="demo",
                                                     text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths, 
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # decode motion
            edited_motion = self.decode_latent_into_motion(edited_motion_z, target_lengths, target_lengths_z,)

        # rs_set = { 
        #           "source_motion":self.datamodule.feat2joint(source_motion)[0,:batch['length_source'],:].detach().cpu(), 
        #           "target_motion":self.datamodule.feat2joint(target_motion)[0,:batch['length_target'],:].detach().cpu(), 
        #           "edited_motion":self.datamodule.feat2joint(edited_motion)[0,:batch['length_target'],:].detach().cpu(),}

        rs_set = { 
                  "source_motion":self.datamodule.feats2joints(source_motion)[0].detach().cpu(), 
                  "target_motion":self.datamodule.feats2joints(target_motion)[0].detach().cpu(), 
                  "edited_motion":self.datamodule.feats2joints(edited_motion)[0].detach().cpu(),}
        return rs_set

    def demo_source_hint(self, batch):
        # the batch is 1
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]
        if type(text) == str:
            text = [text]
        elif type(text) == list:
            text = text

        # motion encode
        with torch.no_grad():
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_one(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text)*2, target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text)*2, target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        prompt = text[0]
        encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[0].unsqueeze(0)).to(target_motion.device).detach(), target_lengths[0], prompt, hint_type='trajectory')
        
        encoder_hidden_states.append(encoder_hidden_state)
        encoder_hidden_states_lengths.append(encoder_hidden_states_length)
        hint_masks.append(hint_mask)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0).to(target_motion.device)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0).to(target_motion.device)
        hint_masks = torch.cat(hint_masks, dim=0).to(target_motion.device)

        with torch.no_grad():
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="demo",
                                                     hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # decode motion
            edited_motion = self.decode_latent_into_motion(edited_motion_z, target_lengths, target_lengths_z,)

        # print(source_motion.shape, target_motion.shape, edited_motion.shape) # [1, frames, 207]
        # rs_set = {"hint": self.datamodule.feat2motion(target_motion)[0,:batch['length_target'],:].detach().cpu(), 
        #           "hint_masks":hint_masks[0,:batch['length_target'],:].detach().cpu(),
        #           "source_motion":self.datamodule.feat2joint(source_motion)[0,:batch['length_source'],:].detach().cpu(), 
        #           "target_motion":self.datamodule.feat2joint(target_motion)[0,:batch['length_target'],:].detach().cpu(), 
        #           "edited_motion":self.datamodule.feat2joint(edited_motion)[0,:batch['length_target'],:].detach().cpu(),
        #         }
        rs_set = {"hint": self.datamodule.feats2joints(target_motion)[0].detach().cpu(), 
                  "hint_masks":hint_masks[0].detach().cpu(),
                  "source_motion":self.datamodule.feats2joints(source_motion)[0].detach().cpu(), 
                  "target_motion":self.datamodule.feats2joints(target_motion)[0].detach().cpu(), 
                  "edited_motion":self.datamodule.feats2joints(edited_motion)[0].detach().cpu(),
                }
        return rs_set

    def demo_inbetween(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        text = batch["text"]
        
        # motion encode
        with torch.no_grad():
            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []
        # input_text = [""]*len(text) + text
        # text_lengths = [0] * len(text) + [77] * len(text)
        input_text = text + text
        text_lengths = [77] * len(text) * 2

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)
        input_text = self.text_encoder(input_text)


        with torch.no_grad():
            # edit motion
            inbetween_motion_z = self.diffusion_reverse(stage="demo",
                                                        text=input_text, text_lengths=text_lengths,
                                                        hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)

            # decode motion
            inbetween_motion = self.decode_latent_into_motion(inbetween_motion_z, target_lengths, target_lengths_z,)

        rs_set = { 
                   "length_target":target_lengths, "hint_masks": hint_masks[0].detach().cpu(),
                   "target_motion": self.datamodule.feats2joints(target_motion)[0].detach().cpu(),
                   "inbetween_motion":self.datamodule.feats2joints(inbetween_motion)[0].detach().cpu(),
                   }
        return rs_set

    def demo_style(self, content_i, style_j):
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        self.datamodule.mean = self.datamodule.mean.to(self.device)
        self.datamodule.std = self.datamodule.std.to(self.device)

        # pick up 32 test samples
        content_motion = self.test_content_feats[content_i].clone().to(self.device).unsqueeze(0)
        content_lengths = [self.test_content_lengths[content_i]]
        style_motion = self.test_style_feats[style_j].clone().to(self.device).unsqueeze(0)
        style_lengths = [self.test_style_lengths[style_j]]

        # motion encode
        with torch.no_grad():
            motion_for_hint = content_motion.clone()
            motion_for_hint = self.datamodule.feats2joints(motion_for_hint).reshape(motion_for_hint.shape[0], motion_for_hint.shape[1], 66)
            hint = torch.zeros(content_motion.shape[0], content_motion.shape[1], 66, device=content_motion.device)
            hint[..., :3] = motion_for_hint[..., :3].clone()
            hint = hint - self.datamodule.mean_motion.reshape(-1)
            hint = hint / self.datamodule.std_motion.reshape(-1)
            hint = hint.repeat(2,1,1) # for CFG
            hint_lengths = lengths_to_mask(content_lengths, device=content_motion.device, max_len=content_motion.shape[1]).repeat(2,1)

            content = content_motion.clone()
            content[..., :3] = 0
            content, _ = self.content_encoder.encode(content, content_lengths)
            content = content.permute(1,0,2).repeat(2,1,1)

            motion_seq = style_motion * self.datamodule.std + self.datamodule.mean
            motion_seq[...,:3]= 0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
            style = self.style_encoder.encoder({'x': motion_seq,
                                                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                                                'mask': lengths_to_mask(style_lengths, device=motion_seq.device)})["mu"].unsqueeze(1)
            style = style.repeat(2,1,1)

            # generate motion
            generated_motion = self.diffusion_reverse(stage="demo",
                                                        style=style, style_lengths=[0] * len(style_lengths) + [1] * len(style_lengths),
                                                        content=content, content_lengths = [7] * len(content_lengths) * 2,
                                                        hint = hint, hint_lengths = hint_lengths,
                                                        target_motion=content_motion, target_lengths=content_lengths, target_lengths_z=content_lengths,)

        content_motion, _ = self.get_motion_eval_emb(content_motion, content_lengths)
        generated_motion, _ = self.get_motion_eval_emb(generated_motion, content_lengths)
        style_motion, _ = self.get_motion_eval_emb(style_motion, style_lengths)
        hint_masks = torch.zeros((content_motion.shape[0], content_motion.shape[1], 22, 3), dtype=torch.bool, device=content_motion.device)
        for i in range(hint_masks.shape[0]):
            hint_masks[i, :content_lengths[i], :, :] = True

        rs_set = {"content_lengths":content_lengths, "hint_masks": hint_masks[0].detach().cpu(),
                  "generated_motion":generated_motion[0, :content_lengths[0], ...].detach().cpu(), "content_motion": content_motion[0, :content_lengths[0], ...].detach().cpu(), "style_motion": style_motion[0, :style_lengths[0], ...].detach().cpu(),
                   }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split == "train" or split == "val":
            if self.stage == "vae":
                rs_set = self.train_paired_vae(batch)
            elif self.stage == "diffusion":
                if self.condition_type == "text":
                    rs_set = self.train_rectified_flow_text(batch)
                elif self.condition_type == "hint":
                    rs_set = self.train_rectified_flow_hint(batch)
                elif self.condition_type == "source_text":
                    rs_set = self.train_rectified_flow_source_text(batch)
                elif self.condition_type == "source_hint":
                    rs_set = self.train_rectified_flow_source_hint(batch)
                elif self.condition_type == "inbetween":
                    rs_set = self.train_rectified_flow_inbetween(batch)
                elif self.condition_type == "style":
                    rs_set = self.train_rectified_flow_style(batch)
            loss = self.losses[split].update(rs_set)

        if split == "val" or split == "test":
            if self.stage == "vae":
                rs_set = self.eval_paired_vae(batch)
                for metric in self.metrics_dict:
                    if metric == "MRMetrics":
                        getattr(self, metric).update(batch["length_source"],rs_set["source_m_rst"],rs_set["source_m_ref"],
                                                    batch["length_target"],rs_set["target_m_rst"],rs_set["target_m_ref"],)
                    elif metric == "UncondSMPLPairedMetrics":
                        getattr(self, metric).update(batch["length_source"],rs_set["source_motion_rst"],rs_set["source_motion"],
                                                    batch["length_target"],rs_set["target_motion_rst"],rs_set["target_motion"],)

            elif self.stage == "diffusion":
                if self.condition_type == "text":
                    for metric in self.metrics_dict:
                        if metric == "TM2TMetrics":
                            rs_set = self.eval_rectified_flow_text_T2M(batch)
                            getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"])
                        if metric == "MMMetrics":
                            rs_set = self.eval_rectified_flow_text_MM(batch)
                            getattr(self, metric).update(rs_set['target_emb_rst'], rs_set["length_target"])

                elif self.condition_type == "hint":
                    rs_set = self.eval_rectified_flow_hint(batch)
                    for metric in self.metrics_dict:
                        if metric == "TextHintMetrics":
                            rs_set = self.eval_rectified_flow_hint(batch)
                            getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
                                                         rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])

                elif self.condition_type == "source_text":
                    for metric in self.metrics_dict:
                        if metric == "MotionFixMetrics":
                            rs_set = self.eval_rectified_flow_source_text(batch)
                            getattr(self, metric).update(rs_set["length_source"], rs_set["length_target"], rs_set['source_motion_ref'] ,rs_set['target_motion_ref'],rs_set['target_motion_rst'])
                        if metric == "SourceTextMetrics":
                            rs_set = self.eval_rectified_flow_source_text(batch)
                            getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'])

                elif self.condition_type == "source_hint":
                    for metric in self.metrics_dict:
                        if metric == "MotionFixHintMetrics":
                            rs_set = self.eval_rectified_flow_source_hint(batch)
                            getattr(self, metric).update(rs_set["length_source"], rs_set["length_target"], 
                                                         rs_set['source_joint_ref'] ,rs_set['target_joint_ref'],rs_set['target_joint_rst'],
                                                         rs_set['target_motion_ref'] ,rs_set['target_motion_rst'], rs_set['hint_masks'],
                                                         )
                        if metric == "SourceHintMetrics": 
                            rs_set = self.eval_rectified_flow_source_hint(batch)
                            getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
                                                 rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
                                                )
                            
                elif self.condition_type == "inbetween":
                    for metric in self.metrics_dict:
                        if metric == "TextInbetweenMetrics":
                            rs_set = self.eval_rectified_flow_inbetween(batch)
                            getattr(self, metric).update(rs_set["text_emb"],
                                                         rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"],
                                                         rs_set['target_motion_rst'], rs_set['target_motion_ref'], rs_set["hint_masks"],
                                                         )
                
                elif self.condition_type == "style":
                    for metric in self.metrics_dict:
                        if self.style_samples >= 1200:
                            continue
                        if metric == "StyleMetrics":
                            rs_set = self.eval_rectified_flow_style(batch)
                            getattr(self, metric).update(rs_set["content_lengths"],
                                                         rs_set["content_ref"], rs_set["content_rst"], 
                                                         rs_set["style_ref"], rs_set["style_rst"], 
                                                         rs_set["target_motion_rst"], rs_set["target_motion_ref"], rs_set["hint_masks"])
        
        if split == "test":
            loss = None

        return loss