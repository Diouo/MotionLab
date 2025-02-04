import numpy as np
import torch
import time
import math
import itertools
import random

import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics import MetricCollection

from .base import BaseModel
from rfmotion.config import instantiate_from_config
from rfmotion.models.losses.rfmotion import RFMotionLosses
from rfmotion.models.modeltype.base import BaseModel
from rfmotion.models.operator.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from rfmotion.utils.temos_utils import lengths_to_mask, lengths_to_query_mask

class RFMOTION(BaseModel):
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
        self.configure_instructions()
        
        self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,params=self.parameters())
        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()
        self.get_t2m_evaluator(cfg)
        self._losses = MetricCollection({split: RFMotionLosses(vae=self.is_vae, mode="xyz", cfg=cfg, prediction_type=self.predict_type) for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

    def diffusion_reverse(self, stage, condition_type, instructions, 
                          target_motion, target_lengths, target_lengths_z,
                          source_motion=None, source_lengths=None, source_lengths_z=None, source_mask=None,
                          text=None, text_lengths=None,
                          hint=None, hint_lengths=None, hint_masks=None,
                          style=None, style_lengths=None, 
                            content=None, content_lengths=None,
                         ):
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
            if condition_type == "uncond" or condition_type == "masked" or condition_type == "recon":
                latent_model_input = noisy_latents
                target_lengths_input = target_lengths
                target_lengths_z_input = target_lengths_z

                source_motion_input = source_motion
                source_lengths_input = source_lengths
                source_lengths_z_input = source_lengths_z
            elif condition_type == "text" or condition_type=="hint" or condition_type == "text_hint" or condition_type == "inbetween" or condition_type == "text_inbetween":
                latent_model_input = torch.cat([noisy_latents] * 2)
                target_lengths_input = target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z

                source_motion_input = None
                source_lengths_input = None
                source_lengths_z_input = None
            elif condition_type == "style":
                latent_model_input = torch.cat([noisy_latents] *2)
                source_motion_input = None
                source_lengths_input = None
                source_lengths_z_input = None
                target_lengths_input = target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z
            elif condition_type == "source_text" or condition_type == "source_hint" or condition_type == "source_text_hint":
                latent_model_input = torch.cat([noisy_latents] * 3)
                target_lengths_input = target_lengths + target_lengths + target_lengths
                target_lengths_z_input = target_lengths_z + target_lengths_z + target_lengths_z

                source_motion_input = torch.cat([source_motion] * 3)
                source_lengths_input = [0] * len(source_lengths) + source_lengths + source_lengths
                source_lengths_z_input = [0] * len(source_lengths_z) + source_lengths_z + source_lengths_z

            v_pred = self.denoiser(instructions=instructions,
                                   hidden_states=latent_model_input,
                                   timestep=t,
                                   text=text,
                                   text_lengths=text_lengths,
                                   hint=hint,
                                   hint_lengths=hint_lengths,
                                   source_motion=source_motion_input,
                                   source_lengths=source_lengths_input,
                                   source_lengths_z=source_lengths_z_input,
                                   source_mask=source_mask,
                                   target_lengths=target_lengths_input,
                                   target_lengths_z=target_lengths_z_input,
                                   style=style, style_lengths=style_lengths,
                                   content=content, content_lengths=content_lengths,
                                   return_dict=False,)[0]

            # perform CFG guidance
            if condition_type == "uncond" or condition_type == "masked" or condition_type == "recon":
                v_pred = v_pred
            elif condition_type == "text":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.text_guidance_scale * (v_cond - v_uncond)
            elif condition_type == "hint":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.hint_guidance_scale * (v_cond - v_uncond)
            elif condition_type == "text_hint":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.text_hint_guidance_scale * (v_cond - v_uncond)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif condition_type == "inbetween":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.inbetween_guidance_scale * (v_cond - v_uncond)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif condition_type == "text_inbetween":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.text_inbetween_guidance_scale * (v_cond - v_uncond)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif condition_type=='source_text':
                v_pred_dropall, v_pred_with_source, v_pred_with_all = v_pred.chunk(3)
                v_pred = v_pred_dropall + self.source_text_guidance_scale_1 * (v_pred_with_source - v_pred_dropall) + self.source_text_guidance_scale_2 * (v_pred_with_all - v_pred_with_source)
            elif condition_type=='source_hint':
                v_pred_dropall, v_pred_with_source, v_pred_with_all = v_pred.chunk(3)
                v_pred = v_pred_dropall + self.source_hint_guidance_scale_1 * (v_pred_with_source - v_pred_dropall) + self.source_hint_guidance_scale_2 * (v_pred_with_all - v_pred_with_source)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif condition_type=='source_text_hint':
                v_pred_dropall, v_pred_with_source, v_pred_with_all = v_pred.chunk(3)
                v_pred = v_pred_dropall + self.source_text_hint_guidance_scale_1 * (v_pred_with_source - v_pred_dropall) + self.source_text_hint_guidance_scale_2 * (v_pred_with_all - v_pred_with_source)
                if self.is_hint_guidance:
                    v_pred = self.hint_guidance(t, noisy_latents, v_pred, hint, hint_masks)
            elif condition_type == "style":
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + self.style_guidance_scale * (v_cond - v_uncond)
            noisy_latents = self.scheduler.step(v_pred, t, noisy_latents, return_dict=False)[0]

        return noisy_latents

    def diffusion_process(self, instructions,
                          target_motion, target_lengths, target_lengths_z,
                          source_motion=None, source_lengths=None, source_lengths_z=None, source_mask=None,
                          text=None, text_lengths=None,
                          hint=None, hint_lengths=None, hint_masks=None,
                          style=None, style_lengths=None,
                          content=None, content_lengths=None,
                          ):

        # prepocess
        bsz = target_motion.shape[0]
        timesteps = torch.randint(0,self.noise_scheduler.config.num_train_timesteps+1,(bsz, ),device=target_motion.device,dtype=torch.long,) 

        # mid motion between source_motion adn target_motion
        noise = torch.randn_like(target_motion, device=target_motion.device)
        noisy_latents = self.noise_scheduler.scale_noise(sample=target_motion.clone(), noise=noise, timestep=timesteps) 

        # predicted result
        v_pred = self.denoiser(instructions=instructions,
                               hidden_states=noisy_latents,
                               timestep=timesteps,
                               text=text,
                               text_lengths=text_lengths,
                               hint=hint,
                               hint_lengths=hint_lengths,
                               source_motion=source_motion,
                               source_lengths=source_lengths,
                               source_lengths_z=source_lengths_z,
                               source_mask=source_mask,
                               target_lengths=target_lengths,
                               target_lengths_z=target_lengths_z,
                               style=style, style_lengths=style_lengths,
                               content=content, content_lengths=content_lengths,
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

    def get_paired_motion_batch(self, batch):
        source_motion = batch["source_motion"].to(self.device)
        target_motion = batch["target_motion"].to(self.device)  

        source_lengths = batch['length_source']
        max_source_lengths = source_motion.shape[1]
        source_lengths = [i if i < max_source_lengths else max_source_lengths for i in source_lengths]

        target_lengths = batch['length_target']
        max_target_lengths = target_motion.shape[1]
        target_lengths = [i if i < max_target_lengths else max_target_lengths for i in target_lengths]

        return source_motion, source_lengths, target_motion, target_lengths
    
    def get_triple_motion_batch(self, batch):
        motion = batch["motion"].to(self.device)
        lengths = batch["length"]

        # reference_motion = batch["reference_motion"].to(self.device)
        # reference_lengths = batch["reference_length"].detach().cpu().tolist()

        source_motion = batch["source_motion"].to(self.device)
        source_lengths = batch['length_source']
        max_source_lengths = source_motion.shape[1]
        source_lengths = [i if i < max_source_lengths else max_source_lengths for i in source_lengths]

        target_motion = batch["target_motion"].to(self.device)  
        target_lengths = batch['length_target']
        max_target_lengths = target_motion.shape[1]
        target_lengths = [i if i < max_target_lengths else max_target_lengths for i in target_lengths]

        max_seq_len = max([motion.shape[1], source_motion.shape[1], target_motion.shape[1]])
        if motion.shape[1] < max_seq_len:
            motion = torch.cat([motion, torch.zeros(motion.shape[0], max_seq_len - motion.shape[1], motion.shape[2], device=motion.device)], dim=1)
        # if reference_motion.shape[1] < max_seq_len:
        #     reference_motion = torch.cat([reference_motion, torch.zeros(reference_motion.shape[0], max_seq_len - reference_motion.shape[1], reference_motion.shape[2], device=reference_motion.device)], dim=1)
        if source_motion.shape[1] < max_seq_len:
            source_motion = torch.cat([source_motion, torch.zeros(source_motion.shape[0], max_seq_len - source_motion.shape[1], source_motion.shape[2], device=source_motion.device)], dim=1)
        if target_motion.shape[1] < max_seq_len:
            target_motion = torch.cat([target_motion, torch.zeros(target_motion.shape[0], max_seq_len - target_motion.shape[1], target_motion.shape[2], device=target_motion.device)], dim=1)   

        output_motion = []
        output_lengths = []
        # for i in range(len(lengths)):
        #     choice = np.random.rand(1)
        #     if choice < 23384/48625:
        #         output_motion.append(motion[i])
        #         output_lengths.append(lengths[i])
        #     elif choice < 23384/48625 + 14467/48625:
        #         output_motion.append(reference_motion[i])
        #         output_lengths.append(reference_lengths[i])
        #     elif choice < 23384/48625 + 14467/48625 + 5387/48625:
        #         output_motion.append(source_motion[i])
        #         output_lengths.append(source_lengths[i])
        #     else:
        #         output_motion.append(target_motion[i])
        #         output_lengths.append(target_lengths[i])

        for i in range(len(lengths)):
            choice = np.random.rand(1)
            if choice < 23384/34158:
                output_motion.append(motion[i])
                output_lengths.append(lengths[i])
            elif choice < 23384/34158 + 5387/34158:
                output_motion.append(source_motion[i])
                output_lengths.append(source_lengths[i])
            else:
                output_motion.append(target_motion[i])
                output_lengths.append(target_lengths[i])
        output_motion = torch.stack(output_motion, dim=0)

        return output_motion, output_lengths

    def encode_motion_into_latent(self, motion, lengths,):
        if self.vae_type in ["mld", "vposert", "actor"]:
            lengths_z = [self.cfg.TRAIN.ABLATION.VAE_LATENT_NUMS] * len(lengths)
            motion_z, dist_m = self.vae.encode(motion, lengths, lengths_z)
            motion_z = motion_z.permute(1, 0, 2)
        elif self.vae_type in ["rfmotion_native"]:
            lengths_z = [math.ceil(l / self.cfg.TRAIN.ABLATION.VAE_LATENT_NUMS) for l in lengths]
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
        if hint_type != "random_joints":
            if hint_type == 'inbetween':
                specify_joints = torch.arange(n_joints, dtype=torch.long, device=target_motion.device)
                controllable_joints_lens = torch.randint(1, math.ceil(target_motion_lens/2), (1,), device=target_motion.device)
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

            elif hint_type == "pelvis":
                specify_joints = torch.tensor([0], dtype=torch.long, device=target_motion.device)
                choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)

            mask1 = torch.zeros((1, target_motion.shape[1], n_joints, 3), dtype=torch.bool, device=target_motion.device)
            mask1[:, choose_seq, :, :] = True
            mask2 = torch.zeros((1, target_motion.shape[1], n_joints, 3), dtype=torch.bool, device=target_motion.device)
            mask2[:, :, specify_joints, :] = True
            hint_mask = mask1 & mask2

        else:
            # specify_joints = self.random_joints[torch.randperm(6,device=target_motion.device)][:1]
            # controllable_joints_lens = torch.randint(1, target_motion_lens, (1,), device=target_motion.device)
            # choose_seq = torch.arange(target_motion_lens, dtype=torch.long, device=target_motion.device)
            # choose_seq = choose_seq[torch.randperm(target_motion_lens)][:controllable_joints_lens]
            # choose_seq = choose_seq.sort()[0]
            hint_mask = torch.rand((1, target_motion.shape[1], n_joints), device=target_motion.device) > 0.5
            hint_mask[:, target_motion_lens:, :] = False
            choose_seq = hint_mask.sum(dim=2) > 0
            choose_seq = choose_seq.squeeze()
            hint_mask = hint_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

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
            x_ = self.datamodule.feats2joints(x)
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

    def train_rectified_flow_uncond(self, batch):
        with torch.no_grad():
            instructions = self.instructions["uncond"]
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions, target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}
    
    def eval_rectified_flow_uncond(self, batch):
        with torch.no_grad():
            instructions = self.instructions["uncond"].repeat(len(batch["text"]), 1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="uncond", instructions=instructions, 
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

            # get eval embeddings
            generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
            target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = {"target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb,}
        return rs_set
    
    def train_rectified_flow_masked(self, batch):
        # motion encode
        with torch.no_grad():
            instructions = self.instructions["masked"].repeat(len(batch["text"]), 1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        source_mask = []
        for i in range(len(target_motion)):
            controllable_joints_lens = torch.randint(0, target_lengths_z[i], (1,), device=target_motion.device)
            choose_seq = torch.arange(target_lengths_z[i], dtype=torch.long, device=target_motion.device)
            choose_seq = choose_seq[torch.randperm(target_lengths_z[i])][:controllable_joints_lens]
            choose_seq = choose_seq.sort()[0]

            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=target_motion.device)
            mask[:, choose_seq] = True
            source_mask.append(mask)
        source_mask = torch.cat(source_mask, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions, 
                                       source_motion=target_motion_z, source_lengths=target_lengths, source_lengths_z=target_lengths_z, source_mask=source_mask,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,
                                       )
        return {**n_set}
       
    def eval_rectified_flow_masked(self, batch):
        with torch.no_grad():
            text = batch["text"]
            instructions = self.instructions["masked"].repeat(len(text), 1)

            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

            seq_len = target_motion.shape[1]
            source_mask = []
            for i in range(len(target_motion)):
                controllable_joints_lens = torch.randint(0, target_lengths_z[i], (1,), device=target_motion.device)
                choose_seq = torch.arange(target_lengths_z[i], dtype=torch.long, device=target_motion.device)
                choose_seq = choose_seq[torch.randperm(target_lengths_z[i])][:controllable_joints_lens]
                choose_seq = choose_seq.sort()[0]

                mask = torch.zeros((1, seq_len), dtype=torch.bool, device=target_motion.device)
                mask[:, choose_seq] = True
                source_mask.append(mask)
            source_mask = torch.cat(source_mask, dim=0)

            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="masked", instructions=instructions, 
                                                        source_motion=target_motion_z, source_lengths=target_lengths, source_lengths_z=target_lengths_z, source_mask=source_mask,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,
                                                        )
            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)
            self.masked_times.append(len(text))

            # get eval embeddings
            generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
            target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "hint_masks":source_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 22, 3), 
                   "target_motion_ref":target_motion, "target_motion_rst":generated_motion,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb, 
                   }
        return rs_set

    def train_rectified_flow_recon(self, batch):
        # motion encode
        with torch.no_grad():
            instructions = self.instructions["recon"].repeat(len(batch["text"]), 1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       source_motion=target_motion_z, source_lengths=target_lengths, source_lengths_z=target_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}
       
    def eval_rectified_flow_recon(self, batch):
        with torch.no_grad():
            text = batch["text"]
            instructions = self.instructions["recon"].repeat(len(text), 1)

            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="recon", instructions=instructions, 
                                                        source_motion=target_motion_z, source_lengths=target_lengths, source_lengths_z=target_lengths_z,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

            # get eval embeddings
            hint_masks = lengths_to_mask(target_lengths, target_motion.device, max_len=generated_motion.shape[1]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 22, 3)
            generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
            target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "hint_masks":hint_masks, 
                   "target_motion_ref":target_motion, "target_motion_rst":generated_motion,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb, 
                   }
        return rs_set

    def train_rectified_flow_text(self, batch):
        with torch.no_grad():
            text = batch["text"]
            text_lengths = [77] * len(text)
            instructions = self.instructions["text"]
            text = self.text_encoder(text)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       text=text, text_lengths=text_lengths,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_text_MM(self,batch):
        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, True)

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)
            target_motion_z = target_motion_z.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            target_lengths = target_lengths * self.cfg.TEST.MM_NUM_REPEATS
            target_lengths_z = target_lengths_z * self.cfg.TEST.MM_NUM_REPEATS

        # text encode
        encoder_hidden_states_lengths = [0] * len(text) + [77] * len(text)
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="text", instructions=instructions, 
                                                        text=encoder_hidden_states, text_lengths=encoder_hidden_states_lengths,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,
                                                        )

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

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [0] * len(text) + [77] * len(text)
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # start
            start = time.time()
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="text", instructions=instructions, 
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

        with torch.no_grad():
            text = batch["text"]
            instructions = self.instructions["hint"].repeat(len(text),1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        input_hint = []
        hint_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="random_joints")
            input_hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_state_mask)
            hint_masks.append(hint_mask)

        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
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

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        input_hint = [torch.zeros((len(text), seq_len, 66), device=target_motion.device)] 
        hint_lengths = [torch.zeros((len(text), seq_len), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="random_joints")
            input_hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_state_mask)
            hint_masks.append(hint_mask)

        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        with torch.no_grad():
            # start
            start = time.time()
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="hint", instructions=instructions, 
                                                        hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.hint_times.append(end - start)
            self.hint_samples += target_motion_z.shape[0]

            # decode motion
            generated_motion = self.decode_latent_into_motion(generated_motion_z, target_lengths, target_lengths_z,)

        # get eval embeddings
        generated_motion, generated_motion_emb = self.get_motion_eval_emb(generated_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "length_target":target_lengths,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":generated_motion_emb, 
                   "target_motion_ref":target_motion, "target_motion_rst":generated_motion, "hint_masks":hint_masks,
                   }
        return rs_set

    def train_rectified_flow_text_hint(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        with torch.no_grad():
            text = batch["text"]
            instructions = self.instructions["text_hint"].repeat(len(text),1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        input_text =  text
        text_lengths = [77] * len(text)
        input_hint = []
        hint_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="trajectory")
            input_hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_state_mask)
            hint_masks.append(hint_mask)

        input_text = self.text_encoder(input_text)
        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       text=input_text, text_lengths=text_lengths,
                                       hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_text_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, False)

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        input_text = [""] * len(text) + text
        text_lengths = [0] * len(text) + [77] * len(text)
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
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="text_hint", instructions=instructions, 
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
        # motion encode
        with torch.no_grad():
            text = batch["edit_text"]
            instructions = self.instructions["source_text"].repeat(len(text), 1)
            text_lengths = [77] * len(text)
            text = self.text_encoder(text)

            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions, text=text, text_lengths=text_lengths, 
                                       source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_source_text(self,batch):
        # motion encode
        with torch.no_grad():
            text = batch["edit_text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_text"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [0] * len(text) * 2 + [77] * len(text)
        text = [""]*len(text)*2 + text
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # start
            start = time.time()
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="eval", condition_type="source_text", instructions=instructions, 
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

        # motion encode
        with torch.no_grad():
            instructions = self.instructions["source_hint"]
            text = batch["edit_text"]
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        hint_masks = []
        for i in range(len(text)):
            prompt = text[i]
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)

        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                       source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_source_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text = batch["edit_text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

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
            edited_motion_z = self.diffusion_reverse(stage="eval", condition_type="source_hint", instructions=instructions, 
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

    def train_rectified_flow_source_text_hint(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            instructions = self.instructions["source_text_hint"]
            text = batch["edit_text"]
            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        hint_masks = []
        for i in range(len(text)):
            prompt = text[i]
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt, hint_type='trajectory')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)

        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)
        input_text =  self.text_encoder(text)
        text_lengths = [77] * len(text)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       text=input_text, text_lengths=text_lengths,
                                       hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                       source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_source_text_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text = batch["edit_text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_text_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

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

        input_text =  [""] * len(text) * 2 + text
        input_text = self.text_encoder(input_text)
        text_lengths = [0] * len(text) * 2 + [77] * len(text)

        with torch.no_grad():
            # start
            start = time.time()
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="eval", condition_type="source_text_hint", instructions=instructions, 
                                                     text=input_text, text_lengths=text_lengths,
                                                     hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            
            # end time
            end = time.time()
            self.source_text_hint_times.append(end - start)
            self.source_text_hint_samples += target_motion_z.shape[0]

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
            instructions = self.instructions["inbetween"].repeat(len(batch["text"]), 1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        encoder_hidden_states = []
        encoder_hidden_states_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)

        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}
    
    def eval_rectified_flow_inbetween(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text = batch["text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["inbetween"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

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
            inbetween_motion_z = self.diffusion_reverse(stage="eval", condition_type="inbetween", instructions=instructions, 
                                                        hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.inbetween_times.append(end - start)
            self.inbetween_samples += target_motion_z.shape[0]

            # decode motion
            inbetween_motion = self.decode_latent_into_motion(inbetween_motion_z, target_lengths, target_lengths_z,)

        # get eval embeddings
        inbetween_motion, inbetween_motion_emb = self.get_motion_eval_emb(inbetween_motion, target_lengths)
        target_motion, target_motion_emb = self.get_motion_eval_emb(target_motion, target_lengths)

        rs_set = { 
                   "length_target":target_lengths, "hint_masks": hint_masks,
                   "target_motion_ref":target_motion, "target_motion_rst":inbetween_motion,
                   "target_emb_ref":target_motion_emb, "target_emb_rst":inbetween_motion_emb, 
                   }
        return rs_set
    
    def train_rectified_flow_text_inbetween(self, batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        with torch.no_grad():
            text = batch["text"]
            instructions = self.instructions["text_inbetween"].repeat(len(text),1)
            target_motion, target_lengths = self.get_triple_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        input_text =  text
        text_lengths = [77] * len(text)
        input_hint = []
        hint_lengths = []
        hint_masks = []

        for i in range(len(target_motion)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="inbetween")
            input_hint.append(encoder_hidden_state)
            hint_lengths.append(encoder_hidden_state_mask)
            hint_masks.append(hint_mask)

        input_text = self.text_encoder(input_text)
        input_hint = torch.cat(input_hint, dim=0)
        hint_lengths = torch.cat(hint_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
                                       text=input_text, text_lengths=text_lengths,
                                       hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                       target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
        return {**n_set}

    def eval_rectified_flow_text_inbetween(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text, text_emb = self.get_text_eval_emb(batch, False)

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text_inbetween"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        seq_len = target_motion.shape[1]
        input_text = [""] * len(text) + text
        text_lengths = [0] * len(text) + [77] * len(text)
        input_hint = [torch.zeros((len(text), seq_len, 66), device=target_motion.device)] 
        hint_lengths = [torch.zeros((len(text), seq_len), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_state_mask, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], prompt=text[i], hint_type="inbetween")
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
            generated_motion_z = self.diffusion_reverse(stage="eval", condition_type="text_inbetween", instructions=instructions, 
                                                        text=input_text, text_lengths=text_lengths,
                                                        hint=input_hint, hint_lengths=hint_lengths, hint_masks=hint_masks,
                                                        target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # end time
            end = time.time()
            self.text_inbetween_times.append(end - start)
            self.text_inbetween_samples += target_motion_z.shape[0]

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

    def train_rectified_flow_style(self, batch):
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        self.datamodule.mean = self.datamodule.mean.to(self.device)
        self.datamodule.std = self.datamodule.std.to(self.device)

        with torch.no_grad():
            instructions = []
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
                if choice <  self.drop_style_guidance_prob:
                    style_lengths[i] = 0
                    instructions.append(self.instructions["content"])
                elif choice < self.drop_style_guidance_prob + self.drop_content_guidance_prob:
                    content_lengths[i] = 0
                    instructions.append(self.instructions["style"])
                else:
                    instructions.append(self.instructions["style_content"])
            instructions = torch.cat(instructions, dim=0)

        # diffusion process
        n_set = self.diffusion_process(instructions=instructions,
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
            instruction1 = self.instructions["content"].repeat(len(style_motion), 1)
            instruction2 = self.instructions["style_content"].repeat(len(content_motion), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

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
            generated_motion = self.diffusion_reverse(stage="eval", instructions=instructions, condition_type="style",
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
        # motion encode
        with torch.no_grad():
            text = batch["text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [0] * len(text) + [77] * len(text)
        text = [""]*len(text) + text 
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # generate motion
            generated_motion_z = self.diffusion_reverse(stage="demo", condition_type="text", instructions=instructions, 
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
        
        # motion encode
        with torch.no_grad():
            text = batch["text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

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

        with torch.no_grad():
            # edit motion
            generated_motion_z = self.diffusion_reverse(stage="demo", condition_type="hint", instructions=instructions, 
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

    def demo_text_hint(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        
        # motion encode
        with torch.no_grad():
            text = batch["text"]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        input_text = [""] * len(text) + text
        text_lengths = [0] * len(text) + [77] * len(text)

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
            generated_motion_z = self.diffusion_reverse(stage="demo", condition_type="text_hint", instructions=instructions, 
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
        # motion encode
        with torch.no_grad():
            text = batch["edit_text"] if type(batch["edit_text"]) == list else [batch["edit_text"]]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_text"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
            source_motion_z, source_lengths_z, source_dist_m, = self.encode_motion_into_latent(source_motion, source_lengths,)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # text encode
        encoder_hidden_states_lengths = [0] * len(text) * 2 + [77] * len(text)
        text = [""]*len(text)*2 + text
        encoder_hidden_states = self.text_encoder(text)

        with torch.no_grad():
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="demo", condition_type="source_text", instructions=instructions, 
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

        # motion encode
        with torch.no_grad():
            text = batch["edit_text"] if type(batch["edit_text"]) == list else [batch["edit_text"]]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
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
            edited_motion_z = self.diffusion_reverse(stage="demo", condition_type="source_hint", instructions=instructions, 
                                                     hint=encoder_hidden_states, hint_lengths=encoder_hidden_states_lengths, hint_masks=hint_masks,
                                                     source_motion=source_motion_z, source_lengths=source_lengths, source_lengths_z=source_lengths_z,
                                                     target_motion=target_motion_z, target_lengths=target_lengths, target_lengths_z=target_lengths_z,)
            # decode motion
            edited_motion = self.decode_latent_into_motion(edited_motion_z, target_lengths, target_lengths_z,)

        rs_set = {"hint": self.datamodule.feats2joints(target_motion)[0].detach().cpu(), 
                  "hint_masks":hint_masks[0].detach().cpu(),
                  "source_motion":self.datamodule.feats2joints(source_motion)[0].detach().cpu(), 
                  "target_motion":self.datamodule.feats2joints(target_motion)[0].detach().cpu(), 
                  "edited_motion":self.datamodule.feats2joints(edited_motion)[0].detach().cpu(),
                }
        return rs_set

    def demo_source_text_hint(self, batch):
        # the batch is 1
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)

        # motion encode
        with torch.no_grad():
            text = batch["edit_text"] if type(batch["edit_text"]) == list else [batch["edit_text"]]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["recon"].repeat(len(text), 1)
            instruction3 = self.instructions["source_text_hint"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2, instruction3], dim=0)

            source_motion, source_lengths, target_motion, target_lengths = self.get_paired_motion_batch(batch)
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

        input_text =  [""] * len(text) * 2 + text
        input_text = self.text_encoder(input_text)
        text_lengths = [0] * len(text) * 2 + [77] * len(text)

        with torch.no_grad():
            # edit motion
            edited_motion_z = self.diffusion_reverse(stage="demo", condition_type="source_text_hint", instructions=instructions, 
                                                     text=input_text, text_lengths=text_lengths,
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
        
        # motion encode
        with torch.no_grad():
            text = batch["text"] if type(batch["text"]) == list else [batch["text"]]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["inbetween"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        for i in range(len(text)):
            encoder_hidden_state, encoder_hidden_states_length, hint_mask = self.hint_mask(self.datamodule.feats2joints(target_motion[i].unsqueeze(0)).detach().to(target_motion.device), target_lengths[i], hint_type='inbetween')
            encoder_hidden_states.append(encoder_hidden_state)
            encoder_hidden_states_lengths.append(encoder_hidden_states_length)
            hint_masks.append(hint_mask)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states_lengths = torch.cat(encoder_hidden_states_lengths, dim=0)
        hint_masks = torch.cat(hint_masks, dim=0)

        with torch.no_grad():
            # edit motion
            inbetween_motion_z = self.diffusion_reverse(stage="demo",  condition_type="inbetween", instructions=instructions,
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

    def demo_text_inbetween(self,batch):
        self.random_joints = self.random_joints.to(self.device)
        self.datamodule.mean_motion = self.datamodule.mean_motion.to(self.device)
        self.datamodule.std_motion = self.datamodule.std_motion.to(self.device)
        
        # motion encode
        with torch.no_grad():
            text = batch["text"] if type(batch["text"]) == list else [batch["text"]]

            instruction1 = self.instructions["uncond"].repeat(len(text), 1)
            instruction2 = self.instructions["text_inbetween"].repeat(len(text), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

            target_motion, target_lengths = self.get_motion_batch(batch)
            target_motion_z, target_lengths_z, target_dist_m, = self.encode_motion_into_latent(target_motion, target_lengths,)

        # hint encode
        encoder_hidden_states=[torch.zeros((len(text), target_motion.shape[1], 66), device=target_motion.device)]
        encoder_hidden_states_lengths=[torch.zeros((len(text), target_motion.shape[1]), device=target_motion.device)]
        hint_masks = []

        input_text =  text + text
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
            # edit motion
            inbetween_motion_z = self.diffusion_reverse(stage="demo",  condition_type="text_inbetween", instructions=instructions,
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
            instruction1 = self.instructions["content"].repeat(len(style_motion), 1)
            instruction2 = self.instructions["style_content"].repeat(len(content_motion), 1)
            instructions = torch.cat([instruction1, instruction2], dim=0)

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
            generated_motion = self.diffusion_reverse(stage="demo", instructions=instructions, condition_type="style",
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

    def get_tasks(self, task_list):
        task_FID = [abs(self.task_FID[task]-self.task_best_FID[task])/self.task_best_FID[task] for task in task_list]
        sum_FID = sum(task_FID)
        if sum_FID == 0:
            task_prob = [1/len(task_FID) for i in range(len(task_FID))]
        else:
            task_prob = [task_FID[i]/sum_FID for i in range(len(task_FID))]

        task = np.random.choice(task_list, p=task_prob)
        return task

    def train_tasks(self, task_name, batch):
        if task_name == "uncond":
            return self.train_rectified_flow_uncond(batch)
        elif task_name == "masked":
            return self.train_rectified_flow_masked(batch)
        elif task_name == "recon":
            return self.train_rectified_flow_recon(batch)
        elif task_name == "inbetween":
            return self.train_rectified_flow_inbetween(batch)
        elif task_name == "hint":
            return self.train_rectified_flow_hint(batch)
        
        elif task_name == "source_hint":
            return self.train_rectified_flow_source_hint(batch)
        elif task_name == "source_text_hint":
            return self.train_rectified_flow_source_text_hint(batch)
        elif task_name == "source_text":
            return self.train_rectified_flow_source_text(batch)
        elif task_name == "style":
            return self.train_rectified_flow_style(batch)

        elif task_name == "text_inbetween":
            return self.train_rectified_flow_text_inbetween(batch)
        elif task_name == "text_hint":
            return self.train_rectified_flow_text_hint(batch)
        elif task_name == "text":
            return self.train_rectified_flow_text(batch)

    # def allsplit_step(self, split: str, batch, batch_idx):
    #     if split == "train":
    #         if self.current_epoch < self.task_epoch[0]: # pretrain
    #             task_list = ["masked", "hint", "inbetween"]
    #             task = self.get_tasks(task_list)
    #             rs_set = self.train_tasks(task, batch)
    #         else:
    #             choice = np.random.rand(1)
    #             if choice < 0.05:
    #                 rs_set = self.train_rectified_flow_uncond(batch)
    #             elif choice < 0.1:
    #                 rs_set = self.train_rectified_flow_recon(batch)
    #             elif choice < 0.55: # old tasks
    #                 if self.current_epoch < self.task_epoch[1]: # text
    #                     task_list = ["inbetween", "hint",]
    #                 elif self.current_epoch < self.task_epoch[2]: # source_hint
    #                     task_list = ["inbetween", "hint", "text"]
    #                 elif self.current_epoch < self.task_epoch[3]: # source_text
    #                     task_list = ["inbetween", "hint", "text", "source_hint"]
    #                 elif self.current_epoch < self.task_epoch[4]: # text_inbetween&hint
    #                     task_list = ["inbetween", "hint", "text", "source_hint", "source_text"]
    #                 elif self.current_epoch < self.task_epoch[5]: # source_text_hint
    #                     task_list = ["inbetween", "hint", "text", "source_hint", "source_text", "text_inbetween", "text_hint"]
    #                 elif self.current_epoch < self.task_epoch[6]: # style
    #                     task_list = ["inbetween", "hint", "text",  "source_hint", "source_text", "text_inbetween", "text_hint", "source_text_hint"]
    #                 else: # depened on FID
    #                     task_list = ["inbetween", "hint", "text",  "source_hint", "source_text", "text_inbetween", "text_hint", "source_text_hint", "style"]
    #                 task = self.get_tasks(task_list)
    #                 rs_set = self.train_tasks(task, batch)
    #             else: # new tasks
    #                 if self.current_epoch < self.task_epoch[1]: # text
    #                     rs_set = self.train_rectified_flow_text(batch)
    #                 elif self.current_epoch < self.task_epoch[2]: # source_hint
    #                     rs_set = self.train_rectified_flow_source_hint(batch)
    #                 elif self.current_epoch < self.task_epoch[3]: # source_text
    #                     rs_set = self.train_rectified_flow_source_text(batch)
    #                 elif self.current_epoch < self.task_epoch[4]: # text_inbetween&hint
    #                     task_list = ["text_inbetween", "text_hint"]
    #                     task = self.get_tasks(task_list)
    #                     rs_set = self.train_tasks(task, batch)
    #                 elif self.current_epoch < self.task_epoch[6]: # source_text_hint
    #                     rs_set = self.train_rectified_flow_source_text_hint(batch)
    #                 elif self.current_epoch < self.task_epoch[7]: # style
    #                     rs_set = self.train_rectified_flow_style(batch)
    #                 else: # depened on FID
    #                     task_list = ["inbetween", "hint", "text", "source_hint", "source_text", "text_inbetween", "text_hint", "source_text_hint", "style",]
    #                     task = self.get_tasks(task_list)
    #                     rs_set = self.train_tasks(task, batch)
    #         loss = self.losses[split].update(rs_set)

    #     elif split == "val": 
    #         loss = None

    #         for metric in self.metrics_dict:
    #             if metric == "MaskedMetrics" and self.current_epoch <= self.task_epoch[0]:
    #                 rs_set = self.eval_rectified_flow_masked(batch)
    #                 getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
    #             if metric == "HintMetrics" :
    #                 rs_set = self.eval_rectified_flow_hint(batch)
    #                 getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], rs_set["length_target"],
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
    #             if metric == "InbetweenMetrics":
    #                 rs_set = self.eval_rectified_flow_inbetween(batch)
    #                 getattr(self, metric).update(rs_set["length_target"], rs_set["hint_masks"],
    #                                              rs_set['target_motion_ref'],rs_set['target_motion_rst'],
    #                                              rs_set['target_emb_ref'],rs_set['target_emb_rst'],)

    #             if metric == "TM2TMetrics" and self.current_epoch > self.task_epoch[0]:
    #                 rs_set = self.eval_rectified_flow_text_T2M(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"])

                
    #             if metric == "SourceHintMetrics" and self.current_epoch > self.task_epoch[1]:
    #                 if split == "val" and self.source_hint_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_hint_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_hint(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
    #                                              rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
    #                                             )
                    
                
    #             if metric == "SourceTextMetrics" and self.current_epoch > self.task_epoch[2]:
    #                 if split == "val" and self.source_text_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_text_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_text(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'])

    #             if metric == "TextInbetweenMetrics" and self.current_epoch > self.task_epoch[3]:
    #                 rs_set = self.eval_rectified_flow_text_inbetween(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    

    #             if metric == "TextHintMetrics" and self.current_epoch > self.task_epoch[3]:
    #                 rs_set = self.eval_rectified_flow_text_hint(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                                    
    #             if metric == "SourceTextHintMetrics" and self.current_epoch > self.task_epoch[4]:
    #                 if split == "val" and self.source_hint_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_hint_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_text_hint(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
    #                                              rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
    #                                             )
                    
                
    #             if metric == "StyleMetrics" and self.current_epoch > self.task_epoch[5] and self.style_samples < 1200:
    #                 rs_set = self.eval_rectified_flow_style(batch)
    #                 getattr(self, metric).update(rs_set["content_lengths"],
    #                                                 rs_set["content_ref"], rs_set["content_rst"], 
    #                                                 rs_set["style_ref"], rs_set["style_rst"], 
    #                                                 rs_set["target_motion_rst"], rs_set["target_motion_ref"], rs_set["hint_masks"])
                    
    #     elif split == "test": 
    #         loss = None

    #         for metric in self.metrics_dict:
    #             if metric == "MaskedMetrics":
    #                 rs_set = self.eval_rectified_flow_masked(batch)
    #                 getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
    #             if metric == "HintMetrics" :
    #                 rs_set = self.eval_rectified_flow_hint(batch)
    #                 getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], rs_set["length_target"],
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
    #             if metric == "InbetweenMetrics":
    #                 rs_set = self.eval_rectified_flow_inbetween(batch)
    #                 getattr(self, metric).update(rs_set["length_target"], rs_set["hint_masks"],
    #                                              rs_set['target_motion_ref'],rs_set['target_motion_rst'],
    #                                              rs_set['target_emb_ref'],rs_set['target_emb_rst'],)

    #             if metric == "TM2TMetrics":
    #                 rs_set = self.eval_rectified_flow_text_T2M(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"])

                
    #             if metric == "SourceHintMetrics":
    #                 if split == "val" and self.source_hint_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_hint_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_hint(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
    #                                              rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
    #                                             )
                    
                
    #             if metric == "SourceTextMetrics":
    #                 if split == "val" and self.source_text_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_text_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_text(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'])

    #             if metric == "TextInbetweenMetrics":
    #                 rs_set = self.eval_rectified_flow_text_inbetween(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    

    #             if metric == "TextHintMetrics":
    #                 rs_set = self.eval_rectified_flow_text_hint(batch)
    #                 getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
    #                                              rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                                    
    #             if metric == "SourceTextHintMetrics":
    #                 if split == "val" and self.source_hint_samples > 330:
    #                     continue
    #                 elif split == "test" and self.source_hint_samples > 1013:
    #                     continue
    #                 rs_set = self.eval_rectified_flow_source_text_hint(batch)
    #                 getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
    #                                              rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
    #                                             )
                    
                
    #             if metric == "StyleMetrics" and self.style_samples < 1200:
    #                 rs_set = self.eval_rectified_flow_style(batch)
    #                 getattr(self, metric).update(rs_set["content_lengths"],
    #                                                 rs_set["content_ref"], rs_set["content_rst"], 
    #                                                 rs_set["style_ref"], rs_set["style_rst"], 
    #                                                 rs_set["target_motion_rst"], rs_set["target_motion_ref"], rs_set["hint_masks"])

    #     return loss


    def allsplit_step(self, split: str, batch, batch_idx):
        if split == "train":
            if self.current_epoch < self.task_epoch[0]: # pretrain
                task_list = ["masked", "hint", "inbetween"]
                task = self.get_tasks(task_list)
                rs_set = self.train_tasks(task, batch)
            else:
                choice = np.random.rand(1)
                if choice < 0.05:
                    rs_set = self.train_rectified_flow_uncond(batch)
                elif choice < 0.1:
                    rs_set = self.train_rectified_flow_recon(batch)
                elif choice < 0.55: # old tasks
                    if self.current_epoch < self.task_epoch[1]: # text
                        task_list = ["inbetween", "hint",]
                    elif self.current_epoch < self.task_epoch[2]: # style
                        task_list = ["inbetween", "hint", "text"]
                    elif self.current_epoch < self.task_epoch[3]: # source_hint
                        task_list = ["inbetween", "hint", "text", "style"]
                    elif self.current_epoch < self.task_epoch[4]: # source_text
                        task_list = ["inbetween", "hint", "text", "style", "source_hint"]
                    elif self.current_epoch < self.task_epoch[5]: # text_inbetween&hint
                        task_list = ["inbetween", "hint", "text", "style", "source_hint", "source_text"]
                    elif self.current_epoch < self.task_epoch[6]: # source_text_hint
                        task_list = ["inbetween", "hint", "text", "style", "source_hint", "source_text", "text_inbetween", "text_hint"]
                    else: # depened on FID
                        task_list = ["inbetween", "hint", "text", "style", "source_hint", "source_text", "text_inbetween", "text_hint", "source_text_hint"]
                    task = self.get_tasks(task_list)
                    rs_set = self.train_tasks(task, batch)
                else: # new tasks
                    if self.current_epoch < self.task_epoch[1]: # text
                        rs_set = self.train_rectified_flow_text(batch)
                    elif self.current_epoch < self.task_epoch[2]: # style
                        rs_set = self.train_rectified_flow_style(batch)
                    elif self.current_epoch < self.task_epoch[3]: # source_hint
                        rs_set = self.train_rectified_flow_source_hint(batch)
                    elif self.current_epoch < self.task_epoch[4]: # source_text
                        rs_set = self.train_rectified_flow_source_text(batch)
                    elif self.current_epoch < self.task_epoch[5]: # text_inbetween&hint
                        task_list = ["text_inbetween", "text_hint"]
                        task = self.get_tasks(task_list)
                        rs_set = self.train_tasks(task, batch)
                    elif self.current_epoch < self.task_epoch[6]: # source_text_hint
                        rs_set = self.train_rectified_flow_source_text_hint(batch)
                    else: # depened on FID
                        task_list = ["inbetween", "hint", "text", "style", "source_hint", "source_text", "text_inbetween", "text_hint", "source_text_hint"]
                        task = self.get_tasks(task_list)
                        rs_set = self.train_tasks(task, batch)
            loss = self.losses[split].update(rs_set)


        elif split == "val":
            for metric in self.metrics_dict:
                if metric == "MaskedMetrics" and self.current_epoch <= self.task_epoch[0]:
                    rs_set = self.eval_rectified_flow_masked(batch)
                    getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                elif metric == "HintMetrics" :
                    rs_set = self.eval_rectified_flow_hint(batch)
                    getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], rs_set["length_target"],
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                elif metric == "InbetweenMetrics":
                    rs_set = self.eval_rectified_flow_inbetween(batch)
                    getattr(self, metric).update(rs_set["length_target"], rs_set["hint_masks"],
                                                 rs_set['target_motion_ref'],rs_set['target_motion_rst'],
                                                 rs_set['target_emb_ref'],rs_set['target_emb_rst'],)

                elif metric == "TM2TMetrics" and self.current_epoch > self.task_epoch[0]:
                    rs_set = self.eval_rectified_flow_text_T2M(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"])


                elif metric == "StyleMetrics" and self.current_epoch > self.task_epoch[1] and self.style_samples < 1200:
                    rs_set = self.eval_rectified_flow_style(batch)
                    getattr(self, metric).update(rs_set["content_lengths"],
                                                    rs_set["content_ref"], rs_set["content_rst"], 
                                                    rs_set["style_ref"], rs_set["style_rst"], 
                                                    rs_set["target_motion_rst"], rs_set["target_motion_ref"], rs_set["hint_masks"])
                    
                
                elif metric == "SourceHintMetrics" and self.current_epoch > self.task_epoch[2] and self.source_hint_samples < 330:
                    rs_set = self.eval_rectified_flow_source_hint(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
                                                 rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
                                                )
                    
                
                elif metric == "SourceTextMetrics" and self.current_epoch > self.task_epoch[3] and self.source_text_samples < 330:
                    rs_set = self.eval_rectified_flow_source_text(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'])

                elif metric == "TextInbetweenMetrics" and self.current_epoch >=self.task_epoch[4]:
                    rs_set = self.eval_rectified_flow_text_inbetween(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    

                elif metric == "TextHintMetrics" and self.current_epoch > self.task_epoch[4]:
                    rs_set = self.eval_rectified_flow_text_hint(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                                    
                elif metric == "SourceTextHintMetrics" and self.current_epoch > self.task_epoch[5] and self.source_text_hint_samples < 330:
                    rs_set = self.eval_rectified_flow_source_text_hint(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
                                                 rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
                                                )
                    
            loss = None


        elif split == "test":
            for metric in self.metrics_dict:
                if metric == "MaskedMetrics":
                    rs_set = self.eval_rectified_flow_masked(batch)
                    getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                elif metric == "HintMetrics" :
                    rs_set = self.eval_rectified_flow_hint(batch)
                    getattr(self, metric).update(rs_set['target_emb_ref'], rs_set['target_emb_rst'], rs_set["length_target"],
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                elif metric == "InbetweenMetrics":
                    rs_set = self.eval_rectified_flow_inbetween(batch)
                    getattr(self, metric).update(rs_set["length_target"], rs_set["hint_masks"],
                                                 rs_set['target_motion_ref'],rs_set['target_motion_rst'],
                                                 rs_set['target_emb_ref'],rs_set['target_emb_rst'],)

                elif metric == "TM2TMetrics":
                    rs_set = self.eval_rectified_flow_text_T2M(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"])


                elif metric == "StyleMetrics" and self.style_samples < 1200:
                    rs_set = self.eval_rectified_flow_style(batch)
                    getattr(self, metric).update(rs_set["content_lengths"],
                                                    rs_set["content_ref"], rs_set["content_rst"], 
                                                    rs_set["style_ref"], rs_set["style_rst"], 
                                                    rs_set["target_motion_rst"], rs_set["target_motion_ref"], rs_set["hint_masks"])
                    
                
                elif metric == "SourceHintMetrics" and self.source_hint_samples < 1013:
                    rs_set = self.eval_rectified_flow_source_hint(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
                                                 rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
                                                )
                    
                
                elif metric == "SourceTextMetrics" and self.source_text_samples< 1013:
                    rs_set = self.eval_rectified_flow_source_text(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'])


                elif metric == "TextInbetweenMetrics":
                    rs_set = self.eval_rectified_flow_text_inbetween(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    

                elif metric == "TextHintMetrics":
                    rs_set = self.eval_rectified_flow_text_hint(batch)
                    getattr(self, metric).update(rs_set["text_emb"], rs_set['target_emb_rst'], rs_set['target_emb_ref'], rs_set["length_target"], 
                                                 rs_set['target_motion_ref'], rs_set['target_motion_rst'], rs_set["hint_masks"])
                    
                                    
                elif metric == "SourceTextHintMetrics" and self.source_text_hint_samples < 1013:
                    rs_set = self.eval_rectified_flow_source_text_hint(batch)
                    getattr(self, metric).update(rs_set['source_emb_ref'] ,rs_set['target_emb_ref'],rs_set['target_emb_rst'],
                                                 rs_set["target_motion_ref"], rs_set["target_motion_rst"], rs_set['hint_masks']
                                                )
                    
            loss = None

        return loss
