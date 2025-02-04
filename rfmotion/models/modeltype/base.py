import os
import yaml
import numpy as np

from pathlib import Path
from collections import OrderedDict

import torch
from pytorch_lightning import LightningModule
from diffusers.optimization import get_scheduler

from rfmotion.models.architectures import t2m_motionenc,t2m_textenc
from rfmotion.models.metrics import ComputeMetrics, MRMetrics,  HUMANACTMetrics, UESTCMetrics, UncondSMPLPairedMetrics, MotionFixMetrics, MotionFixHintMetrics
from rfmotion.models.metrics import UncondMetrics, ReconMetrics, TM2TMetrics, MMMetrics, HintMetrics, SourceTextMetrics, SourceHintMetrics, InbetweenMetrics, StyleMetrics, MaskedMetrics, TextHintMetrics, TextInbetweenMetrics, SourceTextHintMetrics
from rfmotion.models.operator.style_encoder import StyleClassification
from rfmotion.utils.motionclip import get_model_and_data, load_model_wo_clip
from rfmotion.config import instantiate_from_config


class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.task_order = ["pretrain", "source_hint", "source_text_hint", "source_text", "style", "text_inbetween", "text_hint", "text"]
        # self.pretrain_epoch = 800
        # task_epoch = [0, 100, 200, 300, 500, 600, 700, 800]

        self.task_order = ["masked&hint&inbetween", "text", "style", "source_hint", "source_text", "text_inbetween&hint", "source_text_hint"]
        task_epoch = [1000, 300, 300, 200, 200, 200, 200]

        self.task_epoch = np.cumsum(task_epoch).tolist()
        self.task_FID = {
            "masked": 999,
            "hint": 999,
            "inbetween": 999,

            "source_hint": 999,
            "source_text_hint": 999,
            "source_text": 999,
            "style": 999,
            "text_inbetween": 999,
            "text_hint": 999,    
            "text": 999,
            }
        
        self.task_best_FID = {
            "masked": 999,
            "hint": 999,
            "inbetween": 999,

            "source_hint": 999,
            "source_text_hint": 999,
            "source_text": 999,
            "style": 999,
            "text_inbetween": 999,
            "text_hint": 999,    
            "text": 999,
            }

        self.text_times = []
        self.text_hint_times = []
        self.hint_times = []
        self.source_text_times = []
        self.source_text_hint_times = []
        self.source_hint_times = []
        self.inbetween_times = []
        self.text_inbetween_times = []
        self.style_times = []
        self.masked_times = []

        self.text_samples = 0
        self.text_hint_samples = 0
        self.hint_samples = 0
        self.source_text_samples = 0
        self.source_text_hint_samples = 0
        self.source_hint_samples = 0
        self.inbetween_samples = 0
        self.text_inbetween_samples = 0
        self.style_samples = 0

        self.random_joints = torch.tensor([0, 10, 11, 15, 20, 21]) 
        self.controllable_joints = torch.tensor([0, 4, 5, 10, 11, 15, 16, 17, 18, 19, 20, 21])
        # 0: pelvis, 
        # 4: left knee,
        # 5: right knee,
        # 10: left foot, 
        # 11: right foot, 
        # 15: head, 
        # 16: left shoulder,
        # 17: right shoulder,
        # 18: left elbow,
        # 19: right elbow,
        # 20: left hand, 
        # 21: right hand, 


    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"] and self.cfg.model.model_type == "rfmotion_seperate":
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })
            dico.update({
                "total/learning_rate": float(self.trainer.optimizers[0].param_groups[0]['lr']),
            })
        elif split == "train" and self.cfg.model.model_type == "rfmotion":
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })
            dico.update({
                "total/learning_rate": float(self.trainer.optimizers[0].param_groups[0]['lr']),
            })
        

        if split in ["val", "test"]:
            for metrics in self.metrics_dict:
                if metrics == "":
                    continue
            
                if metrics == "MaskedMetrics" and len(self.masked_times) > 0:
                    if self.task_best_FID["masked"] > self.task_FID["masked"]:
                        self.task_best_FID["masked"] = self.task_FID["masked"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    if not self.trainer.sanity_checking:
                        self.task_FID["masked"] = metrics_dict["FID"].item()
                    self.masked_times = []

                if metrics=="SourceTextMetrics" and len(self.source_text_times) > 0:
                    if self.task_best_FID["source_text"] > self.task_FID["source_text"]:
                        self.task_best_FID["source_text"] = self.task_FID["source_text"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"SourceTextMetrics/Infernce_Time": sum(self.source_text_times)/self.source_text_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["source_text"] = metrics_dict["FID"].item()
                    self.source_text_times = []
                    self.source_text_samples = 0

                if metrics=="SourceTextHintMetrics" and len(self.source_text_hint_times) > 0:
                    if self.task_best_FID["source_text_hint"] > self.task_FID["source_text_hint"]:
                        self.task_best_FID["source_text_hint"] = self.task_FID["source_text_hint"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"SourceTextHintMetrics/Infernce_Time": sum(self.source_text_hint_times)/self.source_text_hint_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["source_text_hint"] = metrics_dict["FID"].item()
                    self.source_text_hint_times = []
                    self.source_text_hint_samples = 0

                if metrics=="SourceHintMetrics" and len(self.source_hint_times) > 0:
                    if self.task_best_FID["source_hint"] > self.task_FID["source_hint"]:
                        self.task_best_FID["source_hint"] = self.task_FID["source_hint"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"SourceHintMetrics/Infernce_Time": sum(self.source_hint_times)/self.source_hint_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["source_hint"] = metrics_dict["FID"].item()
                    self.source_hint_times = []
                    self.source_hint_samples = 0

                if metrics=="InbetweenMetrics" and len(self.inbetween_times) > 0:
                    if self.task_best_FID["inbetween"] > self.task_FID["inbetween"]:
                        self.task_best_FID["inbetween"] = self.task_FID["inbetween"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"InbetweenMetrics/Infernce_Time": sum(self.inbetween_times)/self.inbetween_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["inbetween"] = metrics_dict["FID"].item()
                    self.inbetween_times = []
                    self.inbetween_samples = 0  

                if metrics=="TextInbetweenMetrics" and len(self.text_inbetween_times) > 0:
                    if self.task_best_FID["text_inbetween"] > self.task_FID["text_inbetween"]:
                        self.task_best_FID["text_inbetween"] = self.task_FID["text_inbetween"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"TextInbetweenMetrics/Infernce_Time": sum(self.text_inbetween_times)/self.text_inbetween_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["text_inbetween"] = metrics_dict["FID"].item()
                    self.text_inbetween_times = []
                    self.text_inbetween_samples = 0  

                if metrics=="TM2TMetrics" and len(self.text_times) > 0:
                    if self.task_best_FID["text"] > self.task_FID["text"]:
                        self.task_best_FID["text"] = self.task_FID["text"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"TM2TMetrics/Infernce_Time": sum(self.text_times)/self.text_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["text"] = metrics_dict["FID"].item()
                    self.text_times = []
                    self.text_samples = 0

                if metrics=="HintMetrics" and len(self.hint_times) > 0:
                    if self.task_best_FID["hint"] > self.task_FID["hint"]:
                        self.task_best_FID["hint"] = self.task_FID["hint"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"HintMetrics/Infernce_Time": sum(self.hint_times)/self.hint_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["hint"] = metrics_dict["FID"].item()
                    self.hint_times = []
                    self.hint_samples = 0

                if metrics=="StyleMetrics" and len(self.style_times) > 0:
                    if self.task_best_FID["style"] > self.task_FID["style"]:
                        self.task_best_FID["style"] = self.task_FID["style"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"StyleMetrics/Infernce_Time": sum(self.style_times)/self.style_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["style"] = metrics_dict["FID_Style"].item()
                    self.style_times = []
                    self.style_samples = 0

                if metrics=="TextHintMetrics" and len(self.text_hint_times) > 0:
                    if self.task_best_FID["text_hint"] > self.task_FID["text_hint"]:
                        self.task_best_FID["text_hint"] = self.task_FID["text_hint"]
                    metrics_dict = getattr(self,metrics).compute(sanity_flag=self.trainer.sanity_checking)
                    getattr(self, metrics).reset()
                    dico.update({f"{metrics}/{metric}": value.item() for metric, value in metrics_dict.items()})
                    dico.update({"TextHintMetrics/Infernce_Time": sum(self.text_hint_times)/self.text_hint_samples,})
                    if not self.trainer.sanity_checking:
                        self.task_FID["text_hint"] = metrics_dict["FID"].item()
                    self.text_hint_times = []
                    self.text_hint_samples = 0

        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

        torch.cuda.empty_cache()

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self.save_npy(outputs)
        return self.allsplit_epoch_end("test", outputs)

    def read_yaml_to_dict(self, yaml_path: str, ):
        with open(yaml_path) as file:
            dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
            return dict_value

    def on_save_checkpoint(self, checkpoint):
        # don't save clip to checkpoint
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_encoder' in k or "style_encoder" in k or "content_encoder" in k  or "moveencoder" in k or "motionencoder" in k or "textencoder" in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # restore clip state_dict to checkpoint
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in checkpoint['state_dict'].items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

    def load_state_dict(self, state_dict=True):
        # load clip state_dict to checkpoint
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in state_dict.items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v

        super().load_state_dict(new_state_dict, False)

    def get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=259,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        t2m_checkpoint = torch.load(os.path.join(cfg.model.t2m_path, "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def get_style_encoder(self, cfg):
        # style_encoder_path = cfg.model.style_encoder_path
        # style_dict = torch.load(style_encoder_path)
        # style_class = StyleClassification(nclasses=100)
        # style_class.load_state_dict(style_dict, strict=True)
        # self.style_encoder = style_class.eval()
        parameters = self.read_yaml_to_dict("./configs/modules/motionclip_params_263.yaml")
        parameters["device"] = self.device
        self.style_encoder = get_model_and_data(parameters, split='vald')
        checkpointpath = "./checkpoints/mcm-ldm/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.style_encoder, state_dict)

        self.style_encoder.eval()
        self.style_encoder.training = False
        for p in self.style_encoder.parameters():
            p.requires_grad = False
        self.style_encoder.to(self.device)

    def get_content_encoder(self, cfg):
        self.content_encoder = instantiate_from_config(cfg.model.content_encoder)
        checkpointpath = "./checkpoints/mcm-ldm/motion_encoder.ckpt"
        self.content_encoder.load_state_dict(torch.load(checkpointpath))

        self.content_encoder.eval()
        for p in self.content_encoder.parameters():
                p.requires_grad = False
        self.content_encoder.to(self.device)

    def get_style_test_dataset(self, cfg):
        content_path = "./datasets/mcm-ldm/content_test_feats/"
        style_path = "./datasets/mcm-ldm/style_test_feats/"

        content_files = os.listdir(content_path)
        style_files = os.listdir(style_path)

        self.test_content_feats = torch.zeros(40,199,263)
        self.test_content_lengths = []
        self.test_style_feats = torch.zeros(30,59,263)
        self.test_style_lengths = []

        for i in range(len(content_files)):
            content = torch.tensor(np.load(content_path + content_files[i])).unsqueeze(0)
            self.test_content_lengths.append(content.shape[1])

            if content.shape[1] < 199:
                content = torch.cat([content, torch.zeros(1, 199 - content.shape[1], 263)], dim=1)
            self.test_content_feats[i] = content

        for i in range(len(style_files)):
            style = torch.tensor(np.load(style_path + style_files[i])).unsqueeze(0)
            self.test_style_lengths.append(style.shape[1])

            if style.shape[1] < 59:
                style = torch.cat([style, torch.zeros(1, 59 - style.shape[1], 263)], dim=1)
            self.test_style_feats[i] = style

        self.test_content_feats = self.test_content_feats.to(self.device)
        self.test_style_feats = self.test_style_feats.to(self.device)

        self.test_content_feats = self.test_content_feats - self.datamodule.mean
        self.test_content_feats = self.test_content_feats / self.datamodule.std

        self.test_style_feats = self.test_style_feats - self.datamodule.mean
        self.test_style_feats = self.test_style_feats / self.datamodule.std

    def configure_conditions_prob_and_scale(self,cfg, condition_type):
        if condition_type == "text":
            self.text_guidance_scale = cfg.model.text_guidance_scale
            self.text_guidance_prob = cfg.model.text_guidance_prob
            self.none_guidance_prob = cfg.model.none_guidance_prob

        elif condition_type == "hint":
            self.text_hint_guidance_scale = cfg.model.text_hint_guidance_scale
            self.text_hint_guidance_prob = cfg.model.text_hint_guidance_prob
            self.text_guidance_prob = cfg.model.text_guidance_prob
            self.hint_guidance_prob = cfg.model.hint_guidance_prob
            self.none_guidance_prob = cfg.model.none_guidance_prob

        elif condition_type == "inbetween":
            self.inbetween_guidance_scale = cfg.model.inbetween_guidance_scale

        elif condition_type == "source_text":
            self.source_text_guidance_scale_1 = cfg.model.source_text_guidance_scale_1
            self.source_text_guidance_scale_2 = cfg.model.source_text_guidance_scale_2

            self.source_text_guidance_prob = cfg.model.source_text_guidance_prob
            self.source_guidance_prob = cfg.model.source_guidance_prob
            self.none_guidance_prob = cfg.model.none_guidance_prob

        elif condition_type == "source_hint":
            self.source_hint_guidance_scale_1 = cfg.model.source_hint_guidance_scale_1
            self.source_hint_guidance_scale_2 = cfg.model.source_hint_guidance_scale_2

            self.source_hint_guidance_prob = cfg.model.source_hint_guidance_prob
            self.hint_guidance_prob = cfg.model.hint_guidance_prob
            self.source_guidance_prob = cfg.model.source_guidance_prob
            self.none_guidance_prob = cfg.model.none_guidance_prob

        elif condition_type == "style":
            self.style_guidance_scale = cfg.model.style_guidance_scale

            self.all_guidance_prob = cfg.model.all_guidance_prob
            self.drop_hint_guidance_prob = cfg.model.drop_hint_guidance_prob
            self.drop_style_guidance_prob = cfg.model.drop_style_guidance_prob
            self.drop_content_guidance_prob = cfg.model.drop_content_guidance_prob
            self.none_guidance_prob = cfg.model.none_guidance_prob

        elif condition_type == "all":
            self.text_guidance_scale = cfg.model.text_guidance_scale
            self.hint_guidance_scale = cfg.model.hint_guidance_scale
            self.text_hint_guidance_scale = cfg.model.text_hint_guidance_scale
            self.text_inbetween_guidance_scale = cfg.model.text_inbetween_guidance_scale
            self.inbetween_guidance_scale = cfg.model.inbetween_guidance_scale
            self.source_text_guidance_scale_1 = cfg.model.source_text_guidance_scale_1
            self.source_text_guidance_scale_2 = cfg.model.source_text_guidance_scale_2
            self.source_hint_guidance_scale_1 = cfg.model.source_hint_guidance_scale_1
            self.source_hint_guidance_scale_2 = cfg.model.source_hint_guidance_scale_2
            self.source_text_hint_guidance_scale_1 = cfg.model.source_text_hint_guidance_scale_1
            self.source_text_hint_guidance_scale_2 = cfg.model.source_text_hint_guidance_scale_2
            self.style_guidance_scale = cfg.model.style_guidance_scale

            self.drop_style_guidance_prob = cfg.model.drop_style_guidance_prob
            self.drop_content_guidance_prob = cfg.model.drop_content_guidance_prob

    def configure_instructions(self):
        with torch.no_grad():
            self.text_encoder.to(self.device)
            self.instructions = {}
            self.instructions["uncond"] = self.text_encoder("reconstruct given masked source motion.")[0][0]
            self.instructions["recon"] = self.text_encoder("reconstruct given masked source motion.")[0][0]

            # self.instructions["uncond"] = self.text_encoder("unconditionally generate motion.")[0][0]
            # self.instructions["recon"] = self.text_encoder("reconstruct given source motion.")[0][0]

            self.instructions["masked"] = self.text_encoder("reconstruct given masked source motion.")[0][0]
            self.instructions["hint"] = self.text_encoder("generate motion by given trajectory.")[0][0]

            self.instructions["inbetween"] = self.text_encoder("generate motion by given key frames.")[0][0]
            # self.instructions["inbetween"] = self.text_encoder("generate motion by given trajectory.")[0][0]

            self.instructions["source_hint"] = self.text_encoder("edit source motion by given trajectory.")[0][0]
            self.instructions["source_text_hint"] = self.text_encoder("edit source motion by given text and trajectory.")[0][0]
            self.instructions["source_text"] = self.text_encoder("edit source motion by given text.")[0][0]

            self.instructions["style"] = self.text_encoder("generate motion by the given style and content.")[0][0]
            self.instructions["content"] = self.text_encoder("generate motion by the given style and content.")[0][0]
            self.instructions["style_content"] = self.text_encoder("generate motion by the given style and content.")[0][0]

            self.instructions["text_hint"] = self.text_encoder("generate motion by given text and trajectory.")[0][0]
            self.instructions["text_inbetween"] = self.text_encoder("generate motion by given text and key frames.")[0][0]
            # self.instructions["text_inbetween"] = self.text_encoder("generate motion by given text and trajectory.")[0][0]

            self.instructions["text"] = self.text_encoder("generate motion by given text.")[0][0]

    def configure_optimizers(self):
        if not self.warm_up:
            return {"optimizer": self.optimizer}
        else:
            return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler':  get_scheduler("cosine",optimizer=self.optimizer,num_warmup_steps=1000,num_training_steps=self.cfg.TRAIN.END_EPOCH),
                'interval': 'epoch',
                "name": "learning_rate_scheduler",
                            }
                    }

    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == "UncondMetrics":
                self.UncondMetrics = UncondMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "ReconMetrics":
                self.ReconMetrics = ReconMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TemosMetric":
                self.TemosMetric = ComputeMetrics(
                    njoints=self.njoints,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MRMetrics":
                self.MRMetrics = MRMetrics(
                    njoints=self.njoints,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "HUMANACTMetrics":
                self.HUMANACTMetrics = HUMANACTMetrics(
                    datapath=os.path.join(self.cfg.model.humanact12_rec_path,
                                          "humanact12_gru.tar"),
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UESTCMetrics":
                self.UESTCMetrics = UESTCMetrics(
                    cfg=self.cfg,
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UncondSMPLPairedMetrics":
                self.UncondSMPLPairedMetrics = UncondSMPLPairedMetrics(
                    TMR_path=self.cfg.model.TMR_path,
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MotionFixMetrics":
                self.MotionFixMetrics = MotionFixMetrics(
                     TMR_path=self.cfg.model.TMR_path,
                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,   
                )
            elif metric == "MotionFixHintMetrics":
                self.MotionFixHintMetrics = MotionFixHintMetrics(
                     TMR_path=self.cfg.model.TMR_path,
                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,   
                )
            elif metric == "InbetweenMetrics":
                self.InbetweenMetrics = InbetweenMetrics(
                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,   
                )
            elif metric == "HintMetrics":
                self.HintMetrics = HintMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MMMetrics":
                self.MMMetrics = MMMetrics(
                    mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "SourceTextMetrics":
                self.SourceTextMetrics = SourceTextMetrics(
                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,   
                )
            elif metric == "SourceHintMetrics":
                self.SourceHintMetrics = SourceHintMetrics(
                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,   
                )
            elif metric == "StyleMetrics":
                self.StyleMetrics = StyleMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MaskedMetrics":
                self.MaskedMetrics = MaskedMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TextHintMetrics":
                self.TextHintMetrics = TextHintMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TextInbetweenMetrics":
                self.TextInbetweenMetrics = TextInbetweenMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "SourceTextHintMetrics":
                self.SourceTextHintMetrics = SourceTextHintMetrics(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
        else:
            TypeError(f"metric {metric} not supported")

    def save_npy(self, outputs):
        cfg = self.cfg
        output_dir = Path(os.path.join(cfg.FOLDER,str(cfg.model.model_type),str(cfg.NAME),"samples_" + cfg.TIME,))
        if cfg.TEST.SAVE_PREDICTIONS:
            lengths = [i[1] for i in outputs]
            outputs = [i[0] for i in outputs]
            if cfg.TEST.DATASETS[0].lower() in ["humanml3d"]:
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu().numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
