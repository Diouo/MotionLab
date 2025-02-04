import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric


class RFMotionLosses(Metric):
    def __init__(self, vae, mode, cfg, prediction_type):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        self.vae = vae
        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.predict_type = prediction_type
        self.stage = cfg.TRAIN.STAGE

        losses = []

        # vae loss
        if self.stage in ['vae']:
            # reconstruction loss
            losses.append("recons_source")
            losses.append("recons_target")

            # KL loss
            losses.append("kl_source")
            losses.append("kl_target")

        # diffusion loss
        elif self.stage in ['diffusion']:
            # instance noise loss
            if self.predict_type == "epsilon":
                losses.append("epsilon_loss")
            elif self.predict_type == "sample":
                losses.append("sample_loss")
            elif self.predict_type == "v_prediction":
                losses.append("v_loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                losses.append("prior_loss")  # prior noise loss

        losses.append("total")
        for loss in losses:
            self.add_state(loss,default=torch.tensor(0.0),dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'epsilon':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'sample':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'v':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae"]:
            total += self._update_loss("recons_source", rs_set['source_m_rst'],rs_set['source_m_ref'])
            total += self._update_loss("kl_source", rs_set['source_dist_m'], rs_set['source_dist_ref'])

            total += self._update_loss("recons_target", rs_set['target_m_rst'],rs_set['target_m_ref'])
            total += self._update_loss("kl_target", rs_set['target_dist_m'], rs_set['target_dist_ref'])

        elif self.stage in ["diffusion"]:
            # predict noise
            if self.predict_type == "epsilon":
                total += self._update_loss("epsilon_loss", rs_set['noise_pred'],rs_set['noise'])
            # predict x
            elif self.predict_type == "sample":
                total += self._update_loss("sample_loss", rs_set['latent_pred'],rs_set['latent'])
            # predict v
            elif self.predict_type == "v_prediction":
                total += self._update_loss("v_loss", rs_set['v_pred'],rs_set['v_gt'])

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0: # loss - prior loss
                total += self._update_loss("prior_loss", rs_set['noise_prior'],rs_set['dist_m1'])

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
