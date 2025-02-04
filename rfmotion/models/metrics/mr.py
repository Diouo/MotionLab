from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric

from .utils import *


# motion reconstruction metric
class MRMetrics(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "humanml3d",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d",'motionfix']:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",default=torch.tensor(0),dist_reduce_fx="sum")

        self.add_state("MPJPE",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("PAMPJPE",default=torch.tensor([0.0]),dist_reduce_fx="sum")
        self.add_state("ACCL",default=torch.tensor([0.0]),dist_reduce_fx="sum")

        self.MR_metrics = ["MPJPE", "PAMPJPE", "ACCL"]

        # All metric
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):
        if self.force_in_meter:
            # different jointstypes have different scale factors
            # if self.jointstype == 'mmm':
            #     factor = 1000.0
            # elif self.jointstype == 'humanml3d':
            #     factor = 1000.0 * 0.75 / 480
            factor = 1000.0
        else:
            factor = 1.0

        count = self.count
        count_seq = self.count_seq
        mr_metrics = {}
        mr_metrics["MPJPE"] = self.MPJPE / count * factor
        mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
        mr_metrics["ACCL"] = self.ACCL / (count - 2 * count_seq) * factor  # accl error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        return mr_metrics
    
    def update(self, 
               length_source, source_m_rst, source_m_ref,  
               length_target, target_m_rst, target_m_ref, ):
       # align root joints index
        if self.align_root and self.jointstype in ['mmm', 'humanml3d','motionfix']:
            align_inds = [0]
        else:
            align_inds = None

        ## source motion
        self.count += sum(length_source)
        self.count_seq += len(length_source)
        # avoid cuda error of DDP in pampjpe
        rst = source_m_rst
        ref = source_m_ref 
        for i in range(len(length_source)):
            self.MPJPE += torch.sum(calc_mpjpe(rst[i,:length_source[i],:], ref[i,:length_source[i],:], align_inds=align_inds))
            self.PAMPJPE += torch.sum(calc_pampjpe(rst[i,:length_source[i],:], ref[i,:length_source[i],:]))
            self.ACCL += torch.sum(calc_accel(rst[i,:length_source[i],:], ref[i,:length_source[i],:]))

        ## target motion
        self.count += sum(length_target)
        self.count_seq += len(length_target)
        # avoid cuda error of DDP in pampjpe
        rst = target_m_rst
        ref = target_m_ref
        for i in range(len(length_target)):
            self.MPJPE += torch.sum(calc_mpjpe(rst[i,:length_target[i],:], ref[i,:length_target[i],:], align_inds=align_inds))
            self.PAMPJPE += torch.sum(calc_pampjpe(rst[i,:length_target[i],:], ref[i,:length_target[i],:]))
            self.ACCL += torch.sum(calc_accel(rst[i,:length_target[i],:], ref[i,:length_target[i],:]))
