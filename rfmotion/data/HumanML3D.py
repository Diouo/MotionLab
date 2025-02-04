import numpy as np
import torch

from rfmotion.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2, MotionFixRetarget, Text2MotionDatasetCMLDTest, AllDataset


class HumanML3DDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 dataset_name,
                 train_batch_size,
                 eval_batch_size,
                 test_batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(train_batch_size=train_batch_size,
                         eval_batch_size=eval_batch_size,
                         test_batch_size=test_batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = dataset_name
        self.njoints = 22
        if dataset_name.lower() in ["humanml3d"]:
            self.Dataset = Text2MotionDatasetV2
        elif dataset_name.lower() in ["humanml3d_100style"]:
            self.Dataset = Text2MotionDatasetCMLDTest
        elif dataset_name.lower() in ["motionfix_retarget"]:
            self.Dataset = MotionFixRetarget
        elif dataset_name.lower() in ["all"]:
            self.Dataset = AllDataset

        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms
        self.mean_motion = torch.tensor(np.load('./datasets/{}/mean_motion.npy'.format(dataset_name)))
        self.std_motion = torch.tensor(np.load('./datasets/{}/std_motion.npy'.format(dataset_name)))
        self.mean = torch.tensor(self.hparams.mean)
        self.std = torch.tensor(self.hparams.std)

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
