from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .MotionFix import MotionFixDataModule
from .utils import *


def get_mean_std(phase, cfg, dataset_name):
    if phase in ["val"]:
        data_root = pjoin(cfg.model.t2m_path, "Comp_v6_KLD01","meta")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit", "motionfix_retarget", "motionfix", "humanml3d_100style", "all"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, phase="train"):
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate
    elif name.lower() in ["motionfix_retarget"]:
        return motionfix_retarget_collate
    elif name.lower() in ["humanml3d_100style"]:
        return humanml3d_100style_collate
    elif name.lower() in ["all"]:
        return all_collate


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "humanml3d_100style": HumanML3DDataModule,
    "motionfix_retarget": HumanML3DDataModule,
    "motionfix": MotionFixDataModule,
    "all":HumanML3DDataModule,
}
motion_subdir = {"humanml3d": "new_joint_vecs", "motionfix_retarget": "new_joint_vecs", "humanml3d_100style": "new_joint_vecs", "all": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            
            if phase =="demo":
                test_batch_size = cfg.DEMO.BATCH_SIZE
            else:
                test_batch_size = cfg.TEST.BATCH_SIZE
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                train_batch_size=cfg.TRAIN.BATCH_SIZE,
                eval_batch_size=cfg.EVAL.BATCH_SIZE,
                test_batch_size=test_batch_size,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
            cfg.DATASET.NFEATS = datasets[0].nfeats
            cfg.DATASET.NJOINTS = datasets[0].njoints

        elif dataset_name.lower() in ["humanml3d_100style"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            
            if phase =="demo":
                test_batch_size = cfg.DEMO.BATCH_SIZE
            else:
                test_batch_size = cfg.TEST.BATCH_SIZE
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                train_batch_size=cfg.TRAIN.BATCH_SIZE,
                eval_batch_size=cfg.EVAL.BATCH_SIZE,
                test_batch_size=test_batch_size,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
            cfg.DATASET.NFEATS = datasets[0].nfeats
            cfg.DATASET.NJOINTS = datasets[0].njoints

        elif dataset_name.lower() in ["motionfix_retarget"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            
            if phase =="demo":
                test_batch_size = cfg.DEMO.BATCH_SIZE
            else:
                test_batch_size = cfg.TEST.BATCH_SIZE
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                train_batch_size=cfg.TRAIN.BATCH_SIZE,
                eval_batch_size=cfg.EVAL.BATCH_SIZE,
                test_batch_size=test_batch_size,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
            cfg.DATASET.NFEATS = datasets[0].nfeats
            cfg.DATASET.NJOINTS = datasets[0].njoints
            
        elif dataset_name.lower() in ["all"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            
            if phase =="demo":
                test_batch_size = cfg.DEMO.BATCH_SIZE
            else:
                test_batch_size = cfg.TEST.BATCH_SIZE
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                train_batch_size=cfg.TRAIN.BATCH_SIZE,
                eval_batch_size=cfg.EVAL.BATCH_SIZE,
                test_batch_size=test_batch_size,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
            cfg.DATASET.NFEATS = datasets[0].nfeats
            cfg.DATASET.NJOINTS = datasets[0].njoints

        elif dataset_name.lower() in ["motionfix"]:
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")+"/motionfix.pth.tar",
                smplh_path=eval(f"cfg.DATASET.SMPLH_PATH"),
                train_batch_size=cfg.TRAIN.BATCH_SIZE,
                val_batch_size=cfg.EVAL.BATCH_SIZE,
                test_batch_size=cfg.TEST.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                load_feats=eval(f"cfg.DATASET.{dataset_name.upper()}.load_feats"),
                preproc=eval(f"cfg.DATASET.{dataset_name.upper()}.preproc"),
            )
            datasets.append(dataset)
            cfg.DATASET.NFEATS = int(sum(dataset.nfeats))
            cfg.model.motion_vae.nfeats = int(sum(dataset.nfeats))
            cfg.model.denoiser.nfeats = int(sum(dataset.nfeats))
            cfg.DATASET.NJOINTS = 22
        else:
            raise NotImplementedError
    return datasets
