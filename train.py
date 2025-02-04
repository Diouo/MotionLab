import os
from pprint import pformat
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from rfmotion.callback import ProgressLogger
from rfmotion.config import parse_args
from rfmotion.data.get_data import get_datasets
from rfmotion.models.get_model import get_model
from rfmotion.utils.logger import create_logger


def main(cfg,logger):
    # tensorboard logger and wandb logger
    loggers = []
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.FOLDER_EXP,sub_dir="tensorboard",version="",name="")
    loggers.append(tb_logger)

    # create dataset
    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    # optimizer
    metric_monitor = {
        ## For VAE
        "MPJPE": "VAE/MPJPE",
        "PAMPJPE": "VAE/PAMPJPE",
        "ACCL": "VAE/ACCL",
        "FID": "VAE/FID",
        "Diversity": "VAE/Diversity",
        # "FID_GT": "VAE/FID_GT",
        # "Diversity_GT": "VAE/Diversity_GT",

        ## For Rectified Flow
        "Source_Text_R1_G2T": "Source_Text/R1_G2T",
        "Source_Text_R2_G2T": "Source_Text/R2_G2T",
        "Source_Text_R3_G2T": "Source_Text/R3_G2T",
        "Source_Text_AvgR_G2T": "Source_Text/AvgR_G2T",
        "Source_Text_R1_G2S": "Source_Text/R1_G2S",
        "Source_Text_R2_G2S": "Source_Text/R2_G2S",
        "Source_Text_R3_G2S": "Source_Text/R3_G2S",
        "Source_Text_AvgR_G2S": "Source_Text/AvgR_G2S",
        "Source_Text_FID": "Source_Text/FID",
        "Source_Text_Diversity": "Source_Text/Diversity",
        "Source_Text_Inference_Time":"Source_Text/Inference_Time",

        "Source_Hint_R1_G2T": "Source_Hint/R1_G2T",
        "Source_Hint_R2_G2T": "Source_Hint/R2_G2T",
        "Source_Hint_R3_G2T": "Source_Hint/R3_G2T",
        "Source_Hint_AvgR_G2T": "Source_Hint/AvgR_G2T",
        "Source_Hiny_R1_G2S": "Source_Hint/R1_G2S",
        "Source_Hiny_R2_G2S": "Source_Hint/R2_G2S",
        "Source_Hiny_R3_G2S": "Source_Hint/R3_G2S",
        "Source_Hint_AvgR_G2S": "Source_Hint/AvgR_G2S",
        "Source_Hint_FID": "Source_Hint/FID",
        "Source_Hint_Diversity": "Source_Hint/Diversity",
        "Source_Hint_Distance": "Source_Hint/Distance",
        "Source_Hint_Inference_Time":"Source_Hint/Inference_Time",

        "Inbetween_R3": "Inbetween/R3",
        "Inbetween_FID": "Inbetween/FID",
        "Inbetween_Diversity": "Inbetween/Diversity",
        "Inbetween_Distance": "Inbetween/Distance",
        "Inbetween_Skating_Ratio": "Inbetween/Skating_Ratio",
        "Inbetween_Inference_Time":"Inbetween/Inference_Time",

        "Text_R3": "Text/R3",
        "Text_FID": "Text/FID",
        "Text_Matching_Score": "Text/Matching_Score",
        "Text_MultiModality": "Text/MultiModality",
        "Text_Diversity": "Text/Diversity",
        "Text_Inference_Time":"Text/Inference_Time",

        "Hint_R3": "Hint/R3",
        "Hint_FID": "Hint/FID",
        "Hint_Skating_Ratio": "Hint/Skating_Ratio",
        "Hint_Distance": "Hint/Distance",
        "Hint_Diversity": "Hint/Diversity",
        "Hint_Inference_Time":"Hint/Inference_Time",
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="epoch",
            every_n_epochs=cfg.LOGGER.SAVE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),      
    ]
    logger.info("Callbacks initialized")

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy= "ddp" if len(cfg.DEVICE) > 1 else None,
        default_root_dir=cfg.FOLDER_EXP,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.CHECK_VAL_EVERY_N_EPOCH,
    )
    logger.info("Trainer initialized")

    # strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,map_location="cpu")["state_dict"]
        # extract encoder/decoder
        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "vae":
                name = k.replace("vae.", "")
                vae_dict[name] = v
        model.vae.load_state_dict(vae_dict, strict=True)

    # fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,datamodule=datasets[0],ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")


if __name__ == "__main__":
    # parse options
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # resume
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        backcfg = cfg.TRAIN.copy()
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    cfg.TRAIN = backcfg
                    break
            checkpoints = sorted(os.listdir(os.path.join(resume, "checkpoints")),key=lambda x: int(x[6:-5]),reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.TRAIN.PRETRAINED = os.path.join(resume, "checkpoints", checkpoint)
                    break
        else:
            raise ValueError("Resume path is not right.")

    main(cfg,logger)
