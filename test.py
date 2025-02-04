import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf

from rfmotion.callback import ProgressLogger
from rfmotion.config import parse_args
from rfmotion.data.get_data import get_datasets
from rfmotion.models.get_model import get_model
from rfmotion.utils.logger import create_logger


def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main(cfg, logger):
    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
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
    ]
    logger.info("Callbacks initialized")

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
        inference_mode=False,
    )

    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    # calculate metrics
    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    # metrics = trainer.validate(model, datamodule=datasets[0])
    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)


    # save metrics to file
    output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    logger = create_logger(cfg, phase="test")

    # set seed
    pl.seed_everything(cfg.SEED_VALUE) 

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision('high')


    main(cfg, logger)
