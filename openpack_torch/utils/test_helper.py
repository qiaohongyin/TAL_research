from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from openpack_torch.lightning import BaseLightningModule
from openpack_torch.codalab.utils import (
    eval_operation_segmentation_wrapper,
)

logger = getLogger(__name__)

SCENARIO_DICT = {
    "U0101-S0100": "S1",
    "U0101-S0200": "S1",
    "U0101-S0300": "S1",
    "U0101-S0400": "S1",
    "U0101-S0500": "S1",
    "U0102-S0100": "S1",
    "U0102-S0200": "S1",
    "U0102-S0300": "S1",
    "U0102-S0400": "S1",
    "U0102-S0500": "S1",
    "U0103-S0100": "S1",
    "U0103-S0200": "S1",
    "U0103-S0300": "S1",
    "U0103-S0400": "S1",
    "U0103-S0500": "S1",
    "U0104-S0100": "S1",
    "U0104-S0200": "S1",
    "U0104-S0300": "S1",
    "U0104-S0400": "S1",
    "U0105-S0100": "S1",
    "U0105-S0200": "S1",
    "U0105-S0300": "S1",
    "U0105-S0400": "S1",
    "U0105-S0500": "S1",
    "U0106-S0100": "S1",
    "U0106-S0200": "S1",
    "U0106-S0300": "S1",
    "U0106-S0400": "S1",
    "U0106-S0500": "S1",
    "U0107-S0100": "S1",
    "U0107-S0200": "S1",
    "U0107-S0300": "S1",
    "U0107-S0400": "S1",
    "U0107-S0500": "S1",
    "U0108-S0100": "S1",
    "U0108-S0200": "S1",
    "U0108-S0300": "S1",
    "U0108-S0400": "S1",
    "U0108-S0500": "S1",
    "U0109-S0100": "S1",
    "U0109-S0200": "S1",
    "U0109-S0300": "S1",
    "U0109-S0400": "S1",
    "U0109-S0500": "S1",
    "U0110-S0100": "S1",
    "U0110-S0200": "S1",
    "U0110-S0300": "S1",
    "U0110-S0400": "S1",
    "U0110-S0500": "S1",
    "U0111-S0100": "S1",
    "U0111-S0200": "S1",
    "U0111-S0300": "S1",
    "U0111-S0400": "S1",
    "U0111-S0500": "S1",
    "U0201-S0100": "S2",
    "U0201-S0200": "S2",
    "U0201-S0300": "S3",
    "U0201-S0400": "S3",
    "U0201-S0500": "S4",
    "U0202-S0100": "S2",
    "U0202-S0200": "S2",
    "U0202-S0300": "S3",
    "U0202-S0400": "S3",
    "U0202-S0500": "S4",
    "U0203-S0100": "S2",
    "U0203-S0200": "S2",
    "U0203-S0300": "S3",
    "U0203-S0400": "S3",
    "U0203-S0500": "S4",
    "U0204-S0100": "S2",
    "U0204-S0200": "S2",
    "U0204-S0300": "S3",
    "U0204-S0400": "S3",
    "U0204-S0500": "S4",
    "U0205-S0100": "S2",
    "U0205-S0200": "S2",
    "U0205-S0300": "S3",
    "U0205-S0400": "S3",
    "U0205-S0500": "S4",
    "U0206-S0100": "S2",
    "U0206-S0200": "S2",
    "U0206-S0300": "S3",
    "U0206-S0400": "S3",
    "U0206-S0500": "S4",
    "U0207-S0100": "S2",
    "U0207-S0200": "S2",
    "U0207-S0300": "S3",
    "U0207-S0400": "S3",
    "U0207-S0500": "S4",
    "U0208-S0100": "S2",
    "U0208-S0200": "S2",
    "U0208-S0300": "S3",
    "U0208-S0400": "S3",
    "U0208-S0500": "S4",
    "U0209-S0100": "S2",
    "U0209-S0200": "S2",
    "U0209-S0300": "S3",
    "U0209-S0400": "S3",
    "U0209-S0500": "S4",
    "U0210-S0100": "S2",
    "U0210-S0200": "S2",
    "U0210-S0300": "S3",
    "U0210-S0400": "S3",
    "U0210-S0500": "S4",
}

CLASSES_TUPLE = (
    (0, "Picking"),
    (1, "Relocate Item Label"),
    (2, "Assemble Box"),
    (3, "Insert Items"),
    (4, "Close Box"),
    (5, "Attach Box Label"),
    (6, "Scan Label"),
    (7, "Attach Shipping Label"),
    (8, "Put on Back Table"),
    (9, "Fill out Order"),
    (10, "Null"),
)

def test_helper(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    plmodel: BaseLightningModule,
    trainer: pl.Trainer,
    logdir: Path,
):
    dataloaders = datamodule.test_dataloader()
    split = cfg.dataset.split.test if not hasattr(cfg.dataset.split, "spec") else cfg.dataset.split.spec.test

    outputs = dict()
    
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        key = f"{user}-{session}"
        logger.info(f"test on {user}-{session}")

        trainer.test(plmodel, dataloader)
        res = plmodel.test_results
    
        ts_arr = res["unixtime"]
        t_arr = res["t"]
        y_arr = res["y"]

        ts_flat = ts_arr.ravel()
        t_flat = t_arr.ravel()
        if y_arr.ndim == 3:
             num_classes = y_arr.shape[1]
             y_flat = y_arr.transpose(0, 2, 1).reshape(-1, num_classes)
        else:
             y_flat = y_arr

        sort_idx = np.argsort(ts_flat)
        outputs[key] = {
            "y": y_flat[sort_idx].transpose(1, 0)[np.newaxis, ...], 
            "t_idx": t_flat[sort_idx],
            "unixtime": ts_flat[sort_idx],
        }

        if hasattr(plmodel, "clear_test_outputs"):
            print(f"DEBUG: Clearing cache via parent method...")
            plmodel.clear_test_outputs()
        else:
            plmodel.test_step_outputs.clear()

    split_keys = [k.split("-") for k in outputs.keys()]
    df_summary = compute_score_for_each_scenario(cfg, split_keys, outputs)
    path = logdir / "summary" / "test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(path, index=False)
    logger.info(f"write df_summary[shape={df_summary.shape}] to {path}")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    logger.info(f"df_summary:\n{df_summary}")

    return outputs, df_summary

def compute_score_for_each_scenario(cfg,  split, outputs):
    df_summary = []

    for target_scenario in ["S1", "S2", "S3", "S4", "all"]:
        _outputs = dict()
        for _user, _sess in split:
            key = f"{_user}-{_sess}"
            scenario = SCENARIO_DICT[key]
            if (scenario == target_scenario) or (target_scenario == "all"):
                _outputs[key] = outputs[key]

        if len(_outputs) > 0:
            df_tmp = eval_operation_segmentation_wrapper(cfg, _outputs, CLASSES_TUPLE)
            df_tmp["scenario"] = target_scenario
            df_summary.append(df_tmp)

    df_summary = pd.concat(df_summary, axis=0)
    return df_summary