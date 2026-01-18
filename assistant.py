from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import os
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning.callbacks 
import pytorch_lightning.loggers
import openpack_torch as optorch
from openpack_torch.utils.test_helper import test_helper
from openpack_torch.utils.io import cleanup_dir
logger = getLogger(__name__)

# ----------------------------------------------------------------------
class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.OpenPackkd

class DeepConvLSTMLM(optorch.lightning.BaseLightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        Ks = 3
        A = optorch.models.keypoint.get_adjacency_matrix(
            layout="MSCOCO", hop_size=Ks - 1
        )
        
        # --- Teacher Model ---
        self.stgcn_teacher = optorch.models.keypoint.STGCN4Seg(
            in_channels=3,
            num_classes=11,
            Ks=3,
            Kt=9,
            A=A,
        )
        weight_path = "/workspaces/TAL_research/teacher.pt"
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict_raw = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith("net."):
                name = k.replace("net.", "") 
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
                
        self.stgcn_teacher.load_state_dict(new_state_dict, strict=True)
        self.stgcn_teacher.eval()
        for p in self.stgcn_teacher.parameters():
            p.requires_grad = False
            

        self.proj_imu = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
        self.proj_sk = torch.nn.Sequential(
            torch.nn.Linear(128, 128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
        self.temperature = 0.07
        self.loss_weights = nn.Parameter(torch.zeros(4))

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        model = optorch.models.imu.DeepConvLSTM_PT(6, 11)
        return model

    def configure_optimizers(self):
        params = list(self.net.parameters()) + \
                list(self.proj_imu.parameters()) + \
                list(self.proj_sk.parameters())+ \
                 [self.loss_weights] 
        optimizer = torch.optim.Adam(params, lr=1e-3,  weight_decay=0.0001)
        return {"optimizer": optimizer}

    def forward(self, x):
        return self.net(x)
    
    def gaussian_smooth(self, importance_vec):
        """输入: (B, T) -> 输出: (B, T)"""
        B, T = importance_vec.shape
        kernel = torch.tensor([0.25, 0.5, 0.25], device=importance_vec.device).view(1, 1, 3)
        x = importance_vec.unsqueeze(1) 
        x_smooth = F.conv1d(x, kernel, padding=1)
        return x_smooth.squeeze(1)  
    
    def dense_contrastive_loss(self, z_student, z_teacher):
        """ InfoNCE Loss with Random Sampling """
        B, T, C = z_student.shape
        num_samples = 64
        
        indices = torch.randint(0, T, (B, num_samples), device=self.device) # (B, 64)
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, C) # (B, 64, C)
        
        z_stu_sampled = torch.gather(z_student, 1, idx_expanded)
        z_tea_sampled = torch.gather(z_teacher, 1, idx_expanded)
        
        z_stu_flat = z_stu_sampled.reshape(-1, C) 
        z_tea_flat = z_tea_sampled.reshape(-1, C)
        
        # InfoNCE
        logits = torch.matmul(z_stu_flat, z_tea_flat.T) / self.temperature
        labels = torch.arange(z_stu_flat.shape[0], device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
        
    def train_val_common_step(self, batch: Dict, batch_idx):
        x = batch["imu_teacher"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long) 
        sk_x = batch["skeleton"].to(device=self.device, dtype=torch.float) 
        dist_t = batch["wrist_dist"].to(device=self.device, dtype=torch.float)
        
        # Target Normalization (Regression)
        target_mean = 109.799149
        target_std  = 55.504784
        target_norm_reg = (dist_t - target_mean) / target_std
        

        with torch.no_grad():
            teacher_logits, teacher_med, teacher_att = self.stgcn_teacher(sk_x) 
            
            # --- Task A: Attention Distillation Prep ---
            target_joints = [9, 10]
            teacher_hand_att = teacher_att[:, target_joints, :, :]
            hand_att_merged = torch.logsumexp(teacher_hand_att, dim=1) 
            teacher_importance_raw = torch.logsumexp(hand_att_merged, dim=-2)
            teacher_soft_target = self.gaussian_smooth(teacher_importance_raw)
            teacher_norm = teacher_soft_target / (teacher_soft_target.sum(dim=-1, keepdim=True) + 1e-9)
            teacher_feat = teacher_med.mean(dim=3).permute(0, 2, 1)

        student_logits, dist_pred, student_att, student_med = self.net(x)
        
        # --- Task A: Classification (Main) ---
        student_logits = student_logits.squeeze(3)
        B, C, T = student_logits.shape
        logits_flat = student_logits.permute(0, 2, 1).reshape(-1, C)
        t_flat = t.reshape(-1)
        ce_loss = self.criterion(logits_flat, t_flat) 
        
        # --- Task B: Regression ---
        loss_reg = F.mse_loss(dist_pred, target_norm_reg)

        # --- Task C: Contrastive Learning ---
        z_teacher = F.normalize(self.proj_sk(teacher_feat), dim=2)
        z_student = F.normalize(self.proj_imu(student_med), dim=2)
        loss_cl = self.dense_contrastive_loss(z_student, z_teacher)

        # --- Task D: Attention Prep ---
        student_importance = torch.logsumexp(student_att, dim=-2) 
        student_norm = student_importance / (student_importance.sum(dim=-1, keepdim=True) + 1e-9)
        loss_tad = F.mse_loss(student_norm, teacher_norm) * 1000000

        # ==========================================
        # 5. Weighted Sum
        # ==========================================
        precision1 = torch.exp(-self.loss_weights[0])
        l1 = precision1 * ce_loss + self.loss_weights[0]
        
        precision2 = torch.exp(-self.loss_weights[1])
        l2 = precision2 * loss_tad + self.loss_weights[1]
        
        precision3 = torch.exp(-self.loss_weights[2])
        l3 = precision3 * loss_reg + self.loss_weights[2]
        
        precision4 = torch.exp(-self.loss_weights[3])
        l4 = precision4 * loss_cl + self.loss_weights[3]
        
        loss = l1 + l2 + l3 + l4
        acc = self.calc_accuracy(student_logits, t)
        
        return {
            "loss": loss, 
            "acc": acc, 
            "loss_tad": loss_tad, 
            "loss_reg": loss_reg,
            "loss_cl": loss_cl
        }
    
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["imu_teacher"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat,_,_,_= self(x)
        y_hat = y_hat.squeeze(3)
        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        self.test_step_outputs.append(outputs)
        return outputs

# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    logdir = Path("/datastore/code/log/cl")
    logger.debug(f"logdir = {logdir}")
    cleanup_dir(logdir, exclude="hydra")

    datamodule = OpenPackImuDataModule(cfg)
    plmodel = DeepConvLSTMLM(cfg)
    logger.info(plmodel)


    max_epoch = (
        cfg.train.debug.epochs.maximum if cfg.debug else cfg.train.epochs.maximum
    )

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        mode=cfg.train.early_stop.mode,
        monitor=cfg.train.early_stop.monitor,
        filename="{epoch:02d}-{train/loss:.2f}-{val/loss:.2f}",
        verbose=False,
    )

    early_stop_callback = pytorch_lightning.callbacks.EarlyStopping(
        **cfg.train.early_stop,
    )

    pl_logger = pytorch_lightning.loggers.CSVLogger(logdir)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1],
        min_epochs=1,
        max_epochs=max_epoch,
        logger=pl_logger,
        default_root_dir=logdir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=4,
        deterministic=True,
    )

    trainer.fit(plmodel, datamodule)
    logger.info(f"Finish training! (logdir = {logdir})")



def test(cfg: DictConfig, mode: str = "test"):
    assert mode in ("test", "submission", "test-on-submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda:1")
    logdir = Path("/datastore/code/log/cl")

    datamodule = OpenPackImuDataModule(cfg)
    datamodule.setup(mode)
    ckpt_path = Path(logdir, "lightning_logs", "version_0", "checkpoints", "last.ckpt")
    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = DeepConvLSTMLM.load_from_checkpoint(
    ckpt_path,
    cfg=cfg,
    strict=False,    
    map_location=device,
)
    plmodel.to(dtype=torch.float, device=device)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1],
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    test_helper(cfg, datamodule, plmodel, trainer, logdir)


@hydra.main(
    version_base=None, config_path="./configs", config_name="deep-conv-lstm.yaml"
)
def main(cfg: DictConfig):
    pl.seed_everything(seed=10, workers=True)
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")


if __name__ == "__main__":
    main()

