from logging import getLogger
from pathlib import Path
from typing import Dict
import hydra
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import openpack_torch as optorch
from openpack_torch.utils.test_helper import test_helper
from openpack_torch.utils.io import cleanup_dir

logger = getLogger(__name__)

# ----------------------------------------------------------------------
def kd_loss_kd(student_feats: torch.Tensor,
                  teacher_feats: torch.Tensor,
                  temperature: float = 5.0) -> torch.Tensor:
    t = temperature
    with torch.no_grad():
        p_teacher = F.softmax(teacher_feats / t, dim=-1)
    log_p_student = F.log_softmax(student_feats / t, dim=-1)
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return kl    

def min_max_norm(x):
        # x: (B, T)
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-9) 
    
# ----------------------------------------------------------------------

class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.OpenPackkd
 
class DeepConvLSTMLM(optorch.lightning.BaseLightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # --- Assistant Model ---
        ckpt_path = cfg.get("assistant_ckpt_path")
        print(f"Loading Teacher Checkpoint from: {ckpt_path}") 
        ckpt = torch.load(ckpt_path, map_location=self.device)  
        self.assistant_model = optorch.models.imu.DeepConvLSTM_PT1(6, 11)
        state_dict = {k.replace("net.", ""): v 
                    for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        self.assistant_model.load_state_dict(state_dict, strict=False)
        for p in self.assistant_model.parameters():
            p.requires_grad = False
        self.loss_weights = nn.Parameter(torch.zeros(4))       
        
    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        model = optorch.models.imu.DeepConvLSTM_PT(3, 11)
        return model

    def forward(self, x):
        return self.net(x)  
         
    def train_val_common_step(self, batch: Dict, batch_idx):
        x = batch["imu_student"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long) 
        sk_x = batch["skeleton"].to(device=self.device, dtype=torch.float)
        x_x = batch["imu_teacher"].to(device=self.device, dtype=torch.float) 

        # ---- Teacher ----
        with torch.no_grad():
            teacher_logits,_, teacher_att,teacher_med = self.assistant_model(x_x) 

        # ---- Student ----
        s_logits, dist_logits, student_att, student_med = self(x)
        s_logits = s_logits.squeeze(3)  # [B, num_cls, T]
        B, C, T = s_logits.shape
        logits_flat = s_logits.permute(0, 2, 1).reshape(-1, C)
        t_flat = t.reshape(-1)
        ce = self.criterion(logits_flat, t_flat) 

        # ---- KD Loss ----
        loss_kd = kd_loss_kd(student_med, teacher_med)
        
        # ---- 计算att----
        teacher_imp_h = torch.logsumexp(teacher_att, dim=-1)   # (B,H,T)
        student_imp_h = torch.logsumexp(student_att, dim=-1)   # (B,H,T)
        teacher_imp = torch.logsumexp(teacher_imp_h, dim=1)    # (B,T)
        student_imp = torch.logsumexp(student_imp_h, dim=1)    # (B,T)
        student_imp=min_max_norm(student_imp)
        teacher_imp=min_max_norm(teacher_imp)
        loss_tad = F.mse_loss(student_imp,teacher_imp)
        
        # ---- dist loss----
        right_wrist_pos = sk_x[:, :, :, 10]  # (B,C,T)
        diff = torch.zeros_like(right_wrist_pos)
        diff[:, :, 1:] = right_wrist_pos[:, :, 1:] - right_wrist_pos[:, :, :-1]
        movement_target = torch.norm(diff, dim=1)  # (B,T)
        thresholds = torch.tensor([0.016, 0.041] , device=movement_target.device)
        dist_label = torch.bucketize(movement_target, thresholds).long()  # (B,T) in {0,1,2}
        dist_logits_flat = dist_logits.reshape(-1, 3)
        dist_label_flat = dist_label.reshape(-1)
        loss_dist_ce = F.cross_entropy(dist_logits_flat, dist_label_flat)

        
        precision1 = torch.exp(-self.loss_weights[0])
        l1 = precision1 * ce + self.loss_weights[0]

        precision2 = torch.exp(-self.loss_weights[1])
        l2 = precision2 * loss_kd + self.loss_weights[1]
        
        precision3 = torch.exp(-self.loss_weights[2])
        l3 = precision3 * loss_dist_ce + self.loss_weights[2]
        
        precision4 = torch.exp(-self.loss_weights[3])
        l4 = precision4 * loss_tad + self.loss_weights[3]
        
        loss = l1 + l2 +l3 + l4
        acc = self.calc_accuracy(s_logits, t)

        return {
            "loss": loss,
            "acc": acc,
            "ce": ce.detach(),
        }

    
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["imu_student"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat,_,_,_= self(x)
        y_hat = y_hat.squeeze(3)
        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        self.test_step_outputs.append(outputs)
        return outputs
# ----------------------------------------------------------------------
def train(cfg: DictConfig):
    logdir = Path(cfg.logdir)
    logger.debug(f"logdir = {logdir}")
    cleanup_dir(logdir, exclude="hydra")

    datamodule = OpenPackImuDataModule(cfg)
    plmodel = DeepConvLSTMLM(cfg)
    logger.info(plmodel)

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        mode=cfg.train.early_stop.mode,
        monitor=cfg.train.early_stop.monitor,
        filename="{epoch:02d}",
        verbose=False,
    )

    early_stop_callback = pytorch_lightning.callbacks.EarlyStopping(
        **cfg.train.early_stop,
    )

    pl_logger = pytorch_lightning.loggers.CSVLogger(logdir)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=500,
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

    device = torch.device("cuda:2")
    logdir = Path(cfg.logdir)

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
        devices=[0],
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    test_helper(cfg, datamodule, plmodel, trainer, logdir)



@hydra.main(
    version_base=None, config_path="./configs", config_name="deep-conv-lstm_U0105.yaml"
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