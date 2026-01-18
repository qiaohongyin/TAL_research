from logging import getLogger
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy as accuracy_score

logger = getLogger(__name__)

class BaseLightningModule(pl.LightningModule):
    test_step_outputs: list[dict[str, torch.Tensor]]
    test_results: dict[str, np.ndarray]

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.net = self.init_model(cfg)
        self.criterion = self.init_criterion(cfg)

        self.test_step_outputs = []
        self.test_results = {}

    def init_model(self, cfg):
        raise NotImplementedError()

    def init_criterion(self, cfg):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,  weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)  
        return {"optimizer": optimizer,"lr_scheduler": scheduler,}


    def calc_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        preds = F.softmax(y, dim=1)
        (batch_size, num_classes, window_size) = preds.size()
        preds_flat = preds.permute(1, 0, 2).reshape(
            num_classes, batch_size * window_size
        )
        t_flat = t.reshape(-1)

        # FIXME: I want to use macro average score.
        ignore_index = num_classes - 1
        acc = accuracy_score(
            preds_flat.transpose(0, 1),
            t_flat,
            task="multiclass",
            average="weighted",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_val_common_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        output = self.train_val_common_step(batch, batch_idx)

        train_output = {f"train_{key}": val for key, val in output.items()}
        self.log_dict(
            train_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return output

    def validation_step(
        self, batch: Dict, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict:
        output = self.train_val_common_step(batch, batch_idx)

        train_output = {f"val_{key}": val for key, val in output.items()}
        self.log_dict(
            train_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return output

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            raise ValueError(
                "Size of test_step_outputs is 0. Did you forgot to call "
                "`self.test_step_outputs.append(outputs)` in test_step()?"
            )

        outputs = self.test_step_outputs
        keys = tuple(outputs[0].keys())

        results: dict[str, list[np.ndarray]] = {key: [] for key in keys}

        for d in outputs:
            for key in keys:
                results[key].append(d[key].detach().cpu().numpy())

        final_results: dict[str, np.ndarray] = {}
        for key in keys:
            final_results[key] = np.concatenate(results[key], axis=0)

        self.test_results = final_results


    def clear_test_outputs(self):
        self.test_step_outputs = []
        self.test_results = {}

