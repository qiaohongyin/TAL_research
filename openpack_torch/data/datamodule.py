import copy
from logging import getLogger
from typing import Dict, List, Optional,Type, Any
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

log = getLogger(__name__)

class OpenPackBaseDataModule(pl.LightningDataModule):

    dataset_class: Type[Any]

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.num_workers = 3
        self.batch_size = 32

    def _init_datasets(
        self,
        user_session: list[tuple[int, int]],
    ) -> dict[str, torch.utils.data.Dataset]:

        datasets = dict()
        for user, session in user_session:
            key = f"{user}-{session}"
            datasets[key] = self.dataset_class(
                copy.deepcopy(self.cfg), [(user, session)]
            )
        return datasets

    def setup(self, stage: Optional[str] = None) -> None:

        split = self.cfg.dataset.split
        if stage in (None, "fit"):
            self.op_train = self.dataset_class(self.cfg, split.train)
        else:
            self.op_train = None

        if stage in (None, "fit", "validate"):
            self.op_val = self._init_datasets(split.val)
        else:
            self.op_val = None

        if stage in (None, "test"):
            self.op_test = self._init_datasets(split.test)
        else:
            self.op_test = None

    def train_dataloader(self) -> DataLoader:
        assert self.op_train is not None
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> List[DataLoader]:
        assert self.op_val is not None
        dataloaders = []
        for key, dataset in self.op_val.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            )
        return dataloaders

    def test_dataloader(self) -> List[DataLoader]:
        assert self.op_test is not None
        dataloaders = []
        for key, dataset in self.op_test.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            )
        return dataloaders
