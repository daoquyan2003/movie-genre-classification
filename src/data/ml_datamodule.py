from typing import Any, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator

from data.components.ml_dataset import MLDataset

class MLDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/ml1m",
                 text_pretrained: str = "bert-base-uncased",
                 train_val_ratio: Tuple[int, int] = None,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 transforms: Any = None,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.setup_called = False

    @property
    def num_classes(self) -> int:
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.setup_called:
            self.setup_called = True
            train_dataset = MLDataset(
                data_dir=self.hparams.data_dir,
                is_train=True,
                text_pretrained=self.hparams.text_pretrained,
                transform=self.hparams.transforms
            )
            if not self.hparams.train_val_ratio:
                total_samples = len(train_dataset)
                val_size = int(0.15 * total_samples)
                train_size = total_samples - val_size
                self.data_train, self.data_valid = random_split(
                    dataset=train_dataset,
                    lengths=[train_size, val_size],
                    generator=Generator().manual_seed(42)
                )
            else:
                total_samples = len(train_dataset)
                val_size = int(self.hparams.train_val_ratio[1] * total_samples)
                train_size = total_samples - val_size
                self.data_train, self.data_valid = random_split(
                    dataset=train_dataset,
                    lengths=[train_size, val_size],
                    generator=Generator().manual_seed(42)
                )

            self.data_test = MLDataset(
                data_dir=self.hparams.data_dir,
                is_train=False,
                text_pretrained=self.hparams.text_pretrained,
                transform=self.hparams.transforms
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )