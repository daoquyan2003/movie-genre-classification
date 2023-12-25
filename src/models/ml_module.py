import torch
from lightning import LightningModule
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from torchmetrics import MaxMetric, MeanMetric
from typing import Any, List

class MLLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.net = net
        self.criterion = criterion

        self.train_precision = MultilabelPrecision(num_labels=18, threshold=0.5, average='macro')
        self.val_precision = MultilabelPrecision(num_labels=18, threshold=0.5, average='macro')
        self.test_precision = MultilabelPrecision(num_labels=18, threshold=0.5, average='macro')

        self.train_recall = MultilabelRecall(num_labels=18, threshold=0.5, average='macro')
        self.val_recall = MultilabelRecall(num_labels=18, threshold=0.5, average='macro')
        self.test_recall = MultilabelRecall(num_labels=18, threshold=0.5, average='macro')

        self.train_f1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')
        self.val_f1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')
        self.test_f1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, img, title):
        return self.net(img, title)

    def on_train_start(self):
        self.val_loss.reset()

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

        self.val_precision_best.reset()
        self.val_recall_best.reset()
        self.val_f1_best.reset()

    def model_step(self, batch: Any):
        img, title, label = batch

        preds = self.forward(img, title)
        loss = self.criterion(preds, label)

        return loss, preds, label

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)

        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)

        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        # get current scores
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        # update best so far scores
        self.val_precision_best(precision)
        self.val_recall_best(recall)
        self.val_f1_best(f1)
        # log `val_f1_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val_precision_best", self.val_precision_best.compute(), prog_bar=True)
        self.log("val_recall_best", self.val_recall_best.compute(), prog_bar=True)
        self.log("val_f1_best", self.val_f1_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)

        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=filter(lambda p: p.requires_grad, self.trainer.model.parameters()))
        if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
        return {"optimizer": optimizer}