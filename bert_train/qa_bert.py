from collections import OrderedDict
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, BertPreTrainedModel

NO_DECAY_PARAMETER_SUBSTRINGS = ("bias", "gamma", "beta")


class QAMatchingBert(pl.LightningModule):
    def __init__(self,
                 bert_pretrained_model: BertForSequenceClassification,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 weight_decay_rate: float = 0.01,
                 ):
        super().__init__()

        self.model = bert_pretrained_model

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

        self._weight_decay_rate = weight_decay_rate

    def configure_optimizers(self):
        parameters_with_decay = []
        parameters_without_decay = []
        for name, param in self.model.named_parameters():
            if any(substring in name for substring in NO_DECAY_PARAMETER_SUBSTRINGS):
                parameters_without_decay.append(param)
            else:
                parameters_with_decay.append(param)

        optimizer = AdamW(
            params=[
                {
                    "params": parameters_with_decay,
                    "weight_decay_rate": self._weight_decay_rate,
                },
                {
                    "params": parameters_without_decay,
                    "weight_decay_rate": 0.0
                },
            ],
            lr=2e-5,
        )
        return optimizer

    def forward(self, batch):
        labels, input_ids, attention_mask, token_type_ids = batch

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)

        progress_log = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": progress_log,
            "log": progress_log
        })

        return output

    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)

        labels, _, _, _ = batch

        labels_hat = torch.argmax(logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })
        return output

    def validation_end(self, outputs):
        val_acc = sum(out["correct_count"] for out in outputs).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum(out["val_loss"] for out in outputs) / len(outputs)
        progress_log = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        result = {"progress_bar": progress_log, "log": progress_log, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        labels_hat = torch.argmax(logits, dim=1)

        labels, _, _, _ = batch

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })

        return output

    def test_end(self, outputs):
        test_acc = sum(out["correct_count"] for out in outputs).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum(out["test_loss"] for out in outputs) / len(outputs)
        progress_log = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        result = {"progress_bar": progress_log, "log": progress_log}
        return result

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader
