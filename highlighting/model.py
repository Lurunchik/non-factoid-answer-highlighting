from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification

NO_DECAY_PARAMETER_SUBSTRINGS = ('bias', 'gamma', 'beta')

# labels, input_ids, attention_mask, token_type_ids
BertInput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# loss, logits
BertOutput = Tuple[torch.Tensor, torch.Tensor]


class QAMatchingBert(pl.LightningModule):
    def __init__(
        self,
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

    def _get_metrics(self, batch: BertInput) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run prediction on a batch with known labels and return evaluation metrics
        """
        loss, logits = self.forward(batch)
        labels_hat = torch.argmax(logits, dim=1)
        labels, _, _, _ = batch
        accuracy = torch.mean(labels == labels_hat)
        return labels, loss, accuracy

    def configure_optimizers(self):
        parameters_with_decay: List[torch.nn.Parameter] = []
        parameters_without_decay: List[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if any(substring in name for substring in NO_DECAY_PARAMETER_SUBSTRINGS):
                parameters_without_decay.append(param)
            else:
                parameters_with_decay.append(param)

        optimizer = AdamW(
            params=[
                {'params': parameters_with_decay, 'weight_decay_rate': self._weight_decay_rate},
                {'params': parameters_without_decay, 'weight_decay_rate': 0.0},
            ],
            lr=2e-5,
        )
        return optimizer

    def forward(self, batch: BertInput) -> BertOutput:
        labels, input_ids, attention_mask, token_type_ids = batch

        loss, logits = self.model(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,
        )
        return loss, logits

    def training_step(self, batch: BertInput, batch_idx: int):
        loss, _ = self.forward(batch)

        progress_log = {'train_loss': loss}
        output = {'loss': loss, 'progress_bar': progress_log, 'log': progress_log}
        return output

    def validation_step(self, batch: BertInput, batch_idx: int):
        labels, loss, accuracy = self._get_metrics(batch)

        output = {
            'val_loss': loss,
            'val_acc': accuracy,
        }
        return output

    def validation_end(self, outputs: List[Dict[str, torch.Tensor]]):
        val_acc = sum(out['val_acc'] for out in outputs) / len(outputs)
        val_loss = sum(out['val_loss'] for out in outputs) / len(outputs)
        progress_log = {
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        result = {
            'progress_bar': progress_log,
            'log': progress_log,
            'val_loss': val_loss,
        }
        return result

    def test_step(self, batch: BertInput, batch_idx: int):
        labels, loss, accuracy = self._get_metrics(batch)

        output = {
            'test_loss': loss,
            'test_acc': accuracy,
        }
        return output

    def test_end(self, outputs: List[Dict[str, torch.Tensor]]):
        test_acc = sum(out['test_acc'] for out in outputs) / len(outputs)
        test_loss = sum(out['test_loss'] for out in outputs) / len(outputs)
        progress_log = {
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        result = {
            'progress_bar': progress_log,
            'log': progress_log,
            'test_loss': test_loss,
        }
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
