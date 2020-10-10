import logging
import pathlib
from typing import List, Optional

import joblib
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification

from highlighting.data import DATA_FOLDER
from highlighting.dataset import TEST_FILE, TRAIN_FILE, VAL_FILE
from highlighting.model import BASE_BERT_MODEL_NAME, QAMatchingBert
from highlighting.utils import set_random_seed

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 20
NUM_LABELS = 2  # "answer" or "not an answer"


class JoblibDataset(Dataset):
    """
    Dataset that loads numpy arrays from a joblib file and converts them to tensors
    """

    def __init__(self, path: pathlib.Path):
        LOGGER.info('Load data from %s', path)
        self._data = [tuple(map(torch.tensor, inputs)) for inputs in joblib.load(path)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]


def train_model(data_folder: pathlib.Path = DATA_FOLDER, gpus: Optional[List[int]] = None):
    """
    Train BERT model on a question/answer pair matching task

    Args:
        data_folder: Folder where pre-tokenized training data is stored
        gpus: Which GPUs to use for training, set to `None` for training on CPU
    """
    set_random_seed()

    train_dataset = JoblibDataset(data_folder / TRAIN_FILE)
    val_dataset = JoblibDataset(data_folder / VAL_FILE)
    test_dataset = JoblibDataset(data_folder / TEST_FILE)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    LOGGER.info('Loading pretrained bert')
    bert_pretrained = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL_NAME, num_labels=NUM_LABELS)

    model = QAMatchingBert(
        bert_pretrained_model=bert_pretrained,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )

    trainer = pl.Trainer(
        gpus=gpus,
        default_save_path=DATA_FOLDER / 'models' / 'checkpoints' / 'bert' / 'nfL6_classification',
        early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=0.0, patience=1, verbose=True, mode='min'),
    )
    trainer.fit(model)
    trainer.test()
