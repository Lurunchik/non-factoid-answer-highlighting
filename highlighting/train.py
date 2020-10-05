import logging
import pathlib
import random
from typing import List

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification

from highlighting.model import QAMatchingBert

LOGGER = logging.getLogger(__name__)

BASE_BERT = 'bert-large-cased-whole-word-masking'
BATCH_SIZE = 20
NUM_LABELS = 2  # "answer" or "not an answer"

RANDOM_SEED = 146


def save(dataset, name):
    joblib.dump([list(map(np.array, t)) for t in dataset], f'{name}.joblib')


def load_dataset(path: pathlib.Path):
    LOGGER.info('Load data from %s', path)
    return [tuple(map(torch.tensor, t)) for t in joblib.load(path)]


def get_device():
    if torch.cuda.is_available():
        LOGGER.info('There are %d GPU(s) available.', torch.cuda.device_count())
        LOGGER.info('Using GPU:', torch.cuda.get_device_name())
        return torch.device('cuda')

    LOGGER.info('No GPU available, using the CPU instead.')
    return torch.device('cpu')


def set_random_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOGGER.info(f'set random seed to {seed_value}')


def train(
    gpus: List[int] = (1,),
    train_path: pathlib.Path = pathlib.Path('train.joblib'),
    val_path: pathlib.Path = pathlib.Path('val.joblib'),
    test_path: pathlib.Path = pathlib.Path('test.joblib'),
):
    # device = get_device()
    set_random_seed(RANDOM_SEED)

    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)
    test_dataset = load_dataset(test_path)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
    )
    val_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE
    )

    LOGGER.info('Loading pretrained bert')
    bert_pretrained = BertForSequenceClassification.from_pretrained(
        BASE_BERT, num_labels=NUM_LABELS
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.0, patience=1, verbose=True, mode='min'
    )

    trainer = pl.Trainer(
        gpus=gpus,
        default_save_path='../data/models/checkpoints/bert/nfL6_classification/',
        early_stop_callback=early_stop_callback,
    )

    model = QAMatchingBert(
        bert_pretrained, train_dataloader, val_dataloader, test_dataloader
    )

    trainer.fit(model)
    trainer.test()
