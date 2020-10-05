import logging
import random

import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertForSequenceClassification
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

from qa_bert import QAMatchingBert

LOGGER = logging.getLogger(__name__)

BASE_BERT = 'bert-large-cased-whole-word-masking'
BATCH_SIZE = 20
NUM_LABELS = 2  # "answer" or "not an answer"

RANDOM_SEED = 146


def save(dataset, name):
    joblib.dump([list(map(np.array, t)) for t in dataset], f'{name}.joblib')


def load_dataset(name: str):
    return [tuple(map(torch.tensor, t)) for t in joblib.load(f'../{name}.joblib')]


def get_device():
    if torch.cuda.is_available():
        LOGGER.info('There are %d GPU(s) available.', torch.cuda.device_count())
        LOGGER.info('We will use GPU:', torch.cuda.get_device_name())
        return torch.device("cuda")

    LOGGER.info('No GPU available, using the CPU instead.')
    return torch.device("cpu")


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


def main():
    device = get_device()
    set_random_seed(RANDOM_SEED)

    datasets = []
    for t in ('train_cased', 'val_cased', 'test_cased'):
        LOGGER.info('Load dataset %s', t)
        datasets.append(load_dataset(t))

    train, val, test = datasets

    train_dataloader = DataLoader(
        train,
        sampler=RandomSampler(train),

        batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(
        val,
        sampler=SequentialSampler(val),
        batch_size=BATCH_SIZE)

    test_dataloader = DataLoader(
        test,
        sampler=SequentialSampler(test),
        batch_size=BATCH_SIZE)

    LOGGER.info('loading pretrained bert')
    bert_pretrained = BertForSequenceClassification.from_pretrained(BASE_BERT, num_labels=NUM_LABELS)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=1,
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
        gpus=[1],
        default_save_path='../highlighting/data/models/checkpoints/bert/nfL6_classification/',
        early_stop_callback=early_stop_callback,
    )

    model = QAMatchingBert(bert_pretrained, train_dataloader, val_dataloader, test_dataloader)

    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    main()
