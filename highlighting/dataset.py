import csv
import dataclasses
import json
import logging
import pathlib
import pickle
import random
import tempfile
from collections import defaultdict
from itertools import chain
from typing import IO, List, Sequence, Set, Tuple

import joblib
import numpy as np
import rarfile
import requests
import torch
import torch.utils.data
import tqdm
from transformers import BertTokenizer

from highlighting.data import ANTIQUE_IDS_PATH, DATA_FOLDER, STUDY_QUERIES_PATH
from highlighting.model import BASE_BERT_MODEL_NAME
from highlighting.utils import set_random_seed

LOGGER = logging.getLogger(__name__)

NFL6_DATASET_FILE = 'nfL6.json'
NFL6_DATASET_ARCHIVE = 'nfL6.json.rar'
NFL6_DATASET_URL = f'https://ciir.cs.umass.edu/downloads/nfL6/{NFL6_DATASET_ARCHIVE}'

TRAIN_FILE = 'train.joblib'
VAL_FILE = 'val.joblib'
TEST_FILE = 'test.joblib'

MAX_SEQUENCE_LENGTH = 128


@dataclasses.dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str
    label: int

    def encode(self, tokenizer: BertTokenizer) -> Tuple[np.ndarray, ...]:
        """
        Encode this example as BERT input

        Args:
            tokenizer: BERT tokenizer to use for encoding

        Returns:
            Tuple of encoded BERT inputs: label, input_ids, attention_mask, token_type_ids
        """
        inputs = tokenizer.encode_plus(
            self.question, self.answer, add_special_tokens=True, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True
        )

        label = np.array(self.label)
        input_ids = np.array(inputs.input_ids)
        attention_mask = np.array(inputs.attention_mask)
        token_type_ids = np.array(inputs.token_type_ids)

        return label, input_ids, attention_mask, token_type_ids


def _download_file(url: str, file: IO[bytes]) -> None:
    """
    Download a URL into a file

    Args:
        url: URL to download
        file: File-like object that support write
    """
    response = requests.get(url, stream=True)
    size = int(response.headers.get('content-length', 0))
    with tqdm.tqdm(total=size, unit='iB', unit_scale=True, desc='Downloading file') as progress:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            progress.update(len(chunk))


def download_nfl6_dataset(data_folder: pathlib.Path = DATA_FOLDER) -> None:
    """
    Download NFL6 dataset (https://ciir.cs.umass.edu/downloads/nfL6/index.html)

    Args:
        data_folder: Path to the folder where the dataset will be stored
    """
    dataset_path = data_folder / NFL6_DATASET_FILE

    if dataset_path.exists():
        LOGGER.info('NFL6 dataset found in %s ', dataset_path)
        return

    archive_path = data_folder / NFL6_DATASET_ARCHIVE

    if not archive_path.exists():
        with archive_path.open(mode='wb') as file:
            LOGGER.info('Loading NFL6 dataset archive from %s into %s', NFL6_DATASET_URL, archive_path)
            _download_file(url=NFL6_DATASET_URL, file=file)

    LOGGER.info('Unpacking NFL6 dataset from %s into %s', archive_path, dataset_path)
    with rarfile.RarFile(archive_path) as archive:
        archive.extractall(data_folder)

    archive_path.unlink(missing_ok=True)


def _get_test_ids() -> Set[str]:
    """
    Get unique IDs of test examples

    Returns:
        Set of test IDs
    """
    with STUDY_QUERIES_PATH.open(mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        test_ids = set(row['id'].split('_')[0] for row in reader)

    with ANTIQUE_IDS_PATH.open(mode='r') as file:
        test_ids.update(file.read().splitlines())

    return test_ids


def _generate_negatives(dataset: Sequence[QAExample], num_negatives: int = 1) -> List[QAExample]:
    """
    Generate negative examples for each of the positive examples in a dataset

    Args:
        dataset: Positive dataset
        num_negatives: Number of negatives to generate

    Returns:
        Negative examples
    """
    negative_dataset = []

    for positive in tqdm.tqdm(dataset, desc='Generating negatives'):
        negatives = []
        while len(negatives) < num_negatives:
            negative = random.choice(dataset)
            if negative.id != positive.id:
                negatives.append(QAExample(id=positive.id, question=positive.question, answer=negative.answer, label=0))

        negative_dataset.extend(negatives)

    return negative_dataset


def prepare_nfl6_dataset(data_folder: pathlib.Path = DATA_FOLDER, train_ratio: float = 0.8):
    """
    Prepare NFL6 dataset for BERT QA matching model training by splitting it into train/val/test and tokenizing

    Args:
        data_folder: Path to the folder where the dataset splits will be stored
        train_ratio:

    Returns:

    """
    dataset_path = data_folder / NFL6_DATASET_FILE
    if not dataset_path.exists():
        raise ValueError(f'You need to download NFL6 dataset into {data_folder} with {download_nfl6_dataset.__name__}')

    data_folder.mkdir(parents=True, exist_ok=True)

    train_path = data_folder / TRAIN_FILE
    val_path = data_folder / VAL_FILE
    test_path = data_folder / TEST_FILE

    if all(file.exists() for file in [train_path, val_path, test_path]):
        LOGGER.info('Joblib dataset files are already present in %s', data_folder)
        return

    LOGGER.info('Saving joblib dataset files to %s', data_folder)

    set_random_seed()

    with (data_folder / NFL6_DATASET_FILE).open('r', encoding='utf8') as f:
        yahoo_data = {q['id']: q for q in json.load(f)}

    test_ids = _get_test_ids()

    if any(i not in yahoo_data for i in test_ids):
        raise ValueError('Some test example IDs are missing, you might be using the wrong data')

    dataset = [QAExample(id=d['id'], question=d['question'], answer=d['answer'], label=1) for d in yahoo_data.values()]
    dataset += _generate_negatives(dataset)

    train_dataset = [ex for ex in dataset if ex.id not in test_ids]
    test_dataset = [ex for ex in dataset if ex.id in test_ids]

    train_dataset_by_ids = defaultdict(list)
    for ex in train_dataset:
        train_dataset_by_ids[ex.id].append(ex)

    train_ratio = int(train_ratio * len(train_dataset_by_ids))
    val_size = len(train_dataset_by_ids) - train_ratio

    train_ids, val_ids = torch.utils.data.random_split(list(train_dataset_by_ids), [train_ratio, val_size])

    train_dataset = list(chain.from_iterable(train_dataset_by_ids[t_id] for t_id in train_ids))
    val_dataset = list(chain.from_iterable(train_dataset_by_ids[t_id] for t_id in val_ids))

    LOGGER.info(
        'Train size: %i, validation size: %i, test size: %i', len(train_dataset), len(val_dataset), len(test_dataset)
    )

    LOGGER.info('Loading BERT tokenizer')
    tokenizer = BertTokenizer.from_pretrained(BASE_BERT_MODEL_NAME, do_lower_case=True)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset), (test_path, test_dataset)]:
        LOGGER.info('Tokenizing and saving %s', path)
        vectors = [ex.encode(tokenizer) for ex in tqdm.tqdm(dataset, desc='Tokenization')]
        joblib.dump(vectors, path, compress=True, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    download_nfl6_dataset()
    prepare_nfl6_dataset()
