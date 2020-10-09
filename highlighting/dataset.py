import json
import os
import pathlib
import random
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import rarfile
import requests
import torch
import torch.utils
import tqdm
from transformers import BertTokenizer

from highlighting import DATA_FOLDER
from highlighting.model import BASE_BERT

DATASET_LINK = 'https://ciir.cs.umass.edu/downloads/nfL6/nfL6.json.rar'
NFL6_FILENAME = 'nfL6.json'
MAX_LEN = 128


def load_dataset(data_folder: pathlib.Path = DATA_FOLDER):
    if os.path.exists(data_folder / NFL6_FILENAME):
        print(f'Dataset found in {data_folder / NFL6_FILENAME}')
        return

    rarname = 'dataset.json.rar'
    if not os.path.exists(data_folder / rarname):
        print(f'... Loading dataset archive from {DATASET_LINK} into {rarname}')
        r = requests.get(DATASET_LINK)
        with open(data_folder / rarname, 'wb') as f:
            f.write(r.content)

    print(f'... Unpacking dataset into {NFL6_FILENAME}')

    with rarfile.RarFile(data_folder / rarname) as f:
        f.extractall(data_folder)

    os.remove(data_folder / rarname)


def _get_test_ids():
    """
    We want to
    """
    study_queries = pd.read_csv(DATA_FOLDER / 'queries_for_study_from_chhir2019.csv', sep=';')
    test_ids = set(l.split('_')[0] for l in list(study_queries.id))

    with open(DATA_FOLDER / 'antique_ids.txt') as f:
        test_ids.update(q_id.strip() for q_id in f)

    return test_ids


def prepare_dataset(data_folder: pathlib.Path = DATA_FOLDER, force: bool = False, train_percent: float = 0.8):
    for fname in ('train', 'val', 'test'):
        if not os.path.exists(DATA_FOLDER / f'{fname}.joblib'):
            print(f'joblib dataset files are not found in {DATA_FOLDER}')
            break
    else:
        print(f'training dataset files are found in {DATA_FOLDER}')
        return
    load_dataset(data_folder)
    with open(data_folder / NFL6_FILENAME, 'r', encoding='utf8') as f:
        yahoo_data = {q['id']: q for q in json.load(f)}

    dataset = [(d['id'], d['question'], d['answer'], 1) for d in yahoo_data.values()]

    negatives_traning_data = []

    for q in tqdm.tqdm(dataset):
        while True:
            negative = random.choice(dataset)
            if negative[0] != q[0]:
                break
        negatives_traning_data.append((q[0], q[1], negative[2], 0))

    dataset += negatives_traning_data

    test_ids = _get_test_ids()
    if not force:
        assert all(x in yahoo_data for x in test_ids), 'You are probably using different dataset, use force=True option'

    train_dataset, test_dataset = [t for t in dataset if t[0] not in test_ids], [t for t in dataset if t[0] in test_ids]

    train_dataset_by_ids = defaultdict(list)

    for q in train_dataset:
        train_dataset_by_ids[q[0]].append(q)

    assert all(len(q) == 2 for q in train_dataset_by_ids.values())
    assert len(train_dataset_by_ids) + len(test_ids) == len(yahoo_data)

    train_size = int(train_percent * len(train_dataset_by_ids))
    val_size = len(train_dataset_by_ids) - train_size

    train_data_ids, val_data_ids = torch.utils.data.random_split(
        list(train_dataset_by_ids.keys()), [train_size, val_size]
    )
    print(f'Train part: {len(train_data_ids)}, validation: {len(val_data_ids)}, test: {len(test_ids)}')

    train_data = list(chain.from_iterable(train_dataset_by_ids[t_id] for t_id in train_data_ids))
    val_data = list(chain.from_iterable(train_dataset_by_ids[t_id] for t_id in val_data_ids))

    assert len(set(q[0] for q in train_data) & set(q[0] for q in val_data)) == 0
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(BASE_BERT, do_lower_case=False)

    for name, part in {'train': train_data, 'val': val_data, 'test': test_dataset}.items():
        if name == 'test':
            print(f'Saving {name}.joblib to {DATA_FOLDER}')
            vectors = list(map(partial(preprocess, tokenizer), tqdm.tqdm(part)))
            joblib.dump([list(map(np.array, t)) for t in vectors], DATA_FOLDER / f'{name}.joblib')


def preprocess(tokenizer: BertTokenizer, data: Tuple[int, str, str, int]):
    _, query, answer, label = data
    inputs = tokenizer.encode_plus(query, answer, add_special_tokens=True, max_length=MAX_LEN, )

    input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
    attention_mask = [1] * len(input_ids)
    padding_length = MAX_LEN - len(input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    assert len(input_ids) == MAX_LEN, 'Error with input length {} vs {}'.format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, 'Error with input length {} vs {}'.format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, 'Error with input length {} vs {}'.format(len(token_type_ids), MAX_LEN)

    label = torch.tensor(label)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    return (label, input_ids, attention_mask, token_type_ids)


if __name__ == '__main__':
    prepare_dataset()
