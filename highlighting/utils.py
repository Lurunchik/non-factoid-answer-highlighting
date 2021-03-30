import logging
import math
import random
from string import punctuation
from typing import List, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

HTML_ESCAPE_TABLE = {
    '&': '&amp;',
    '"': '&quot;',
    "'": '&apos;',
    '>': '&gt;',
    '<': '&lt;',
}
EXCLUDED_PUNCTUATION = punctuation + ''')(\\/"\''''


def set_random_seed(seed_value: int = 146):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOGGER.info('Set random seed to %i', seed_value)


def html_replace(text):
    for c in HTML_ESCAPE_TABLE.values():
        text = text.replace(c, ' ')
    return text


def get_text_heatmap_html(text: str, tokens: List[str], weights: List[Optional[float]], max_alpha: float = 0.8):
    offset = 0
    processed_tokens = []
    for token, weight in zip(tokens, weights):
        start = text.lower().find(token, offset)
        if start < 0:
            raise ValueError(f'Cannot find "{token}" in "{text}" from position {offset}')

        token = text[start : start + len(token)]
        if weight is not None:
            highlighted_token = f'<span style="background-color:rgba(191,63,63,{weight / max_alpha});">{token}</span>'
            processed_tokens.append(highlighted_token)
        else:
            processed_tokens.append(token)

        offset = start + len(token)
        while offset < len(text) and text[offset] == ' ':
            processed_tokens.append(' ')
            offset += 1

    if offset < len(text):
        processed_tokens.append(text[offset + 1 :])

    return ''.join(processed_tokens)


_REGRESSION_INTERCEPT = 5.072644377132419
_REGRESSION_COEFFICIENT = 0.01133515
_REGRESSION_COEFFICIENT_FOR_SHORT_TEXTS = 0.2911335


def predict_number_of_highlighted_words(token_count: int) -> int:
    """
    Predict the number of tokens to highlight for a given token count using Piecewise Linear Regression

    Args:
        token_count: Number of tokens in a text

    Returns:
        Number of tokens to highlight
    """
    if token_count < 20:
        return math.ceil(_REGRESSION_COEFFICIENT_FOR_SHORT_TEXTS * token_count)
    return math.ceil(_REGRESSION_INTERCEPT + _REGRESSION_COEFFICIENT * token_count)


STOPWORDS = [
    'i',
    'me',
    'my',
    'myself',
    'we',
    'our',
    'ours',
    'ourselves',
    'you',
    "you're",
    "you've",
    "you'll",
    "you'd",
    'your',
    'yours',
    'yourself',
    'yourselves',
    'he',
    'him',
    'his',
    'himself',
    'she',
    "she's",
    'her',
    'hers',
    'herself',
    'it',
    "it's",
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    "that'll",
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'but',
    'if',
    'or',
    'because',
    'as',
    'until',
    'while',
    'of',
    'at',
    'by',
    'for',
    'with',
    'about',
    'against',
    'between',
    'into',
    'through',
    'during',
    'before',
    'after',
    'above',
    'below',
    'to',
    'from',
    'up',
    'down',
    'in',
    'out',
    'on',
    'off',
    'over',
    'under',
    'again',
    'further',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'all',
    'any',
    'both',
    'each',
    'few',
    'more',
    'most',
    'other',
    'some',
    'such',
    'no',
    'nor',
    'not',
    'only',
    'own',
    'same',
    'so',
    'than',
    'too',
    'very',
    's',
    't',
    'can',
    'will',
    'just',
    'don',
    "don't",
    'should',
    "should've",
    'now',
    'd',
    'll',
    'm',
    'o',
    're',
    've',
    'y',
    'ain',
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    'ma',
    'mightn',
    "mightn't",
    'mustn',
    "mustn't",
    'needn',
    "needn't",
    'shan',
    "shan't",
    'shouldn',
    "shouldn't",
    'wasn',
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't",
]
