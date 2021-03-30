import pathlib
from collections import defaultdict
from enum import Enum
from string import punctuation
from typing import Optional, Set

import numpy as np

from bertviz.neuron_view import get_attention
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from highlighting.utils import STOPWORDS, html_replace, predict_highlighting_len

BASE_BERT_ARCH = 'bert-large-uncased-whole-word-masking'
DEFAULT_MODEL_PATH = str(pathlib.Path(__file__).parent) + '/data/nfl6_pretrained'

_EXCLUDING_PUNCTUATION = punctuation + ''')(\\/"\''''
max_alpha = 0.8


def _get_text_heatmap_html(text, tokens, weights, color='191,63,63'):
    offset = len(tokens[0])
    result_tokens = [text[:offset]]

    i = 1
    while i < len(tokens):
        token = tokens[i]
        weight = weights[i]
        i += 1

        start = text.lower().find(token, offset)
        assert start > 0, (text, token, offset)

        orig_token = text[start : start + len(token)]
        offset = start + len(token)

        if weight is not None:
            if color is not None:
                orig_token = (
                    f'<span style="background-color:rgba({color},'
                    + str(weight / max_alpha)
                    + ');">'
                    + orig_token
                    + '</span>'
                )
            else:
                orig_token = f'**' + orig_token + '**'

        result_tokens.append(orig_token)

        j = 0
        while offset + j < len(text) and text[offset + j] == ' ':
            result_tokens.append(' ')
            j += 1

    if offset < len(text):
        result_tokens.append(text[offset + 1 :])

    return ''.join(result_tokens)


def _iter_answer_tokens(
    att_res, ignore_tokens: Set[str] = None, merge_word_pieces: bool = False
):
    ignore_tokens = ignore_tokens or set()
    answer_tokens = att_res['ab']['right_text']
    i = 0
    while i < len(answer_tokens):
        token = answer_tokens[i]
        if token not in ignore_tokens:
            if merge_word_pieces:
                j = i + 1
                while j < len(answer_tokens) and answer_tokens[j].startswith('##'):
                    token += answer_tokens[j].lstrip('##')
                    j += 1
                i = j - 1
            yield token
        i += 1


def calculate_mean_attn(att_res, ignore_tokens: Set[str] = None):
    filtered_tokens = list(_iter_answer_tokens(att_res, ignore_tokens, True))
    mean_wghts = np.zeros(
        (
            len(att_res['ab']['attn']),
            len(att_res['ab']['attn'][0]),
            len(filtered_tokens),
        )
    )

    for attention in [att_res['ab']['attn']]:
        for layer, data in enumerate(attention):
            for head, word_weights in enumerate(data):
                token_index = 0
                for i, w in enumerate(_iter_answer_tokens(att_res, ignore_tokens)):
                    query_weights = np.sum(att_res['aa']['attn'][layer][head], axis=0)
                    cur_attn = np.mean(
                        np.array([x[i] for x in word_weights]) * query_weights
                    )

                    if w.startswith('##'):
                        mean_wghts[layer][head][token_index - 1] = np.mean(
                            [mean_wghts[layer][head][token_index - 1], cur_attn]
                        )
                    else:
                        mean_wghts[layer][head][token_index] = np.mean(
                            [mean_wghts[layer][head][token_index], cur_attn]
                        )
                        token_index += 1

    return filtered_tokens, mean_wghts


class BertHighlighter:
    _bert_architecture = BASE_BERT_ARCH
    _model_type = 'bert'

    def __init__(self, tokenizer: BertTokenizer, qa_model: BertModel):
        self._tokenizer = tokenizer
        self._model = qa_model
        self._ignore_tokens = set(_EXCLUDING_PUNCTUATION)
        self._ignore_tokens.update(self.special_tokens)

    @property
    def special_tokens(self):
        return self._tokenizer.special_tokens_map.values()

    @classmethod
    def from_pretrained(cls, model_path=DEFAULT_MODEL_PATH):
        return cls(
            tokenizer=BertTokenizer.from_pretrained(
                cls._bert_architecture, do_lower_case=True
            ),
            qa_model=BertModel.from_pretrained(model_path),
        )

    def highlight(
        self, question: str, answer: str, color_fill_html: Optional[str] = '191,63,63'
    ):
        att_res = get_attention(
            self._model,
            self._model_type,
            self._tokenizer,
            html_replace(question),
            html_replace(answer),
        )

        answer_tokens, weights = calculate_mean_attn(att_res, self._ignore_tokens)

        top_tokens = self._get_sorted_tokens(answer_tokens, weights)[
            : predict_highlighting_len(len(answer_tokens))
        ]
        max_token_weight = max(top_tokens, key=lambda x: x[1])[1]
        token_weights = {k: w / float(max_token_weight) for k, w in top_tokens}

        all_answer_tokens = list(
            _iter_answer_tokens(
                att_res, ignore_tokens=self.special_tokens, merge_word_pieces=True
            )
        )
        return _get_text_heatmap_html(
            answer,
            all_answer_tokens,
            [token_weights.get(t) for t in all_answer_tokens],
            color_fill_html,
        )

    def _get_sorted_tokens(
        self,
        tokens,
        attn,
        exclude_stopwords: bool = True,
        layer: int = None,
        head: int = None,
    ):
        if layer is None:
            layer = -1

        if head is None:
            vec = np.mean(attn[layer], axis=0)
        else:
            vec = attn[layer][head]

        assert len(tokens) == len(vec)

        tokens_top = defaultdict(int)

        for t, v in zip(tokens, vec):
            add_token = not exclude_stopwords or t not in STOPWORDS
            if add_token:
                tokens_top[t] += v

        return sorted(tokens_top.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    highlighter = BertHighlighter.from_pretrained()

    print(
        highlighter.highlight(
            question='What is a computer microphone?',
            answer='''Microphone. A microphone is a device that captures audio by converting sound waves into an electrical signal. This signal can be amplified as an analog signal or may be converted to a digital signal, which can be processed by a computer or other digital audio device. ''',
            color_fill_html=None,
        )
    )
