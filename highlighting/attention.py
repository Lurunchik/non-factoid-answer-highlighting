from collections import defaultdict
from typing import Dict, Iterator, Optional, Sequence, Set

import numpy as np

from highlighting.utils import EXCLUDED_PUNCTUATION, STOPWORDS


class AttentionMap:
    def __init__(self, attention_map: Dict[str, Dict[str, np.ndarray]], special_tokens: Sequence[str] = ()):
        """
        Args:
            attention_map: Attention map produced by `bertviz.neuron_view.get_attention`
                A dictionary of attentions with the following structure is expected:
                {
                    'all': All attention (source = AB, target = AB)
                    'aa': Sentence A self-attention (source = A, target = A) (if sentence_b is not None)
                    'bb': Sentence B self-attention (source = B, target = B) (if sentence_b is not None)
                    'ab': Sentence A -> Sentence B attention (source = A, target = B) (if sentence_b is not None)
                    'ba': Sentence B -> Sentence A attention (source = B, target = A) (if sentence_b is not None)
                }
                where each value is a dictionary:
                {
                    'left_text': list of source tokens, to be displayed on the left of the vis
                    'right_text': list of target tokens, to be displayed on the right of the vis
                    'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
                    'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
                    'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
                }
            special_tokens: Sequence of special tokens used in the model, these tokens are ignored in the attention map
        """
        self._map = attention_map
        self._ignore_tokens = set(EXCLUDED_PUNCTUATION).union(special_tokens)
        self._filtered_tokens, self._token_weights = self._calculate_mean_attn()

    @property
    def answer_tokens(self):
        return self._map['ab']['right_text']

    def iter_answer_tokens(
        self, ignored_tokens: Optional[Set[str]] = None, merge_subtokens: bool = False
    ) -> Iterator[str]:
        answer_tokens = self.answer_tokens
        ignored_tokens = ignored_tokens or set()

        i = 0
        while i < len(answer_tokens):
            token = answer_tokens[i]
            if token not in ignored_tokens:
                if merge_subtokens:
                    j = i + 1
                    while j < len(answer_tokens) and answer_tokens[j].startswith('##'):
                        token += answer_tokens[j].lstrip('#')
                        j += 1
                    i = j - 1
                yield token
            i += 1

    def get_heaviest_tokens(
        self, exclude_stopwords: bool = True, layer: int = None, head: int = None,
    ):
        if layer is None:
            layer = -1

        if head is None:
            vec = np.mean(self._token_weights[layer], axis=0)
        else:
            vec = self._token_weights[layer][head]

        assert len(self._filtered_tokens) == len(vec)

        tokens_top = defaultdict(int)

        for t, v in zip(self._filtered_tokens, vec):
            add_token = not exclude_stopwords or t not in STOPWORDS
            if add_token:
                tokens_top[t] += v

        return sorted(tokens_top.items(), key=lambda x: x[1], reverse=True)

    def _calculate_mean_attn(self):
        ignore_tokens = self._ignore_tokens

        filtered_tokens = list(self.iter_answer_tokens(ignore_tokens, merge_subtokens=True))
        mean_wghts = np.zeros((len(self._map['ab']['attn']), len(self._map['ab']['attn'][0]), len(filtered_tokens),))

        for attention in [self._map['ab']['attn']]:
            for layer, data in enumerate(attention):
                for head, word_weights in enumerate(data):
                    token_index = 0
                    for i, w in enumerate(self.iter_answer_tokens(ignore_tokens, merge_subtokens=False)):
                        query_weights = np.sum(self._map['aa']['attn'][layer][head], axis=0)
                        cur_attn = np.mean(np.array([x[i] for x in word_weights]) * query_weights)

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
