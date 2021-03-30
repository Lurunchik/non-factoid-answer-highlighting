import pathlib
from typing import Optional

from bertviz.neuron_view import get_attention
from bertviz.transformers_neuron_view import BertModel, BertTokenizer

from highlighting.attention import AttentionMap
from highlighting.utils import get_text_heatmap_html, html_replace, predict_highlighting_len

BASE_BERT_ARCH = 'bert-large-uncased-whole-word-masking'
DEFAULT_MODEL_PATH = str(pathlib.Path(__file__).parent) + '/data/nfl6_pretrained'


class BertHighlighter:
    _bert_architecture = BASE_BERT_ARCH
    _model_type = 'bert'

    def __init__(self, tokenizer: BertTokenizer, qa_model: BertModel):
        self._tokenizer = tokenizer
        self._model = qa_model

    @property
    def special_tokens(self):
        return self._tokenizer.special_tokens_map.values()

    @classmethod
    def from_pretrained(cls, model_path: str = DEFAULT_MODEL_PATH):
        return cls(
            tokenizer=BertTokenizer.from_pretrained(cls._bert_architecture, do_lower_case=True),
            qa_model=BertModel.from_pretrained(model_path),
        )

    def highlight(self, question: str, answer: str, color_fill_html: Optional[str] = '191,63,63'):
        attention = AttentionMap(
            attention_map=get_attention(
                self._model, self._model_type, self._tokenizer, html_replace(question), html_replace(answer),
            ),
            special_tokens=self.special_tokens,
        )

        target_highlight_len = predict_highlighting_len(len(attention.answer_tokens))
        top_tokens = attention.get_heaviest_tokens(exclude_stopwords=True)[:target_highlight_len]
        max_token_weight = max(top_tokens, key=lambda x: x[1])[1]
        token_weights = {k: w / float(max_token_weight) for k, w in top_tokens}

        all_answer_tokens = list(
            attention.iter_answer_tokens(ignore_tokens=self.special_tokens, merge_word_pieces=True)
        )
        return get_text_heatmap_html(
            answer, all_answer_tokens, [token_weights.get(t) for t in all_answer_tokens], color_fill_html,
        )


if __name__ == '__main__':
    highlighter = BertHighlighter.from_pretrained()

    print(
        highlighter.highlight(
            question='What is a computer microphone?',
            answer='''Microphone. A microphone is a device that captures audio by converting sound waves into an electrical signal. This signal can be amplified as an analog signal or may be converted to a digital signal, which can be processed by a computer or other digital audio device. ''',
            color_fill_html=None,
        )
    )
