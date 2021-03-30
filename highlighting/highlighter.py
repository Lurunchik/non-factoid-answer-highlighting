import pathlib

from bertviz.neuron_view import get_attention
from bertviz.transformers_neuron_view import BertModel, BertTokenizer

from highlighting.attention import AttentionMap
from highlighting.data import PRETRAINED_MODEL_FOLDER
from highlighting.model import BASE_BERT_MODEL_NAME
from highlighting.utils import get_text_heatmap_html, html_replace, predict_number_of_highlighted_words

BERT_MODEL_TYPE = 'bert'


class BertHighlighter:
    """
    BERT token highlighter that uses attention map to highlight potentially important tokens
    """

    def __init__(self, tokenizer: BertTokenizer, model: BertModel):
        """
        Args:
            tokenizer: BERT tokenizer to use
            model: Pre-trained question/answer pair matching BERT model
        """
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_pretrained(cls, model_path: pathlib.Path = PRETRAINED_MODEL_FOLDER):
        return cls(
            tokenizer=BertTokenizer.from_pretrained(BASE_BERT_MODEL_NAME, do_lower_case=True),
            model=BertModel.from_pretrained(str(model_path)),
        )

    def highlight(self, question: str, answer: str):
        special_tokens = self._tokenizer.special_tokens_map.values()

        attention = AttentionMap(
            attention_map=get_attention(
                model=self._model,
                model_type=BERT_MODEL_TYPE,
                tokenizer=self._tokenizer,
                sentence_a=html_replace(question),
                sentence_b=html_replace(answer),
            ),
            special_tokens=special_tokens,
        )

        target_highlight_len = predict_number_of_highlighted_words(len(attention.answer_tokens))
        top_tokens = attention.get_heaviest_tokens(exclude_stopwords=True)[:target_highlight_len]
        max_token_weight = max(top_tokens, key=lambda x: x[1])[1]
        token_weights = {k: w / float(max_token_weight) for k, w in top_tokens}

        all_answer_tokens = list(attention.iter_answer_tokens(ignored_tokens=special_tokens, merge_subtokens=True))
        return get_text_heatmap_html(
            text=answer, tokens=all_answer_tokens, weights=[token_weights.get(t) for t in all_answer_tokens],
        )


if __name__ == '__main__':
    highlighter = BertHighlighter.from_pretrained()

    print(
        highlighter.highlight(
            question='What is a computer microphone?',
            answer='Microphone. A microphone is a device that captures audio by converting sound waves into an '
            'electrical signal. This signal can be amplified as an analog signal or may be converted to a '
            'digital signal, which can be processed by a computer or other digital audio device. ',
        )
    )
