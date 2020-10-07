import math
from string import punctuation
from typing import List

HTML_ESCAPE_TABLE = {
    '&': '&amp;',
    '"': '&quot;',
    "'": '&apos;',
    '>': '&gt;',
    '<': '&lt;',
}
EXCLUDING_PUNCTUATION = punctuation + ''')(\\/"\''''


def html_replace(text):
    for c in HTML_ESCAPE_TABLE.values():
        text = text.replace(c, ' ')
    return text


def get_text_heatmap_html(text: str, tokens: List[str], weights: List[float], color: str = '191,63,63'):
    max_alpha = 0.8
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


def predict_highlighting_len(token_count: int):
    """
    Piecewise Linear Regression
    :param token_count: int
    :return: int highlighted
    """
    intercept_ = 5.072644377132419
    coef_ = 0.01133515
    small_coef_ = 0.2911335

    if token_count < 20:
        return math.ceil(small_coef_ * token_count)
    return math.ceil(intercept_ + token_count * coef_)


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
