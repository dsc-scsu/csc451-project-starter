"""Helpers for short-text data such as tweets.

notebooks/05_text_data.ipynb uses these, in this order:

    1. clean_text      lower case; replace links and @names with tags
    2. tokenize        split a text into tokens (words, hashtags, emoji)
    3. build_vocab     the words the model knows, from TRAINING texts only
    4. encode          turn a text into a fixed-length row of word numbers
    5. make_text_model BagOfWords (the baseline) or TextCNN

app/app_text.py repeats clean_text, tokenize and encode. If you change
them here, change them there too, or the app reads text differently
from training (train/serve skew).
"""

import re
from collections import Counter

import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"  # word numbers 0 and 1


# ---------------------------------------------------------------------
# 1. Clean
# ---------------------------------------------------------------------
def clean_text(text):
    """Lower case, and replace parts that are unique to one tweet.

    https://t.co/abc  ->  <url>     every link looks different
    @someone          ->  <user>    so does every user name
    RT at the start   ->  removed   marks a retweet, not content
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"^rt\b", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    return " ".join(text.split())  # one space between words


# ---------------------------------------------------------------------
# 2. Tokenize
# ---------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"<url>|<user>|#?\w+(?:'\w+)?|[^\w\s]")


def tokenize(text):
    """Split a cleaned text into tokens.

    "loved it!! #happy 😀"  ->  ["loved", "it", "!", "!", "#happy", "😀"]

    A token is a word (with an apostrophe, like "don't"), a hashtag,
    a tag such as <url>, or a single symbol (punctuation or an emoji).
    """
    return TOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------
# 3. Vocabulary: from the training texts only
# ---------------------------------------------------------------------
def build_vocab(train_texts, max_words=10000, min_count=2):
    """Return the list of known tokens; a token's position is its number.

    Only the training texts are counted. Counting validation or test
    texts too would let them leak into training (Week 13).
    Tokens seen fewer than `min_count` times are left out: the model
    cannot learn much from a word it saw once.
    """
    counts = Counter()
    for text in train_texts:
        counts.update(tokenize(text))
    common = [w for w, c in counts.most_common() if c >= min_count]
    return [PAD, UNK] + common[:max_words - 2]


# ---------------------------------------------------------------------
# 4. Encode: text -> a row of word numbers, all the same length
# ---------------------------------------------------------------------
def encode(text, word_to_idx, max_len):
    """Turn a cleaned text into exactly `max_len` word numbers.

    Unknown tokens become 1 (<unk>). Longer texts are cut; shorter
    texts are padded with 0 (<pad>) so a batch forms one tensor.
    """
    ids = [word_to_idx.get(t, 1) for t in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


def encode_all(texts, word_to_idx, max_len):
    """Encode many texts into one tensor of shape (N, max_len)."""
    return torch.tensor([encode(t, word_to_idx, max_len) for t in texts])


# ---------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------
class BagOfWords(nn.Module):
    """Baseline: the average of the text's word vectors, then a layer.

    Word order is ignored: "not good, bad" and "bad, not good" give the
    same result. Padding (word number 0) is left out of the average.
    """

    def __init__(self, vocab_size, n_classes, embed_dim=64):
        super().__init__()
        self.bag = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean",
                                   padding_idx=0)
        self.out = nn.Linear(embed_dim, n_classes)

    def forward(self, ids):          # ids: (N, max_len)
        x = self.bag(ids)            # (N, embed_dim)
        return self.out(torch.relu(x))


class TextCNN(nn.Module):
    """A 1D CNN over the words: Week 11's ideas applied to a sequence.

    Embedding: each word number becomes a vector of `embed_dim` numbers,
               learned in training. These are the input channels.
    Conv1d:    each filter slides over 3 neighboring words at a time and
               detects a short phrase, such as "not good at".
    Max pool:  keeps each filter's strongest match anywhere in the text
               (global max pooling).
    """

    def __init__(self, vocab_size, n_classes, embed_dim=64, filters=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, filters, kernel_size=3,
                              padding=1)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(filters, n_classes)

    def forward(self, ids):          # ids: (N, max_len)
        x = self.embed(ids)          # (N, max_len, embed_dim)
        x = x.transpose(1, 2)        # (N, embed_dim, max_len): channels
        x = torch.relu(self.conv(x))   # (N, filters, max_len)
        x = x.max(dim=2).values      # (N, filters): strongest match
        return self.out(self.drop(x))


def make_text_model(name, vocab_size, n_classes, embed_dim=64):
    if name == "bag_of_words":
        return BagOfWords(vocab_size, n_classes, embed_dim)
    if name == "text_cnn":
        return TextCNN(vocab_size, n_classes, embed_dim)
    raise ValueError('MODEL must be "bag_of_words" or "text_cnn"')
