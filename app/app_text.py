# CSC 451 - Streamlit app for a short-text model (tweets, reviews)
#
# Run on your laptop, from the app/ folder:   streamlit run app_text.py
# This folder needs:    app_text.py, best_text.pt, text_info.json
#                       (both saved by notebook 05, Step 14)
#
# Change only the parts marked "CHANGE".

import json
import re

import pandas as pd
import streamlit as st
import torch
from torch import nn

MODEL_FILE = "best_text.pt"
INFO_FILE = "text_info.json"  # model name, classes, vocabulary, max_len


# ---------------------------------------------------------------
# 1. Read text exactly as training did (copied from src/text_utils.py)
# ---------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"^rt\b", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    return " ".join(text.split())


TOKEN_PATTERN = re.compile(r"<url>|<user>|#?\w+(?:'\w+)?|[^\w\s]")


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def encode(text, word_to_idx, max_len):
    ids = [word_to_idx.get(t, 1) for t in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


# ---------------------------------------------------------------
# 2. The models (copied from src/text_utils.py)
# ---------------------------------------------------------------
class BagOfWords(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dim=64):
        super().__init__()
        self.bag = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean",
                                   padding_idx=0)
        self.out = nn.Linear(embed_dim, n_classes)

    def forward(self, ids):
        return self.out(torch.relu(self.bag(ids)))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dim=64, filters=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, filters, kernel_size=3,
                              padding=1)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(filters, n_classes)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)
        x = torch.relu(self.conv(x)).max(dim=2).values
        return self.out(self.drop(x))


@st.cache_resource
def load_model():
    with open(INFO_FILE) as f:
        info = json.load(f)
    n_words, n_classes = len(info["vocab"]), len(info["classes"])
    if info["model"] == "bag_of_words":
        model = BagOfWords(n_words, n_classes, info["embed_dim"])
    else:
        model = TextCNN(n_words, n_classes, info["embed_dim"])
    state = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    word_to_idx = {w: i for i, w in enumerate(info["vocab"])}
    return model, info, word_to_idx


# ---------------------------------------------------------------
# 3. Predict one text
# ---------------------------------------------------------------
def predict(model, info, word_to_idx, text):
    ids = encode(clean_text(text), word_to_idx, info["max_len"])
    x = torch.tensor([ids])  # shape (1, max_len)
    with torch.no_grad():
        logits = model(x)
    return torch.softmax(logits, dim=1)[0]


# ---------------------------------------------------------------
# 4. The page
# ---------------------------------------------------------------
# CHANGE: your project's title and instructions.
st.title("My Tweet Classifier")
st.write("Paste a tweet (up to 280 characters), then click Predict.")

model, info, word_to_idx = load_model()
classes = info["classes"]

text = st.text_area("Tweet", max_chars=280, height=120)

if st.button("Predict") and text.strip():
    probs = predict(model, info, word_to_idx, text)
    best = probs.argmax().item()
    st.subheader(f"Prediction: {classes[best]}")
    st.write(f"Confidence: {probs[best].item():.1%}")

    chart = pd.DataFrame({"class": classes,
                          "probability": probs.tolist()})
    st.bar_chart(chart, x="class", y="probability")

    tokens = tokenize(clean_text(text))
    unknown = [t for t in tokens if t not in word_to_idx]
    if unknown:
        st.caption(f"Words the model has never seen: {', '.join(unknown)}")

st.caption("A class project model. Do not rely on it for real decisions.")
