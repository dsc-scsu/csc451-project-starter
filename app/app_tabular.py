# CSC 451 - Streamlit skeleton for a model on a table of numbers
#
# Run on your laptop, from the app/ folder:   streamlit run app_tabular.py
# This folder needs:    app_tabular.py, best.pt, info.json
#
# Change only the parts marked "CHANGE".

import json

import streamlit as st
import torch
from torch import nn

MODEL_FILE = "best.pt"   # saved with torch.save(model.state_dict())
INFO_FILE = "info.json"  # features, classes, mean and std (see notebook)


# ---------------------------------------------------------------
# 1. Load the model once
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(INFO_FILE) as f:
        info = json.load(f)
    n_in = len(info["features"])
    n_out = len(info["classes"])

    # CHANGE: the same layers you trained, in the same order.
    model = nn.Sequential(
        nn.Linear(n_in, 16),
        nn.ReLU(),
        nn.Linear(16, n_out),
    )

    state = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, info


# ---------------------------------------------------------------
# 2. Predict one row
# ---------------------------------------------------------------
def predict(model, info, values):
    x = torch.tensor([values], dtype=torch.float32)  # shape (1, n)
    mean = torch.tensor(info["mean"])
    std = torch.tensor(info["std"])
    x = (x - mean) / std  # the same scaling you used in training
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    return probs


# ---------------------------------------------------------------
# 3. The page
# ---------------------------------------------------------------
# CHANGE: your project's title and instructions.
st.title("My Tabular Classifier")
st.write("Enter the measurements, then click Predict.")

model, info = load_model()

# One number box per feature, filled in with the training average.
values = []
for name, avg in zip(info["features"], info["mean"]):
    v = st.number_input(name, value=round(avg, 3), format="%.3f")
    values.append(v)

if st.button("Predict"):
    probs = predict(model, info, values)
    best = probs.argmax().item()
    st.subheader(f"Prediction: {info['classes'][best]}")
    st.write(f"Confidence: {probs[best].item():.1%}")

st.caption("A class project model. Do not rely on it for real decisions.")
