# CSC 451 - Streamlit skeleton for an image classifier
#
# Run on your laptop, from the app/ folder:   streamlit run app.py
# This folder needs:    app.py, best.pt, classes.json, model_info.json
#                       (all three saved by notebook 02, Step 12)
#
# Change only the parts marked "CHANGE".

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------
MODEL_FILE = "best.pt"         # saved with torch.save(model.state_dict())
CLASSES_FILE = "classes.json"  # the class names, in index order
INFO_FILE = "model_info.json"  # which model, and its image size

# Read which model notebook 02 trained. Without the file: ResNet-18.
info = {"model": "resnet18", "image_size": 224}
if Path(INFO_FILE).exists():
    with open(INFO_FILE) as f:
        info = json.load(f)
IMAGE_SIZE = info["image_size"]

# The same eval transforms as notebook 02 (src/train_utils.py).
eval_tfms = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 256 / 224)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------
# 2. Load the model once
# ---------------------------------------------------------------
def build_model(name, n_classes):
    """The same three models as src/train_utils.py."""
    if name == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    side = IMAGE_SIZE // 8
    if name == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * side * side, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )
    raise ValueError(f"Unknown model in {INFO_FILE}: {name}")


@st.cache_resource
def load_model():
    with open(CLASSES_FILE) as f:
        classes = json.load(f)

    # CHANGE only if you changed a model's layers in train_utils.py.
    model = build_model(info["model"], len(classes))

    state = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, classes


# ---------------------------------------------------------------
# 3. Predict one image
# ---------------------------------------------------------------
def predict(model, img):
    x = eval_tfms(img).unsqueeze(0)  # shape (1, 3, size, size)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    return probs


# ---------------------------------------------------------------
# 4. The page
# ---------------------------------------------------------------
# CHANGE: your project's title and instructions.
st.title("My Image Classifier")
st.write("Upload a photo. The model predicts its class.")

model, classes = load_model()

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your image", width=250)

    probs = predict(model, img)
    best = probs.argmax().item()

    st.subheader(f"Prediction: {classes[best]}")
    st.write(f"Confidence: {probs[best].item():.1%}")

    chart = pd.DataFrame({"class": classes,
                          "probability": probs.tolist()})
    st.bar_chart(chart, x="class", y="probability")

st.caption("A class project model. Do not rely on it for real decisions.")
