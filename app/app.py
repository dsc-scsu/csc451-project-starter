# CSC 451 - Streamlit skeleton for an image classifier
#
# Run on your laptop, from the app/ folder:   streamlit run app.py
# This folder needs:    app.py, best.pt, classes.json
#
# Change only the parts marked "CHANGE".

import json

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
CLASSES_FILE = "classes.json"  # saved with json.dump(train_ds.classes)

# CHANGE: copy the eval transforms from your notebook.
# They must match the ones you used for validation.
eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------
# 2. Load the model once
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(CLASSES_FILE) as f:
        classes = json.load(f)

    # CHANGE: build the same architecture you trained.
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    state = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, classes


# ---------------------------------------------------------------
# 3. Predict one image
# ---------------------------------------------------------------
def predict(model, img):
    x = eval_tfms(img).unsqueeze(0)  # shape (1, 3, 224, 224)
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
