"""Helpers for training and evaluating a model.

notebooks/02_train_image_model.ipynb and 03_table_data.ipynb use these.
The code follows the Week 9 and Week 12 slides.
"""

import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    """Make a run repeatable: same seed, same random numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Transforms (Week 12): randomness in training only
# ---------------------------------------------------------------------
def make_transforms(image_size=224, flip=True):
    """Return (train_tfms, eval_tfms).

    flip=False when a mirror image changes the meaning: letters,
    chest X-rays, anything with a left and a right side.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_steps = [transforms.RandomResizedCrop(image_size,
                                                scale=(0.6, 1.0))]
    if flip:
        train_steps.append(transforms.RandomHorizontalFlip())
    train_steps += [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ]
    train_tfms = transforms.Compose(train_steps)

    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tfms, eval_tfms


# ---------------------------------------------------------------------
# Model (Week 12): pretrained backbone, new head
# ---------------------------------------------------------------------
def make_resnet18(n_classes, pretrained=True):
    """ResNet-18 with a new head for `n_classes` classes."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def make_small_cnn(n_classes, image_size=64):
    """The Week 11 CNN: three conv blocks, then two Linear layers.

    Each block is Conv2d -> ReLU -> MaxPool2d(2), so the height and width
    are halved three times: 64 -> 32 -> 16 -> 8. Images are RGB, so the
    first Conv2d has 3 input channels.
    """
    side = image_size // 8  # height and width after three poolings
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),    # conv block 1
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),   # conv block 2
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),  # conv block 3
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),                      # classifier
        nn.Linear(128 * side * side, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )


def make_mlp(n_classes, image_size=64):
    """An MLP baseline: flatten the image, then two Linear layers.

    It sees the image as one long row of numbers (Week 11, Part 2), so
    it shows how much the convolution layers add.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * image_size * image_size, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )


def make_model(name, n_classes, image_size, pretrained=True):
    """Build the model named "resnet18", "small_cnn" or "mlp"."""
    if name == "resnet18":
        return make_resnet18(n_classes, pretrained)
    if name == "small_cnn":
        return make_small_cnn(n_classes, image_size)
    if name == "mlp":
        return make_mlp(n_classes, image_size)
    raise ValueError('MODEL must be "resnet18", "small_cnn" or "mlp"')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model):
    """Freeze every layer except the head (model.fc)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------
# One epoch of training, and evaluation
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, opt, device,
                    frozen_backbone=False):
    """Train for one pass over `loader`. Returns (mean loss, accuracy)."""
    if frozen_backbone:
        model.eval()        # keep BatchNorm averages fixed (Week 12)
        model.fc.train()
    else:
        model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def evaluate(model, loader, loss_fn, device):
    """Return (mean loss, accuracy) on `loader`, without training."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def predict_all(model, loader, device):
    """Return (true labels, predicted labels, probabilities) as arrays."""
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu()
            ys.append(y)
            preds.append(p.argmax(dim=1))
            probs.append(p)
    return (torch.cat(ys).numpy(), torch.cat(preds).numpy(),
            torch.cat(probs).numpy())


def fit(model, train_dl, val_dl, loss_fn, opt, device, n_epochs,
        checkpoint, patience=5, frozen_backbone=False, best_acc=-1.0):
    """Train with checkpointing and early stopping (Week 12).

    Saves the weights with the best validation accuracy to `checkpoint`
    and reloads them at the end. Returns the history as a list of dicts.

    best_acc: the best validation accuracy of an earlier phase. Phase 2
    then only overwrites the checkpoint if it does better than phase 1.
    """
    bad_epochs = 0
    history = []
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, loss_fn, opt,
                                          device, frozen_backbone)
        va_loss, va_acc = evaluate(model, val_dl, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc})
        mark = ""
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(model.state_dict(), checkpoint)
            mark = "  <- best so far, saved"
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d}  train loss {tr_loss:.3f} "
              f"acc {tr_acc:.3f}  |  val loss {va_loss:.3f} "
              f"acc {va_acc:.3f}{mark}")
        if bad_epochs >= patience:
            print(f"No improvement for {patience} epochs: stopping.")
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return history


# ---------------------------------------------------------------------
# Experiment log (Week 9: change one thing per run, write it down)
# ---------------------------------------------------------------------
def log_run(log_file, **values):
    """Add one row to a CSV log of experiments.

    log_run("experiments.csv", change="lr 1e-3 -> 3e-4", val_acc=0.91)
    The first call creates the file with these column names.
    """
    log_file = Path(log_file)
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    row.update(values)
    write_header = not log_file.exists()
    if not write_header:
        with open(log_file) as f:
            header = f.readline().strip().split(",")
        row = {k: row.get(k, "") for k in header}
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
