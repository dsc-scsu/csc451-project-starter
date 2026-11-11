"""Helpers for loading an image dataset from files.

notebooks/01_load_image_data.ipynb uses every function in this file,
in the order they appear here:

    0. unzip_once           unzip data.zip (and show_folders to look)
    1. list_images          find every image file
    2. labels_from_...      get one label per image
    3. find_bad_images      files that will not open
    4. find_duplicates      the same file saved twice
    5. make_table           one row per image: path, label, group
    6. make_splits          add a "split" column: train / val / test
    7. ImageDataset         turn the table into (tensor, label) pairs
"""

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------
# 0. Unzip the data and look at the folders
# ---------------------------------------------------------------------
def unzip_once(zip_path, dest):
    """Unzip `zip_path` into `dest`, unless that was already done.

    In Colab, unzip to /content (the session's own disk): reading many
    small files from it is much faster than reading them from Drive.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print("Already unzipped in", dest)
        return
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Check PROJECT_FOLDER and DATA_ZIP_NAME.")
    print("Unzipping", zip_path.name, "...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print("Done:", dest)



def _visible(folder):
    """Entries of a folder, without hidden files and Mac zip junk."""
    return sorted(p for p in Path(folder).iterdir()
                  if not p.name.startswith(".") and p.name != "__MACOSX")


def show_folders(root, depth=2, max_items=8):
    """Print the folders under `root`, with how many files each holds."""
    def walk(folder, level):
        items = _visible(folder)
        dirs = [p for p in items if p.is_dir()]
        n_files = len(items) - len(dirs)
        print("    " * level + f"{folder.name}/  ({n_files} files)")
        if level + 1 < depth:
            for d in dirs[:max_items]:
                walk(d, level + 1)
            if len(dirs) > max_items:
                print("    " * (level + 1) + f"... {len(dirs) - max_items}"
                      " more folders")
    walk(Path(root), 0)


def find_image_root(folder):
    """Skip down through folders that only hold one other folder.

    Zip files often add one extra top folder, such as
    data/my_images/cat/... ; this returns data/my_images.
    """
    folder = Path(folder)
    while True:
        items = _visible(folder)
        if len(items) == 1 and items[0].is_dir():
            folder = items[0]
        else:
            return folder


# ---------------------------------------------------------------------
# 1. Find the image files
# ---------------------------------------------------------------------
def list_images(root):
    """Return every image file under `root`, in any subfolder, sorted.

    Hidden files such as .DS_Store, and files that are not images
    (by their extension), are skipped.
    """
    root = Path(root)
    paths = []
    for p in root.rglob("*"):
        if p.name.startswith("."):
            continue
        if "__MACOSX" in p.parts:  # junk folder made by Mac zip
            continue
        if p.suffix.lower() in IMAGE_TYPES:
            paths.append(p)
    return sorted(paths)


# ---------------------------------------------------------------------
# 2. Get one label per image (pick ONE of these three)
# ---------------------------------------------------------------------
def labels_from_folders(paths):
    """Label = the name of the folder that holds the image.

    data/cat/001.jpg  ->  "cat"
    """
    return [p.parent.name for p in paths]


def labels_from_filenames(paths, pattern=r"(.+)_\d+$"):
    """Label = the part of the file name inside the ( ) of `pattern`.

    The default pattern keeps everything before the last "_" + digits:
    basset_hound_71.jpg  ->  "basset_hound"

    Files whose name does not match get the label None.
    """
    labels = []
    for p in paths:
        m = re.match(pattern, p.stem)
        if m:
            labels.append(m.group(1))
        else:
            labels.append(None)
    return labels


def labels_from_csv(csv_file, root, file_col="filename",
                    label_col="label"):
    """Read file names and labels from a CSV file.

    The CSV needs one column with the file name (relative to `root`)
    and one column with the label. Returns (paths, labels).
    Rows whose file does not exist are skipped and printed.
    """
    df = pd.read_csv(csv_file)
    paths, labels = [], []
    missing = 0
    for name, label in zip(df[file_col], df[label_col]):
        p = Path(root) / str(name)
        if p.exists():
            paths.append(p)
            labels.append(str(label))
        else:
            missing += 1
            if missing <= 5:
                print("Not found:", p)
    if missing:
        print(missing, "rows in the CSV point to files that do not exist")
    return paths, labels


# ---------------------------------------------------------------------
# 3. Files that will not open
# ---------------------------------------------------------------------
def find_bad_images(paths):
    """Return the files that PIL cannot open as an image."""
    bad = []
    for p in paths:
        try:
            with Image.open(p) as img:
                img.convert("RGB")
        except Exception:
            bad.append(p)
    return bad


# ---------------------------------------------------------------------
# 4. The same file saved twice
# ---------------------------------------------------------------------
def file_hash(path):
    """A short fingerprint of the file's bytes. Same bytes, same hash."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_duplicates(paths):
    """Return the paths that are exact copies of an earlier path.

    If a copy lands in train and the original in test, the test score
    is too high (data leakage). Drop these before splitting.
    """
    seen = set()
    copies = []
    for p in paths:
        h = file_hash(p)
        if h in seen:
            copies.append(p)
        else:
            seen.add(h)
    return copies


# ---------------------------------------------------------------------
# 5. One table with everything
# ---------------------------------------------------------------------
def make_table(paths, labels, root, groups=None):
    """Build a table with one row per image.

    path   the file path relative to `root`, so the table still works
           after you unzip the data again in a new Colab session
    label  the class name
    group  (optional) who or what is in the photo; photos with the
           same group always go to the same split
    """
    root = Path(root)
    table = pd.DataFrame({
        "path": [str(Path(p).relative_to(root)) for p in paths],
        "label": labels,
    })
    if groups is not None:
        table["group"] = groups
    return table


def group_from_filename(paths, sep="_"):
    """Group = the part of the file name before the first `sep`.

    rex_03.jpg  ->  "rex"
    """
    return [p.stem.split(sep)[0] for p in paths]


# ---------------------------------------------------------------------
# 6. Train / validation / test split
# ---------------------------------------------------------------------
def make_splits(table, val_size=0.15, test_size=0.15, by_group=False,
                seed=0):
    """Add a "split" column with the values train, val and test.

    by_group=False: a random split that keeps the same share of each
                    class in all three sets (stratified).
    by_group=True:  every photo of one group goes to the same split.
                    Use this when one subject appears in several photos.
    """
    table = table.copy()
    hold = val_size + test_size  # share of rows that leave train

    if by_group:
        if "group" not in table.columns:
            raise ValueError("by_group=True needs a 'group' column")
        # Whole groups move together, and each split keeps roughly the
        # same share of each class. The sizes are close to, not exactly,
        # val_size and test_size.
        k1 = max(2, round(1 / hold))
        g1 = StratifiedGroupKFold(n_splits=k1, shuffle=True,
                                  random_state=seed)
        train_idx, rest_idx = next(g1.split(table, table["label"],
                                            table["group"]))
        rest = table.iloc[rest_idx]
        k2 = max(2, round(hold / test_size))
        g2 = StratifiedGroupKFold(n_splits=k2, shuffle=True,
                                  random_state=seed)
        val_idx, test_idx = next(g2.split(rest, rest["label"],
                                          rest["group"]))
        train_rows = table.index[train_idx]
        val_rows = rest.index[val_idx]
        test_rows = rest.index[test_idx]
    else:
        counts = table["label"].value_counts()
        if counts.min() < 3:
            raise ValueError(
                "Every class needs at least 3 images to split. "
                f"Too few: {list(counts[counts < 3].index)}")
        train_rows, rest_rows = train_test_split(
            table.index, test_size=hold, stratify=table["label"],
            random_state=seed)
        rest = table.loc[rest_rows]
        val_rows, test_rows = train_test_split(
            rest.index, test_size=test_size / hold,
            stratify=rest["label"], random_state=seed)

    table.loc[train_rows, "split"] = "train"
    table.loc[val_rows, "split"] = "val"
    table.loc[test_rows, "split"] = "test"
    return table


# ---------------------------------------------------------------------
# 7. Dataset: one (tensor, label) pair per row of the table
# ---------------------------------------------------------------------
class ImageDataset(Dataset):
    """Images listed in a table, with their labels as class numbers.

    table      rows of one split (columns "path" and "label")
    root       the folder the paths are relative to
    classes    the list of class names; its order sets the numbers
    transform  the torchvision transforms for this split
    """

    def __init__(self, table, root, classes, transform):
        self.root = Path(root)
        self.paths = list(table["path"])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [class_to_idx[c] for c in table["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
