# CSC 451 Final Project Starter

This repo gives your team a working pipeline for the final project:

1. **Load** your own images (or a table, or short texts such as
   tweets) from files, clean them, and split them into train,
   validation and test sets.
2. **Train** a pretrained ResNet-18, a small CNN from scratch, or an
   MLP baseline, with learning curves and an experiment log.
3. **Evaluate** it: a confusion matrix and the worst mistakes,
   precision-recall and ROC curves, a decision threshold chosen from the
   cost of each mistake, and a chart comparing your experiments.
4. **Serve** it in a Streamlit app.

You change settings, not code: every place you must edit is marked
**CHANGE**. Read this whole page once before you start; it takes
about 15 minutes and saves hours.

---

## Contents

- [What is in this repo](#what-is-in-this-repo)
- [The workflow at a glance](#the-workflow-at-a-glance)
- [Step 0 · Make your team's copy of the repo](#step-0--make-your-teams-copy-of-the-repo)
- [Step 1 · Organize your images](#step-1--organize-your-images)
- [Step 2 · Zip the data and put it on Google Drive](#step-2--zip-the-data-and-put-it-on-google-drive)
- [Step 3 · Run notebook 01: load and split](#step-3--run-notebook-01-load-and-split)
- [Step 4 · Run notebook 02: train and evaluate](#step-4--run-notebook-02-train-and-evaluate)
- [Step 4b · Run notebook 04: evaluate the final model](#step-4b--run-notebook-04-evaluate-the-final-model)
- [Step 5 · Table data instead of images: notebook 03](#step-5--table-data-instead-of-images-notebook-03)
- [Step 5b · Tweets and other short texts: notebook 05](#step-5b--tweets-and-other-short-texts-notebook-05)
- [Step 6 · Run the Streamlit app](#step-6--run-the-streamlit-app)
- [Step 7 · What to commit to GitHub](#step-7--what-to-commit-to-github)
- [Where each project requirement comes from](#where-each-project-requirement-comes-from)
- [Troubleshooting](#troubleshooting)

---

## What is in this repo

```
csc451-project-starter/
├── README.md                       this guide
├── requirements.txt                packages, if you run notebooks on your laptop
├── notebooks/
│   ├── 01_load_image_data.ipynb    find, clean and split your images
│   ├── 02_train_image_model.ipynb  train (ResNet-18, small CNN or MLP),
│   │                               log, test, save the model
│   ├── 03_table_data.ipynb         the same for a CSV table
│   ├── 04_evaluate_model.ipynb     curves, threshold, experiment chart
│   └── 05_text_data.ipynb          the same for tweets and short texts
├── src/
│   ├── data_utils.py               the code that loads images from files
│   ├── train_utils.py              training, evaluation, experiment log
│   └── text_utils.py               cleaning and encoding short texts
└── app/
    ├── app.py                      Streamlit app for an image model
    ├── app_tabular.py              Streamlit app for a table model
    ├── app_text.py                 Streamlit app for a text model
    └── requirements.txt            packages the app needs
```

The notebooks call the functions in `src/`. You do not need to edit
`src/`, but read `src/data_utils.py`: every function is short and
explained, and you should be able to say in your presentation what
each step of your data pipeline does.

---

## The workflow at a glance

| Step | Where it runs | What you get |
|---|---|---|
| 0. Make the team repo | GitHub | Your own copy of this repo |
| 1–2. Organize, zip, upload the data | Your laptop, Google Drive | `data.zip` in Drive |
| 3. Notebook 01 | Colab | `splits.csv`, `classes.json` in Drive |
| 4. Notebook 02 (many runs) | Colab, GPU | `best.pt`, learning curves, `experiments.csv` |
| 4b. Notebook 04 (final model) | Colab | Curves, a chosen threshold, an experiment chart |
| 6. App | Your laptop | A web page that runs your model |

Everything the notebooks produce is saved in **your Google Drive**,
in `csc451_project/outputs/`, so it survives when Colab disconnects.

---

## Step 0 · Make your team's copy of the repo

One teammate does this:

1. Open the course starter repo:
   [github.com/dsc-scsu/csc451-project-starter](https://github.com/dsc-scsu/csc451-project-starter).
2. Click **Use this template → Create a new repository**.
3. Owner: your GitHub account. Name: anything, for example
   `csc451-project-team3`.
4. Choose **Public**. Colab can download a public repo without
   signing in. (A private repo needs extra steps; see
   [Troubleshooting](#troubleshooting).)
5. Click **Create repository**.
6. Add your teammates: **Settings → Collaborators → Add people**.

Then copy the repo's address: the green **Code** button shows it,
ending in `.git`. You paste it into the first cell of each notebook
(`REPO_URL`) in Step 3. It starts out pointing at the course starter
repo; change it to your team's copy.

---

## Step 1 · Organize your images

The notebooks need to know two things about every image: **where the
file is** and **what its label is**. Labels can come from one of
three places. Pick the one that matches your data, and set
`LABEL_SOURCE` to it in notebook 01.

### Option A: labels in folder names (`LABEL_SOURCE = "folders"`)

One folder per class. The folder name is the label.

```
my_images/
├── cat/
│   ├── 001.jpg
│   └── 002.jpg
└── dog/
    ├── 001.jpg
    └── 002.jpg
```

This is the simplest option. If you collect your own photos, use it.

### Option B: labels in file names (`LABEL_SOURCE = "filenames"`)

All images in one folder; the label is part of the name.

```
pets/
├── basset_hound_1.jpg
├── basset_hound_2.jpg
└── pug_1.jpg
```

By default the label is everything **before the last underscore and
number**: `basset_hound_1.jpg` → `basset_hound`. If your names look
different, change `FILENAME_PATTERN` in notebook 01. It is a
*regular expression* (a text pattern); the part inside `( )` becomes
the label. Ask in office hours if you are not sure how to write it.

### Option C: labels in a CSV file (`LABEL_SOURCE = "csv"`)

A CSV file lists each image and its label. Many Kaggle datasets work
this way.

```
labels.csv            images/
filename,label        ├── a81f.jpg
a81f.jpg,cat          ├── b223.jpg
b223.jpg,dog          └── ...
```

Set `CSV_NAME`, `CSV_FILE_COLUMN` and `CSV_LABEL_COLUMN` to your
file's name and column names. If the file names in the CSV are
relative to a subfolder (here, `images/`), set
`CSV_IMAGE_FOLDER = "images"`.

### Is the same subject in several photos? Use groups

If you took 10 photos of each of your friends' 6 dogs, a random split
puts some photos of each dog in training and some in the test set.
The model then learns to recognize *those dogs*, and the test score is
higher than it would be on new dogs. This is **data leakage**.

To prevent it, give each photo a **group**: who or what is in it.
Every photo of one group then goes to the same split.

- In file names: start each name with the group and an underscore,
  such as `rex_01.jpg`, `rex_02.jpg`, `bella_01.jpg`, and set
  `GROUP_SOURCE = "filename"`.
- In a CSV: add a column such as `subject` and set
  `GROUP_SOURCE = "csv"` and `CSV_GROUP_COLUMN = "subject"`.

You need several groups per class (at least 5 is a good minimum),
or some classes end up with no test images.

### Before you zip: a checklist

- [ ] Each image is its own file: `.jpg`, `.jpeg`, `.png`, `.bmp` or
      `.webp`. iPhone photos are often `.heic`: export them as JPEG.
- [ ] Class names have no typos (`cat` and `Cat` are two classes).
- [ ] Each class has at least 20 images; more is better.
- [ ] No other files are mixed in (the notebook skips non-images,
      but a stray folder can become a fake class).

You do **not** need to resize images, remove duplicates, or find
broken files by hand: notebook 01 does that.

---

## Step 2 · Zip the data and put it on Google Drive

1. Zip the top folder of your data (for example `my_images`):
   - **Mac:** right-click the folder → **Compress**.
   - **Windows:** right-click the folder → **Send to → Compressed
     (zipped) folder**.
2. Rename the zip to `data.zip`.
3. In Google Drive, in **My Drive**, create a folder called
   `csc451_project` and upload `data.zip` into it.
4. Share that folder with your teammates (right-click → **Share**).
   Each teammate then right-clicks the shared folder → **Organize →
   Add shortcut → My Drive**, so the path is the same for everyone.

Why a zip? Colab reads thousands of small files from Drive very
slowly. The notebooks copy the one zip file and unzip it onto
Colab's own disk, which takes seconds.

---

## Step 3 · Run notebook 01: load and split

**Open it in Colab.** Go to [colab.research.google.com](https://colab.research.google.com),
choose **File → Open notebook → GitHub**, paste your repo's address,
and click `notebooks/01_load_image_data.ipynb`.

**Run it:**

1. In the first cell, set `REPO_URL` to your repo's address.
2. In the **Step 2 · Settings** cell, set `LABEL_SOURCE` and, if
   needed, the other options. The notebook explains each one.
3. Choose **Runtime → Run all**. When asked, allow access to your
   Google Drive.
4. Read the output of every step, from top to bottom:

| Step | Check that… |
|---|---|
| 3 | The folder listing shows your classes, not one extra folder level |
| 4 | The number of images found is about what you expect |
| 5 | The class names are right |
| 6–7 | Broken and duplicate files are listed (these are removed) |
| 8 | No class is tiny; note the largest class's share: that is your baseline |
| 10 | Every class has images in train, val and test |
| 11 | The images match their labels, and nothing odd (a watermark, a frame, a background) differs between classes |
| 12 | The batch has shape `(16, 3, 224, 224)` and the augmented images still show the subject |

**Run notebook 01 only once.** It saves the split to
`outputs/splits.csv`, and every later run of notebook 02 uses that
same split. If you change your data, set `OVERWRITE_SPLITS = True`,
run it again, then set it back to `False`.

---

## Step 4 · Run notebook 02: train and evaluate

Open `notebooks/02_train_image_model.ipynb` the same way. First
choose **Runtime → Change runtime type → T4 GPU**; training on the
CPU is many times slower.

**Choose a model** with `MODEL` in the settings cell:

| `MODEL` | What it is | `IMAGE_SIZE` |
|---|---|---|
| `"resnet18"` | Pretrained ResNet-18 (Week 12). Trains in two phases: first only the new head, then the whole network at a smaller learning rate. The best choice for photos | 224 |
| `"small_cnn"` | The three-block CNN from Week 11, trained from scratch. Good for images unlike everyday photos: drawings, spectrograms, sensor plots | 64 (or 96) |
| `"mlp"` | The image flattened into one row, then two Linear layers. This is your **MLP baseline** | 64 |

`small_cnn` and `mlp` train every layer in one phase, using
`LR_SCRATCH` and `EPOCHS_SCRATCH`. Keep their `IMAGE_SIZE` small: at
224, the first Linear layer would have tens of millions of weights.

**The first run:** set `REPO_URL`, keep the default settings, and
choose **Runtime → Run all**.

Check these outputs:

- **Step 5, baseline:** the accuracy of always answering the largest
  class. Your model must beat it clearly.
- **Step 7:** the number of parameters in your model (Week 11).
- **Step 6, overfit one batch:** the loss should end near 0. If not,
  something is wrong with the data or the labels; fix that first.
- **Steps 7–8, training:** one line per epoch. The best epoch's
  weights are saved to `outputs/best.pt` on Drive.
- **Step 9, learning curves:** saved as `outputs/learning_curves.png`.

**Your two baselines.** The final project asks for two: the
largest-class baseline (Step 5 prints it every run) and an **MLP**.
Do one run with `MODEL = "mlp"` and `IMAGE_SIZE = 64`, and keep its
row in the log, so you can show how much your CNN adds.

**Every run after that: change one thing.** Change one setting in
Step 2, write what you changed in `RUN_NOTE` (for example
`"LR_FINE 1e-4 -> 3e-5"`), and run all cells. Each run adds a row to
`outputs/experiments.csv`. Compare `best_val_acc` across rows to decide
what helped. Things worth trying, roughly in this order:

1. `FLIP = False` if mirrored images change the meaning.
2. `LR_HEAD` and `LR_FINE`, each up or down by about 3×.
3. More epochs (`EPOCHS_FINE`), if validation accuracy was still
   rising when training stopped.
4. `FINE_TUNE = False`, if your dataset is very small.
5. More or cleaner data. This usually helps more than any setting.

**Test once, at the end.** When you have chosen your final settings,
set `EVALUATE_TEST = True` and run all cells one last time. Step 11
shows the test accuracy, the accuracy for each class, a confusion
matrix (`outputs/confusion_matrix.png`), and the most confidently
wrong test images. Use these in your error analysis. Do not go back
and tune settings after you look at the test score: then it no longer
measures how the model does on new data.

**At the end,** Step 12 downloads `best.pt`, `classes.json` and
`model_info.json`. You need all three for the app.

---

## Step 4b · Run notebook 04: evaluate the final model

Run `notebooks/04_evaluate_model.ipynb` after notebook 02 has trained
your final model. It loads `outputs/best.pt` and covers the Week 10
evaluation topics.

In its settings cell, set:

- `POSITIVE_CLASS`: the class that means "yes", the one you most need
  to find (for example `"pneumonia"` or `"defective"`).
- `COST_MISSED` and `COST_FALSE_ALARM`: how bad each kind of mistake
  is. Only their ratio matters: 5 and 1 mean a missed case is 5 times
  worse than a false alarm. Explain your choice in the presentation.

It then shows:

| Step | Output, saved in `outputs/` |
|---|---|
| 4 | Precision-recall and ROC curves with their AUC: `pr_roc_curves.png` |
| 5 | Total cost at every threshold, and the cheapest threshold: `cost_vs_threshold.png` |
| 6 | Confusion matrices at 0.5 and at the chosen threshold |
| 7 | With `EVALUATE_TEST = True`: test results at the chosen threshold, `test_confusion_matrix.png` |
| 8 | Every run in `experiments.csv` as one bar chart: `experiments_chart.png` |

The threshold is chosen on the **validation** set; the test set only
reports how well that choice works. With more than two classes, the
notebook treats `POSITIVE_CLASS` against all the others. The last cell
shows how to use your threshold in the app.

---

## Step 5 · Table data instead of images: notebook 03

If your data is a CSV table (one row per example), use
`notebooks/03_table_data.ipynb` instead of notebooks 01 and 02:

1. Upload your CSV to `csc451_project` in Drive.
2. In the settings cell, set `CSV_NAME`, `TARGET_COLUMN` (the column
   to predict) and `DROP_COLUMNS` (IDs, names, dates, and anything only
   known *after* the answer, which would be leakage).
3. Run all cells.

It uses the number columns as features and lists any text columns it
leaves out. It fills missing values with the median and scales every
column, using **training rows only**. At the end it saves `info.json`
for `app/app_tabular.py`.

---

## Step 5b · Tweets and other short texts: notebook 05

If each example is a short text with a label (a tweet, a review, a
headline), use `notebooks/05_text_data.ipynb` instead of notebooks
01 and 02.

**Your data:** one CSV file in `csc451_project` on Drive, with one
column for the text and one for the label, for example:

```
text,label
"Loved the new update!! 🎉 https://t.co/x1",positive
"@brand_help my order is late again",negative
```

In the settings cell, set `CSV_NAME`, `TEXT_COLUMN` and `LABEL_COLUMN`,
then run all cells. The notebook:

1. **Cleans** each text: lower case; links become `<url>` and user
   names become `<user>` (every link and name is different, so the
   model cannot learn from them otherwise); `RT` is removed. Hashtags,
   emoji and punctuation stay.
2. **Removes duplicates**, which are common in tweets (retweets, bots).
   A copy in training and another in the test set is data leakage.
   Texts that appear with two different labels are removed too.
3. **Splits** the texts (stratified) and saves the split.
4. Builds the **vocabulary** (the words the model knows) from the
   training texts only, and reports how many validation words are
   unknown.
5. **Trains** one of two models, set with `MODEL`:
   - `"bag_of_words"`: averages the word vectors and ignores word
     order. This is your **baseline**.
   - `"text_cnn"`: a 1D CNN whose filters slide over 3 neighboring
     words and detect short phrases such as "not good". It uses the
     same ideas as the Week 11 CNN, on a sequence of words.
6. Logs each run in `outputs/experiments_text.csv`, and (with
   `EVALUATE_TEST = True`) shows the test confusion matrix and the
   most confidently wrong texts for your error analysis.
7. Saves `best_text.pt` and `text_info.json` for `app/app_text.py`.

Run it once with each model and compare them in the log. Notebook 04
is written for image models; for text, use Step 13 of notebook 05.

**Where to get labeled tweets:** the X (Twitter) API has no free way
to read tweets; it charges for every tweet it returns. Most teams
start from a public labeled dataset (Kaggle has many) and clean it
here. Say in your README where
the data came from and how the labels were made.

---

## Step 6 · Run the Streamlit app

The app runs on your laptop, not in Colab.

1. Download your repo to your laptop:
   `git clone <your repo address>`, or **Code → Download ZIP** on
   GitHub.
2. Put `best.pt`, `classes.json` and `model_info.json` (from Step 4)
   into the `app/` folder, all three from the **same run**. For a table
   model, put `best.pt` and `info.json` there.
3. In a terminal, install the packages once:
   ```
   cd app
   pip install -r requirements.txt
   ```
4. Start the app from the `app/` folder:
   ```
   streamlit run app.py
   ```
   (For a table model: `streamlit run app_tabular.py`; for a text
   model: `streamlit run app_text.py`, with `best_text.pt` and
   `text_info.json` in `app/`. On Windows, if
   `streamlit` is not found, use `python -m streamlit run app.py`.)
5. Your browser opens `http://localhost:8501`. Ctrl+C in the terminal
   stops the app.

**Edit the parts marked CHANGE** in `app.py`: the title and the
instructions. The app reads `model_info.json` to build the same model
and resize images to the same size as notebook 02 did. An app that
prepares images differently from validation predicts worse than your
test results (this is *train/serve skew*, Week 13). If you change a
model's layers in `src/train_utils.py`, change `build_model` in
`app.py` the same way.

---

## Step 7 · What to commit to GitHub

**Commit:**

- The notebooks, with their outputs, after your final run. In Colab,
  **File → Save a copy in GitHub** saves a notebook straight into your
  repo (pick your repo and the `notebooks/` path).
- `app/` with `best.pt`, `classes.json` and `model_info.json` (or
  `info.json` for a table model).
  ResNet-18's `best.pt` is about 45 MB; GitHub accepts files up to
  100 MB.
- A `results/` folder with copies of `experiments.csv` and the plots
  from notebooks 02 and 04 (the `.png` files in `outputs/` on Drive).
- This README, updated to describe **your** project: the data, how to
  run it, and your results.

**Do not commit:**

- Your data (`data.zip` or the images). It stays in Drive. Say in the
  README where it came from.
- Passwords or access tokens, ever.

To change files in `src/` or `app/`, edit them on GitHub (open the
file, click the pencil icon) or on your laptop and push. Edits made
inside Colab's file browser are lost when the session ends.

---

## Where each project requirement comes from

| Final project asks for | Where you get it |
|---|---|
| A dataset you built or cleaned | Steps 1–2; notebook 01 removes broken files and duplicates |
| A proper data split | Notebook 01, Step 10: `splits.csv` (plus groups, if needed) |
| Baselines (largest class and MLP) | Notebook 02: Step 5 every run, plus one run with `MODEL = "mlp"` |
| A small CNN or a pretrained CNN | Notebook 02: `MODEL = "small_cnn"` or `"resnet18"` |
| Text projects: a baseline and a CNN | Notebook 05: `MODEL = "bag_of_words"` and `"text_cnn"` |
| Learning curves | Notebook 02, Step 9: `learning_curves.png` |
| A log of experiments | Notebook 02, Step 10: `experiments.csv` |
| An honest analysis of errors | Notebook 02, Step 11: per-class accuracy, confusion matrix, worst mistakes |
| Evaluation beyond accuracy | Notebook 04: precision-recall and ROC curves, a threshold chosen from costs |
| Comparing experiments | Notebook 04, Step 8: `experiments_chart.png` |
| A Streamlit app | `app/` |
| Optional: Cloud Run deployment | Week 14 slides; the `app/` folder is what you deploy |

The starter gives you the machinery. The project grade comes from
your data, your experiments, and what you conclude from them.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `data.zip not found` | Check that `csc451_project` is directly in **My Drive** and that `PROJECT_FOLDER` and `DATA_ZIP_NAME` match exactly (capital letters count). For a shared folder, add a shortcut to My Drive (Step 2). |
| `No module named 'data_utils'` | Run the first cell (Setup) before any other cell. |
| `git clone` fails or asks for a password | The repo is private. Make it public (**Settings → General → Change visibility**), or ask your instructor for another way. |
| `0 image files found` | Look at the folder listing in Step 3. Check the file types: `.heic` and `.gif` are not read; convert them to `.jpg`. |
| A class called `my_images` (your top folder) | The images are not in one-folder-per-class layout. Look at the Step 3 listing and reorganize (Step 1). |
| `Every class needs at least 3 images to split` | A class is almost empty, often a typo in a label. Fix the label or remove the class. |
| A class has 0 images in val or test | Too few images or groups in that class. Collect more. |
| `CUDA out of memory` | Set `BATCH_SIZE = 16` (or 8) in notebook 02. |
| Colab disconnected during training | `best.pt` and the split are on Drive. Reconnect and run all cells again. |
| Training is very slow | Check that the runtime is **T4 GPU** (notebook 02 prints `Device: cuda`). |
| Accuracy stays at the baseline | Check Step 6 (overfit one batch). Then try a learning rate 3× smaller. Look at the Step 11 images in notebook 01: are the labels right? |
| Validation accuracy is near 100% from the first epoch | Suspect leakage: duplicates under different names, the same subject in train and val (use groups), or something in the image that gives the answer away (a watermark, a background). |
| Test accuracy is much lower than validation accuracy | You may have tuned on validation for many runs, or the test images differ (another source, another time). Describe it honestly in your report. |
| The app predicts the same class for everything | The app's transforms differ from training (`IMAGE_SIZE`, normalization), or `classes.json` is from another run. |
| `size mismatch` in the app | `best.pt`, `classes.json` and `model_info.json` come from different runs. Copy all three from the same run. |
| `CUDA out of memory` or a huge parameter count with `small_cnn` or `mlp` | `IMAGE_SIZE` is too large for these models. Use 64. |

Still stuck? Bring the notebook (with its outputs) to office hours.
