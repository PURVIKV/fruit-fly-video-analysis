# Real-Time Detailed Video Analysis of Fruit Flies

A computer-vision and machine-learning pipeline that processes top-down videos of a
male and a female fruit fly in a circular arena and, for every frame, estimates:

- **how many flies** are present (including two flies touching in one blob),
- the **sex** of each separated fly,
- each fly's **orientation** (heading, 0–360°),
- the **wing angles** of the male (left and right).

It reimplements, in simplified form, the approach of the Stanford CS229 project
[*Real-time detailed video analysis of fruit flies*](https://github.com/sgherbst/cs229-project)
by Steven Herbst (MIT license).

## Problem

Measuring courtship behaviour (who is where, which way each fly faces, how far the
male extends his wings) by hand is slow. The goal is to extract this per frame,
automatically, fast enough to keep up with the video.

## Method

All models start from the same detector (`src/detection.py`): grayscale → Gaussian
blur → inverted Otsu threshold → external contours with area ≥ 500 px that do not
touch the image border. Training and the video pipeline call the same functions,
so their features are computed identically.

| Task | Features (from the frame only) | Model |
|---|---|---|
| Fly count | contour area | `DecisionTreeClassifier(max_depth=2)` predicting 0 / 1 / 2 flies per contour; the frame count is the sum over contours |
| Sex | contour area + bounding-box aspect ratio | `StandardScaler` + `LogisticRegression` |
| Orientation | body axis from image moments; 240 px patch rotated so the axis is vertical; HOG | `PCA(20)` + `LogisticRegression` decides whether the head is at the other end (+180°) |
| Wing angle (male) | 288 px patch rotated upright, split into right and mirrored left halves; HOG | `PCA(80)` + `LinearRegression` (Ridge also reported), one regressor for both wings |

The orientation and wing models follow the Stanford project's approach: rotate the
fly upright, compute HOG, reduce with PCA, then fit a linear model.

## Dataset

The dataset comes from the Stanford CS229 project and is **not included** in this
repository.

- **326 annotated images** in **5 recording sessions** (one folder per session):

  | Session folder | Images | Content |
  |---|---|---|
  | `12-04_17-54-43` | 100 | sampled uniformly from the original video |
  | `12-05_12-43-00` | 33 | the two flies together |
  | `12-07_16-45-00` | 50 | flies running along the wall |
  | `12-08_11-15-00` | 102 | wing-angle frames from `test4.mp4` |
  | `12-08_22-00-00` | 41 | wing-angle frames from `test1.mp4` |

- Point annotations (labelme JSON) for each image:

  | Label | Meaning | Images |
  |---|---|---|
  | `mp`, `fp` | male / female body point | 326 |
  | `mh`, `ma` | male head / abdomen | 277 |
  | `fh`, `fa` | female head / abdomen | 217 |
  | `mp2` | second male body point (thorax) | 258 |
  | `mw` | male wing tips (two per image) | 257 |

- Five test videos, `test1.mp4` … `test5.mp4` (1530 × 1530, 35 FPS).

## Setup

Python 3.11. With conda:

```bash
conda create -n cs229-project python=3.11
conda activate cs229-project
pip install -r requirements.txt
```

### Where to put the data

Place the dataset in `./input` at the repository root. A symlink to an existing
copy also works.

```text
input/
├── images/
│   ├── 12-04_17-54-43/   1.bmp, 1.json, 2.bmp, 2.json, ...
│   ├── 12-05_12-43-00/   *.png + *.json
│   ├── 12-07_16-45-00/
│   ├── 12-08_11-15-00/
│   └── 12-08_22-00-00/
└── videos/
    ├── test1.mp4
    └── ...
```

`input/` is git-ignored. Never commit it.

## How to run

Run every command from the repository root, in this order:

```bash
# 1. Train the four models (each writes only its own file in models/)
python train/train_fly_count.py     # also writes results/metrics/fly_count_tree.txt
python train/train_sex.py
python train/train_orientation.py
python train/train_wing_angle.py

# 2. Cross-validated evaluation -> results/metrics/evaluation.{csv,md}, results/plots/*_pca.png
python evaluate.py

# 3. Video pipeline -> results/videos/final_demo.mp4, results/metrics/pipeline_stats.json
python main.py                          # full test1.mp4 with a preview window (Q to stop)
python main.py --max-frames 50          # quick smoke test
python main.py --no-display --video input/videos/test4.mp4

# 4. Regenerate results/metrics/model_results.txt from steps 2 and 3
python make_report.py
```

Each trainer prints a grouped hold-out check, then saves a model fit on all its
data. `src/model_io.py` makes sure each trainer can only write its own model file.

## Results

`evaluate.py` scores every model in three ways, each reported as mean ± std:

- **Random split:** the original style, a random 80/20 split over rows, repeated
  with 10 seeds.
- **GroupKFold by image:** 5 folds; all contours and both wings of one image stay
  in the same fold.
- **Leave-one-session-out:** a whole recording folder is held out. This is the
  strictest test, because frames within a session are near-duplicates. Only
  sessions that contain examples of a task get a fold (see Limitations).

| Model | Before fixes (hand-written, one split) | After: GroupKFold by image | After: leave-one-session-out |
|---|---|---|---|
| Fly count, per contour | 100% (circular labels) | 100.0 ± 0.0% | 99.6 ± 0.8% |
| Fly count, per image (= 2) | — | 86.0 ± 1.8% | 74.0 ± 37.6% |
| Sex | 88.46% (one 52-sample split) | 96.9 ± 2.4% | 83.7 ± 17.1% |
| Orientation, flip error | 0% (keypoint features, unusable on video) | 1.8 ± 1.7% | 1.7 ± 2.6% |
| Orientation, mean angular error | — | 11.9 ± 1.3° | 13.4 ± 10.2° |
| Wing angle MAE, true orientation | 4.24° (keypoint features, random split) | 4.0 ± 0.4° | 7.5 ± 1.0° |
| Wing angle MAE, predicted orientation | — | 10.8 ± 1.0° | 13.8 ± 4.3° |
| Pipeline speed (test1.mp4) | 45.17 FPS | 193.5 FPS processing, 38.4 FPS wall clock with video writing and preview (Apple M5) | |

Baselines, leave-one-session-out:

- **Orientation:** always predicting the majority flip gives a 23.1% flip error and
  a 49.6° mean angular error.
- **Wing angle:** predicting the training mean gives a 21.0° MAE.

Even with a perfect flip, the moment axis alone has a 13.1° mean angular error. Most
of the remaining orientation error therefore comes from the axis, not from the flip
classifier.

The full tables are in:

- `results/metrics/evaluation.md`
- `results/metrics/model_results.txt` (includes the original numbers in a
  "before fixes" section)

The PCA-size curves are in `results/plots/orientation_pca.png` and
`results/plots/wing_pca.png`. The audit that motivated the fixes is in
`results/metrics/audit.md`.

### Why the "before" numbers were wrong

- **Fly count:** each video frame was labelled `min(#contours, 2)`, and the
  features were the sizes of those same contours. The 100% was guaranteed by
  construction, and the training script was a copy of the sex trainer that
  overwrote `models/sex_model.pkl`.
- **Orientation and wing angle:**
  - The models were trained on distances and angles between the annotated
    keypoints, from which the label is derived.
  - At inference, the pipeline fed them placeholder constants.
  - The random splits put near-duplicate rows (for example, the two wings of one
    image) into both train and test.

## Limitations

- **Small data:** 326 annotated images from only 5 sessions, and not every task
  has examples in every session. Leave-one-session-out has 5 folds for the fly
  count, 4 for orientation and wing angle, and only 3 for sex. The standard
  deviations are large: sex accuracy is 91%, 100% or 60% depending on the
  held-out session, and the 60% fold has only 10 flies.
- **Detector recall:**
  - 188 annotated flies, mostly near the arena wall, merge with the dark wall into
    a border-touching blob that the detector discards.
  - 76 of the 326 images have no usable contour at all.
  - The per-image count accuracy is therefore an upper bound.
  - On `test1.mp4` the frame count is 2 in 256 of 351 frames. On `test4.mp4` it is
    2 in all 632 frames.
- **Touching flies:** a contour predicted to contain two flies is counted
  correctly, but it is not split, so sex, orientation and wing angle are not
  predicted for it.
- **Orientation labels:**
  - Head and abdomen labels exist for males (`mh`/`ma`, 277 images) and for
    females (`fh`/`fa`, 217 images), so orientation is trained and applied for
    both sexes.
  - Only isolated flies can be used: 138 males and 88 females.
- **Wing angle:**
  - Only 137 isolated males have both wing tips annotated, and the model runs only
    on flies predicted male.
  - The target is the angle between the head→abdomen axis and the line from the
    body point `mp` to the wing tip.
- **Orientation error propagates to the wings:** the moment axis is bent by spread
  wings, and this error carries into the wing angle. The wing MAE roughly doubles
  when the patch is rotated by the predicted heading instead of the annotated one.
- **Hyperparameters:**
  - The PCA sizes (20 for orientation, 80 for wing) were chosen with GroupKFold by
    image, not with nested cross-validation.
  - The leave-one-session-out curves show that the choice is not critical for
    orientation.
- **The demo video is not held-out data:** session `12-08_22-00-00` was taken
  from `test1.mp4` (and `12-08_11-15-00` from `test4.mp4`), so the demo shows
  frames close to the training data.
- **Speed:** FPS depends on the machine. The wall-clock number includes writing a
  1530 × 1530 video and the preview window.

## Project structure

```text
fruit-fly-video-analysis/
├── main.py                  entry point (--max-frames, --no-display, --video)
├── evaluate.py              random / GroupKFold / leave-one-session-out evaluation
├── make_report.py           writes results/metrics/model_results.txt
├── audit_phase1.py          checks behind results/metrics/audit.md
├── src/
│   ├── detection.py         shared thresholding + contour detection + moment axis
│   ├── features.py          shared count / upright-patch / HOG / wing-half features
│   ├── annotations.py       loading of labelme JSON + matching flies to contours
│   ├── model_io.py          save guard: each trainer may only write its own model
│   ├── pipeline.py          video pipeline
│   └── ...                  earlier exploratory scripts (fly_count.py is deprecated)
├── train/
│   ├── train_fly_count.py
│   ├── train_sex.py
│   ├── train_orientation.py
│   └── train_wing_angle.py
├── models/                  trained models (*.pkl)
└── results/
    ├── metrics/             evaluation.md/.csv, model_results.txt, audit.md, ...
    ├── plots/
    └── videos/              final_demo.mp4 (git-ignored)
```

## Credits

The dataset, problem setup and the HOG + PCA approach come from the Stanford CS229
project by **Steven Herbst**:
[github.com/sgherbst/cs229-project](https://github.com/sgherbst/cs229-project),
released under the MIT license (© 2018 Steven Herbst). The original video was
provided to that project by Ryan York.
