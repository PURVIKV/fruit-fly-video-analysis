# Numbers for the slides

Final results of the fruit-fly video analysis (branch `audit-and-evaluation`).
Every number below is produced by the scripts. The sources are
`results/metrics/evaluation.md`, `video_counts.md`, `orientation_by_sex.md`,
`detector_experiments.md` and `model_results.txt`.

Unless marked otherwise, the numbers are **leave-one-session-out**: a whole
recording session is held out, the model is trained on the other 4, and the
result is the mean ± std over the 5 held-out sessions. This is the strictest test
we have.

## Final results table

| Task | Original claim | Fixed models, old Otsu detector | **Final** |
|---|---|---|---|
| Video frames with the correct fly count (2), all 5 test videos | not measured | 71.3% | **100%** (2703 / 2703) |
| Fly count per annotated image | 100% (circular labels) | 74.0 ± 37.6% | **100%** |
| Sex | 88.46% (one 52-sample split) | 83.7 ± 17.1% | **98.2 ± 3.1%** |
| Orientation: head/tail flip error | 0% (leaky keypoint features) | 1.7% | **0.9 ± 1.2%** |
| Orientation: mean angular error | — | 13.4° | **3.3 ± 2.5°** |
| Body-axis error, male / female | — | 17.2° / 3.6° | **2.6° / 1.1°** |
| Male wing angle MAE, predicted orientation (what the demo shows) | — | 13.8° | **8.6 ± 2.2°** |
| Male wing angle MAE, true orientation | 4.24° (leaky, random split) | 7.5° | 7.8 ± 2.6° (slightly worse) |
| Speed on test1 (Apple M5) | 45.17 FPS | 193.5 FPS | **176.4 FPS** processing; 37.2 FPS including video writing and preview (video is 35 FPS) |

Baselines for comparison:

- **Orientation, always guessing the most common flip:** 63.3% flip error and
  113.5° angular error.
- **Wing angle, always guessing the average:** 24.7° MAE.

Dataset sizes with the final detector:

- 326 annotated images from 5 recording sessions
- 564 flies for sex
- 434 flies for orientation (256 male, 178 female)
- 236 males with both wing tips annotated

## Five key messages

1. **The original 100% results were not real.**
   - The fly-count labels were made from the model's own input.
   - Orientation and wing models were trained on the answer keypoints.
   - The video demo fed those models placeholder numbers.
2. **We now measure honestly.** Every model is tested on whole recording
   sessions it never saw during training (leave-one-session-out), and every
   number comes from a script.
3. **Fixing the detector mattered more than any model change.**
   - Masking the arena and keeping only the dark fly body (threshold 115, as in
     the original Stanford project) stopped flies merging with the arena wall.
   - The video fly count went from 71.3% to 100% of frames, and our fly-count
     rule (area 10,442 px) matches the Stanford model.
4. **Orientation is now accurate for both sexes.**
   - Measuring the body axis on the body core instead of the full silhouette
     cut the male axis error from 17.2° to 2.6°. The male's spread wings had
     been bending the axis.
   - The final heading error is 3.3°, against 113.5° for the baseline.
5. **Limits remain.**
   - Only 326 annotated images.
   - Touching flies (0–14.5% of frames per video) are counted but not
     classified.
   - No fully held-out video with labels: test1 and test4 overlap the training
     data.
   - Wing angle with true orientation did not improve (7.5° → 7.8°).

## Plots and images to use

| What | Path | Notes |
|---|---|---|
| Orientation error vs. number of PCA components | `results/plots/orientation_pca.png` | Leave-one-session-out, final detector |
| Wing-angle error vs. number of PCA components | `results/plots/wing_pca.png` | Shows the gap between true and predicted orientation |
| Demo stills | `results/videos/final_demo_test5_h264.mp4` | Take screenshots, e.g. frame 629 (flies at the arena edge, both detected) |
| Detector comparison table | `results/metrics/detector_experiments.md` | Six candidates; the "Detector choice" table in the README is a short version |
| Do **not** use | `results/plots/model_accuracy.png` | Bar chart of the original, invalid accuracies |

The videos and `results/debug/` are git-ignored, so they exist only on the machine
that ran the scripts. With the final detector there are no count failures, so no
failure frames are saved.

## Demo video: test5

Use `results/videos/final_demo_test5_h264.mp4` (H.264, plays in PowerPoint), made
with `python main.py --video input/videos/test5.mp4 --h264`.

Why test5:

- **Probably held out:** no annotated session was cut from it. test1 and test4
  were used to make training sessions, so they are not a fair test.
- **Clean predictions:** the fly count is correct in all 632 frames, one male and
  one female are predicted in 96.7% of frames, and no frame shows two flies of the
  same sex.
- **Shows the interesting cases:** the male extends his wings (up to about 80°),
  the flies touch in 21 frames, and they meet at the arena edge. Before the
  detector fix they were lost there (frame 629 was a failure; now both are
  detected).

Backup: `final_demo_test3_h264.mp4` tells the before/after story best (17.5% →
100% correct counts, flies running along the wall). But its wings are quiet, and
about 1% of its frames label both flies female.
