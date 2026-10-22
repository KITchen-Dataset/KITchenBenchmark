# KITchen benchmark v1.0: design and rules

Status: `1.0`, frozen on 2026-10-21. Numbers on earlier release candidates
are not comparable with `1.0`.

## 1. Goals

1. **Measure what the dataset is good at.** KITchen has 2D boxes and masks for
   112 classes on real kitchen frames seen from a humanoid's head camera,
   RGB-D, and 6D poses for 111 scanned objects. The tasks follow the
   annotations: 2D detection, instance segmentation, 6D pose.
2. **Plug into existing tools.** COCO protocol for 2D, BOP protocol for 6D, BOP
   file formats, so a detector trained with Detectron2/MMDetection/Ultralytics
   or a pose method from the BOP challenge runs without glue code, and numbers
   are comparable with COCO/BOP literature.
3. **Be reproducible.** Every ground-truth file is checksummed in
   `benchmark.json`; every result card carries that checksum, the prediction
   checksum and the evaluator version. Versions are frozen and never edited.
4. **Be honest about the labels.** Only verified annotations are scored;
   uncertain ones are ignore regions, never false negatives.

## 2. Splits

KITchen has no recorded sequence boundaries. The release is short clips in
mixed order, consecutive frames are close to duplicates, and the same physical setup (table plus object arrangement) comes
back in many clips. A random frame split, or a split by blocks of frames, would
put the same scene in train and test. So the split is made by kitchen and by
setup:

* The recordings come from **two kitchens** that never interleave: a lab room
  and a built-in kitchen.
* **Train is the lab room; val and test are the built-in kitchen.** Every
  score therefore measures a method in a kitchen it has not seen during
  training.
* Inside each kitchen, frames are grouped into **physical setups** by
  clustering the object poses (two clips share a setup when their objects
  agree within 30 mm): 65 setups, 46 in the lab room and 19 in the built-in
  kitchen. No setup appears in both kitchens.
* The built-in kitchen's setups go **whole** to val or test (fixed seed 2431),
  about one third of its frames to val and two thirds to test. No setup is in
  two splits.
* Clips whose setup could not be identified (0.5 % of frames) are left out.

| Split | Frames | Setups | Scored pose targets | Boxes and masks |
|---|---|---|---|---|
| train | 73.5 % | 46 | | 2,387,772 annotations |
| val | 8.6 % | 9 | 144,964 | 178,522 |
| test | 17.4 % | 10 | 377,846 | 443,898 |

* The paper gives 70/20/10 for train/val/test without fixing how frames are
  assigned. The kitchen boundary gives 73.5 / 8.6 / 17.4 %, close to it, with
  the larger share kept for test because a larger test set ranks methods more
  reliably.
* All 107 posed objects of the built-in kitchen are in both val and test; one
  object (TomatoSoup, id 90) appears only in the lab room.
* **Every val and test frame is scored** (`splits/{val,test}_eval.txt` list
  all of them).
* `splits/setups.csv` gives the setup and split of every frame. BOP
  `scene_id = frame // 250` is only a file grouping, not a split unit.

`splits/{train,val,test}.txt` list every frame of each split, so training can
use all train frames.

## 3. Ground truth

| Annotation | Scored | Ignored (no reward, no penalty) |
|---|---|---|
| Boxes and masks | every confirmed label row | unconfirmed rows (listed in `ignore.json`) become COCO `iscrowd` regions |
| 6D poses | accepted poses (the pose version is named in `benchmark.json`) | any (frame, object) pair where an instance lacks an accepted pose: listed in `test_ignore.json`; estimates for it are dropped |
| Visibility | objects with ≥10% of their silhouette visible (BOP rule) | objects with less |

Class 0 ("other") is not part of the benchmark. The Coke bottle has no
released poses (it is transparent) and is scored for 2D only.

A few rows of the release have an empty mask file; they become box-only ignore
regions, and their objects are not pose targets in that frame.

Boxes are a little looser than the tight boxes of the masks (median IoU 0.93),
so a method that takes its boxes from its masks scores lower `det` AP than
`segm` AP.

`visib_fract` is the annotated mask area divided by the area the posed model
covers in the image.

## 4. Metrics

**2D (`det`, `segm`).** COCO AP averaged over IoU 0.50:0.95, with AP50, AP75,
small/medium/large AP. KITchen frames hold up to ~80 objects, so up to 300
detections per image count (as in LVIS); AR300 is reported. Scored with
`pycocotools`.

**6D (`pose`).** The paper defines the task as instance-level 6D pose of every
known object from one egocentric RGB-D view, scored with BOP errors. Primary: BOP 2024 6D detection,
`AP_6D = (AP_MSSD + AP_MSPD) / 2`, each averaged over the 10 BOP thresholds
(MSSD 5-50% of the object diameter; MSPD 5-50 px at 640 px width) and over
objects. Secondary: BOP 2019 6D localization recall `AR_6D = (AR_MSSD + AR_MSPD) / 2`
(the number of instances per object is given). Up to 200 estimates per image
count. MSSD/MSPD are symmetry-aware through `models_info.json`; symmetries
were proposed by shape and texture analysis, then checked on rendered views
of each textured model: eight cups and bowls have a continuous axis, the oil
bottle and the corn can a 180° flip. VSD is
not part of the KITchen score because it needs a renderer; since the test set
is in BOP format, the official `bop_toolkit` can add it. Our implementation is
tested to give the same numbers as `bop_toolkit_lib` (`tests/test_bop_parity.py`).

Per-object scores are kept in the result card so papers can show where a
method fails.

## 5. Rules

1. Train on `train` only. `val` is for tuning and model selection, never
   training data, because it shows the test kitchen. No test frame and no
   test annotation may be used for training, tuning or model selection.
2. Declare the input (`rgb`, `rgbd`, `depth`) and all training data, including
   pre-training and synthetic data. RGB and RGB-D methods are ranked
   separately.
3. Object meshes may be used (model-based setting). Report which depth you
   used (raw or completed).
4. Report results on the `test` split of a named benchmark version. Do not
   tune on test; report each method variant once.
5. **One model for all objects** (pose). This rule comes from the KITchen paper
   (Younes et al., 2024): a robot in a kitchen cannot load one network per
   object. Declare it with `single_model` in the method file.
6. **At least 5 fps** (pose), also from the paper, so methods are usable on a
   robot such as ARMAR-6. Every pose row must carry `time`: wall time in
   seconds to estimate all objects of the image, measured as BOP does, with
   the hardware stated. The card shows the frame rate and whether both rules
   are met; on the leaderboard, methods that meet them rank first. Methods
   that break them may still be reported but are marked.

## 6. Versioning

`benchmark.json` holds the version, the source versions (pose QA file and its
sha256, models, symmetries), the split configuration, per-split counts and the
sha256 of every ground-truth file. `kitchenbench verify` checks them. Any
change to labels, poses, splits or rules produces a new version; old versions
stay available so published numbers remain reproducible.

| Version | Released | Pose release | Ground truth checksum | Notes |
|---|---|---|---|---|
| 1.0 | 2026-10-21 | v3.3 | `665635036314` | first public version |
