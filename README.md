# Fruit type, ripeness, and quality classifiers (MVP)

PyTorch transfer learning (MobileNetV3) -> Core ML. Three image models, plus frame-sampling inference for video.

| Model | Predicts | Label space |
|---|---|---|
| `fruit` | avocado / dragon_fruit / mango / papaya / pineapple / sawo | folder-level fruit identity |
| `ripeness` | ripeness stage | **original labels prefixed by fruit** (e.g. `mango/Semiripe`, `avocado/5_rotten`) |
| `quality` | DragonFruit quality | original folder labels `Defect Dragon Fruit` / `Fresh Dragon Fruit` |

No labels are merged by default. `label_map.yaml` documents the originals and an optional name-identical normalized view
(`unripe/semiripe/ripe/overripe`; `Half_Ripe`, `Fully Ripe`, `Perished`, `rotten` are excluded from it).
DragonFruit maturity labels (`Immature Dragon Fruit` / `Mature Dragon Fruit`) are included in ripeness when
`prepare.include_dragon_maturity_in_ripeness` is true. Quality is a separate task because one fruit can be mature and
fresh/defect at the same time; quality labels are not mixed into the mutually exclusive ripeness classes.

Banana folders remain in the dataset on disk, but Banana is excluded from classification: `prepare_splits.py` does not
scan those folders, and `train.py` filters fruit and ripeness rows through `label_map.yaml`'s `fruit_types` (including
when an older split manifest is supplied).

## Status

* `smoke_test.py` exercises the pipeline on synthetic data. Real fruit and ripeness checkpoints/exports are present in `runs/`,
  but the existing ripeness checkpoint predates DragonFruit maturity being enabled. No quality checkpoint exists yet.
* Existing metrics are from the generated dataset split, not independent phone-camera validation. The README's source/background
  and papaya video caveats still apply. Core ML `--verify-image` should be run on macOS.

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py            # ~1 min, synthetic data, checks your install end to end
```

## Run

```bash
DATA=/path/to/dataset/root      # the folder that contains DragonFruit/, avocado_hass_by_class/, ...

# 1. Scan (read-only), dedupe, build leakage-aware groups, write the split manifest + report
python prepare_splits.py --data-root "$DATA"
#    -> artifacts/splits.csv, artifacts/data_report.md   (READ the report first)

# 2. Train (device auto-detects MPS/CUDA/CPU; first run downloads ImageNet weights)
python train.py --task fruit
python train.py --task ripeness
python train.py --task quality
#    -> runs/<task>_mobilenet_v3_large/{best.pth, metrics.json, confusion_{val,test}.png/csv,
#       classification via metrics.json, predictions_*.csv, history.csv, labels.json}

# 3. Export to Core ML
python export_coreml.py --ckpt runs/fruit_mobilenet_v3_large/best.pth    --verify-image some_fruit.jpg
python export_coreml.py --ckpt runs/ripeness_mobilenet_v3_large/best.pth --verify-image some_fruit.jpg
python export_coreml.py --ckpt runs/quality_mobilenet_v3_large/best.pth  --verify-image some_dragonfruit.jpg

# 4. Inference (run inside the Ultralytics environment; detection happens before classification)
python infer.py --det-ckpt runs/detect/fruit_detection_clean/weights/best.mlpackage \
                --fruit-ckpt runs/fruit_mobilenet_v3_large/best.pth \
                --ripeness-ckpt runs/ripeness_mobilenet_v3_large/best.pth --input photo.jpg
python infer.py --det-ckpt runs/detect/fruit_detection_clean/weights/best.mlpackage \
                --fruit-ckpt runs/fruit_mobilenet_v3_large/best.pth \
                --ripeness-ckpt runs/ripeness_mobilenet_v3_large/best.pth \
                --quality-ckpt runs/quality_mobilenet_v3_large/best.pth --input dragonfruit.jpg
python infer.py --det-ckpt runs/detect/fruit_detection_clean/weights/best.mlpackage \
                --fruit-ckpt runs/fruit_mobilenet_v3_large/best.pth \
                --ripeness-ckpt runs/ripeness_mobilenet_v3_large/best.pth \
                --input clip.mp4 --sample-fps 2 --max-frames 32 --per-frame
```

Useful options: `--arch mobilenet_v3_small|efficientnet_b0`, `--epochs`, `--img-size`, `--batch-size`, `--workers 0`,
`--ripeness-labels normalized`. Defaults live in `config.yaml` (seed 42, early stopping on val macro-F1, per-class cap of
1,500 images per epoch, class-weighted loss, no hue jitter).

## Troubleshooting

* **`urlopen error ... nodename nor servname provided`** (cannot download ImageNet weights): fix DNS/VPN, test with
  `curl -I https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth`. If your Mac cannot reach it, download that file elsewhere
  (browser, phone hotspot) and put it, with the same file name, in `~/.cache/torch/hub/checkpoints/`. Training then picks it up offline.
  (`mobilenet_v3_small`: `mobilenet_v3_small-047dcff4.pth`; `efficientnet_b0`: `efficientnet_b0_rwightman-7f5810bc.pth`.)
  `--no-pretrained` trains from scratch and will be much worse.
* **`prepare_splits.py` exits with "split is UNUSABLE"**: some class is missing from train/val/test (usually one giant group). It writes
  `artifacts/splits_REJECTED.csv` instead of `splits.csv`, and prints the largest groups. Read `artifacts/data_report.md`.

## How leakage is handled

* One manifest and one split for all tasks (70/15/15, stratified by fruit and available task label, split **by group**).
* Exact duplicates (MD5) are merged. Near-duplicates (64-bit dHash, Hamming <= 4) are merged only for images WITHOUT a filename/CSV group key
  (mango, DragonFruit), and never into a group larger than `near_dup_max_group` (50). Images that have keys (avocado, papaya, pineapple, sawo) are grouped by their keys.
* Groups: avocado = `fruit_id` (from `labels_matched.csv`, falling back to the filename); papaya = video + 10-frame block;
  pineapple = source image/video block; sawo = source image; mango and DragonFruit have no IDs.
* The script asserts that no group or hash spans two splits. The original Training/Testing folders are ignored and resplit.

## Known limitations (also written into every `metrics.json`)

* Mango/DragonFruit: same-fruit photos may straddle splits -> optimistic scores.
* DragonFruit quality labels come from a separate source collection without fruit IDs; Fresh/Defect metrics may be optimistic and need independent validation.
* Papaya: each class comes from one video, so ripeness scores largely measure scene identity. Treat papaya ripeness as untrusted until you have new data.
* Fruit-type task: each fruit comes from a different source, so the model can learn backgrounds/cameras. There is no "not a fruit" class.
  Use `--accept-conf` (default 0.6) and test on your own phone photos.
* Not production-ready evidence. Collect an independent held-out set from the iOS camera before shipping.

## iOS integration notes

* Model input is an RGB image named `image`; scaling/normalization is inside the model; output is softmax probabilities + `classLabel`.
* In Vision set `request.imageCropAndScaleOption = .centerCrop` (matches training/eval preprocessing).
* Ripeness model uses fruit-prefixed labels: take the fruit-model winner, keep only ripeness classes starting with `"<fruit>/"`, renormalize,
  and show nothing for fruits without ripeness classes (`ripeness_fruits` in the model metadata / `labels.json`). `infer.py` is the reference logic.
* For DragonFruit, return maturity from the ripeness output and quality from the optional `quality` model. The quality model is only emitted when the fruit prediction is `dragon_fruit`.
* Detection-first `infer.py` runs YOLO on each input frame, takes the highest-confidence box, crops it with 5% padding, then calls the classifiers.
  It returns `no_fruit_detected` without classifier predictions if no box clears `--det-conf`.
* For video, sample ~2 frames/s, classify detected crops, average probabilities (weighting ripeness by fruit confidence), and flag low-confidence results.

## Files

`prepare_splits.py` scan/dedupe/groups/splits/report - `train.py` fruit/ripeness/quality train+evaluate - `prepare_detection_dataset.py` unified detector data - `infer.py` detector then classifiers
- `smoke_test.py` synthetic end-to-end check - `config.yaml` settings - `label_map.yaml` label documentation - `common.py`, `models.py` helpers.

## Detection data

Detection data for Avocado, Dragon Fruit, Mango, Papaya, Pineapple, and Sawo is combined by `prepare_detection_dataset.py` into
`detection_preprocessed/` as a single `fruit` class, without changing classification splits. Train with the installed Ultralytics
CLI and pass `patience=20` for detection early stopping. `infer.py` requires a YOLO `.pt` or Core ML `.mlpackage` detector checkpoint;
run it in an environment with Ultralytics installed.
