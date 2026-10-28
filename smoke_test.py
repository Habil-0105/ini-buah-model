#!/usr/bin/env python
"""End-to-end CODE-PATH test on a tiny SYNTHETIC dataset that mimics the handoff folder layout.

It checks that scanning, dedupe, grouping, training, evaluation, export and inference run on your machine.
The images are random colour noise: the metrics it prints mean nothing. Run:  python smoke_test.py
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def img(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.array(color)[None, None, :] + rng.normal(0, 40, (128, 128, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path, quality=90)


def img_blob(path, color):
    """Plain background + centred blob: every image gets (nearly) the same dHash, like lab photos of fruit.
    Regression test for the 'everything chains into one giant group' failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((128, 128, 3), 210.0)
    yy, xx = np.mgrid[:128, :128]
    a[(yy - 64) ** 2 + (xx - 64) ** 2 < 40 ** 2] = np.array(color)
    a += rng.normal(0, 2, a.shape)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path, quality=90)


def color_for(key):
    r = np.random.default_rng(abs(hash(key)) % 2**32)
    return r.integers(30, 225, 3)


def build(root: Path):
    import pandas as pd
    for lab in ["Immature Dragon Fruit", "Mature Dragon Fruit"]:
        for i in range(14):
            img(root / "DragonFruit/Maturity_Grading" / lab / f"{lab.split()[0]}_Dragon_Original_Data{i:04d}.jpg",
                color_for(("dragon", lab)))
    for lab in ["Defect Dragon Fruit", "Fresh Dragon Fruit"]:
        for i in range(14):
            img(root / "DragonFruit/Quality_Grading" / lab / f"{lab.split()[0]}_Dragon_Original_Data{i:04d}.jpg",
                color_for(("dragonq", lab)))
    for lab in ["Unripe", "Semiripe", "Fully Ripe", "Overripe", "Perished"]:
        for i in range(14):
            path = root / "Mango Shelf-Life Dataset" / lab / f"image_0_{abs(hash((lab, i))) % 10**6}.jpg"
            img(path, color_for(("mango", lab)))
            if lab == "Unripe" and i == 0:
                shutil.copy(path, path.with_name("image_0_exact_duplicate.jpg"))
    rows = []
    for k, cls in enumerate(["1_unripe", "2_semiripe", "3_ripe", "4_overripe", "5_rotten"]):
        for i in range(14):
            split = "Training" if i % 4 else "Testing"
            fid = f"T10_{k * 20 + i:03d}"
            name = f"{fid}__T10_d01_{k * 20 + i:03d}_a_{k + 1}.jpg"
            img_blob(root / "avocado_hass_by_class" / split / cls / name, color_for(("avocado", cls)) * 0.6)
            rows.append(dict(src="x", dst=f"/other/machine/{split}/{cls}/{name}", fruit_id=fid, storage="T10",
                             label=k + 1, **{"class": cls}, split=split))
    pd.DataFrame(rows).to_csv(root / "avocado_hass_by_class/labels_matched.csv", index=False)
    for cls in ["1_unripe", "2_semiripe", "3_ripe", "4_rotten"]:
        for i in range(14):
            split = "Training" if i % 4 else "Testing"
            img(root / "papaya/papaya_by_class" / split / cls / f"a_f{100 + 10 * i:03d}_aug0_b0.jpg",
                color_for(("papaya", cls)))
    for k, cls in enumerate(["1_unripe", "2_semiripe", "3_ripe"]):
        for i in range(14):
            split = "Training" if i % 4 else "Testing"
            img(root / "pineapple/pineapple_by_class" / split / cls / f"Pineapple{k * 1000 + i * 60:05d}_jpg_rf_00000000_00000000_b0.jpg",
                color_for(("pineapple", cls)))
    for k, cls in enumerate(["1_unripe", "2_ripe", "3_rotten"]):
        for i in range(14):
            split = "Training" if i % 4 else "Testing"
            img(root / "sawo/sawo_by_class" / split / cls / f"sawo_{k}_{i:03d}_00000000_b0.jpg",
                color_for(("sawo", cls)))


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"FAILED: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="fruit_smoke_"))
    root, work = tmp / "dataset", tmp / "work"
    root.mkdir()
    build(root)
    work.mkdir()
    cfg_over = ["--img-size", "96", "--epochs", "2", "--batch-size", "16", "--workers", "0", "--no-pretrained",
                "--arch", "mobilenet_v3_small", "--device", "cpu"]
    # run everything with artifacts/runs inside the temp dir
    import os
    os.chdir(work)
    env_cfg = work / "config.yaml"
    import yaml
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg["prepare"]["artifacts_dir"] = str(work / "artifacts")
    cfg["prepare"]["workers"] = 1
    cfg["train"]["runs_dir"] = str(work / "runs")
    cfg["train"]["cap_per_class_per_epoch"] = 40
    env_cfg.write_text(yaml.safe_dump(cfg))
    c = ["--config", env_cfg]
    run([HERE / "prepare_splits.py", *c, "--data-root", root])
    run([HERE / "train.py", "--task", "fruit", *c, *cfg_over])
    run([HERE / "train.py", "--task", "ripeness", *c, *cfg_over])
    run([HERE / "train.py", "--task", "ripeness", *c, *cfg_over, "--ripeness-labels", "normalized",
         "--out", work / "runs/ripeness_normalized"])
    fck = work / "runs/fruit_mobilenet_v3_small/best.pth"
    rck = work / "runs/ripeness_mobilenet_v3_small/best.pth"
    sample = next((root / "Mango Shelf-Life Dataset/Unripe").glob("*.jpg"))
    run([HERE / "infer.py", "--skip-detection", "--fruit-ckpt", fck, "--ripeness-ckpt", rck,
         "--input", sample, "--device", "cpu"])
    try:
        import cv2
        vid = work / "synthetic.mp4"
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
        for _ in range(40):
            vw.write(cv2.cvtColor(np.asarray(Image.open(sample).convert("RGB")), cv2.COLOR_RGB2BGR))
        vw.release()
        run([HERE / "infer.py", "--skip-detection", "--fruit-ckpt", fck, "--ripeness-ckpt", rck,
             "--input", vid, "--device", "cpu",
             "--sample-fps", "2", "--max-frames", "8"])
    except ImportError:
        print("opencv not installed: video test skipped")
    if not args.skip_export:
        run([HERE / "export_coreml.py", "--ckpt", fck, "--verify-image", sample])
    print("\nSMOKE TEST PASSED (code paths only; metrics above are meaningless synthetic-noise numbers).")
    print("temp dir:", tmp)
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
