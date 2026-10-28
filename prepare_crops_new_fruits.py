#!/usr/bin/env python
"""Preprocess and crop bounding boxes from object detection datasets (Pineapple and Sawo)
into class-labeled single-fruit images for image classification training.

Usage:
    python prepare_crops_new_fruits.py --fruit all
    python prepare_crops_new_fruits.py --fruit sawo
    python prepare_crops_new_fruits.py --fruit pineapple --video-stride 3
"""
import argparse
import collections
import csv
import os
import re
import sys
import time
from pathlib import Path
from PIL import Image

def crop_sawo(root="sawo", out=None, pad=0.10):
    root_p = Path(root)
    out_p = Path(out or root_p / "sawo_by_class")
    names = ["Matang", "Mentah", "Rusak"]
    folder_map = {"Mentah": "1_unripe", "Matang": "2_ripe", "Rusak": "3_rotten"}
    splits = {"train": "Training", "test": "Testing"}
    
    out_p.mkdir(parents=True, exist_ok=True)
    for sp in splits.values():
        for c in folder_map.values():
            (out_p / sp / c).mkdir(parents=True, exist_ok=True)
            
    rows = []
    counts = collections.Counter()
    t0 = time.time()
    
    for split_dir, sp in splits.items():
        lbl_dir = root_p / split_dir / "labels"
        img_dir = root_p / split_dir / "images"
        if not lbl_dir.is_dir():
            continue
            
        for lf in sorted(lbl_dir.glob("*.txt")):
            stem = lf.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if not img_path:
                continue
                
            m = re.match(r"^(.*?)_([a-zA-Z0-9]+)\.rf\.[a-f0-9]{32}$", stem)
            orig_stem = m.group(1) if m else stem
            rf_hash = stem.split(".rf.")[-1] if ".rf." in stem else "orig"
            
            try:
                im = Image.open(img_path).convert("RGB")
                W, H = im.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
                
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]
                
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cls_name = names[cls_id]
                x, y, w, h = (float(v) for v in parts[1:5])
                
                if w * W < 15 or h * H < 15:
                    continue
                    
                x0 = max(0, (x - w/2 - w*pad) * W)
                y0 = max(0, (y - h/2 - h*pad) * H)
                x1 = min(W, (x + w/2 + w*pad) * W)
                y1 = min(H, (y + h/2 + h*pad) * H)
                
                safe_orig = re.sub(r"[^a-zA-Z0-9_-]", "_", orig_stem)
                crop_name = f"{safe_orig}_{rf_hash[:8]}_b{i}.jpg"
                folder_cls = folder_map[cls_name]
                dst = out_p / sp / folder_cls / crop_name
                
                im.crop((round(x0), round(y0), round(x1), round(y1))).save(dst, quality=95)
                counts[(sp, folder_cls)] += 1
                rows.append({
                    "src": str(img_path), "dst": str(dst), "split": sp, "orig_stem": orig_stem,
                    "rf_hash": rf_hash, "label": cls_name, "class": folder_cls,
                    "crop_w": round(x1 - x0), "crop_h": round(y1 - y0)
                })
                
    csv_path = out_p / "crops.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
        
    print(f"Done cropping Sawo! Total crops: {len(rows)} in {time.time()-t0:.1f}s")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


def crop_pineapple(root="pineapple", out=None, video_stride=3, pad=0.10):
    root_p = Path(root)
    out_p = Path(out or root_p / "pineapple_by_class")
    names = ["Ripe", "Semi_Ripe", "Un_Ripe"]
    folder_map = {"Un_Ripe": "1_unripe", "Semi_Ripe": "2_semiripe", "Ripe": "3_ripe"}
    splits = {"train": "Training", "valid": "Validation", "test": "Testing"}
    
    out_p.mkdir(parents=True, exist_ok=True)
    for sp in splits.values():
        for c in folder_map.values():
            (out_p / sp / c).mkdir(parents=True, exist_ok=True)
            
    rows = []
    counts = collections.Counter()
    t0 = time.time()
    
    for split_dir, sp in splits.items():
        lbl_dir = root_p / split_dir / "labels"
        img_dir = root_p / split_dir / "images"
        if not lbl_dir.is_dir():
            continue
            
        files = sorted(lbl_dir.glob("*.txt"))
        print(f"Cropping pineapple {split_dir} ({len(files)} files, video stride={video_stride}) ...")
        
        for lf in files:
            stem = lf.stem
            m_vid = re.match(r"^Pineapple(\d+)_", stem)
            if m_vid:
                frame_idx = int(m_vid.group(1))
                if frame_idx % video_stride != 0:
                    continue
                orig_group = f"video_{frame_idx // 30}"
            else:
                m_orig = re.match(r"^(.*?)_([a-zA-Z0-9]+)\.rf\.[a-f0-9]{32}$", stem)
                orig_group = m_orig.group(1) if m_orig else stem
                
            rf_hash = stem.split(".rf.")[-1] if ".rf." in stem else "orig"
            
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if not img_path:
                continue
                
            try:
                im = Image.open(img_path).convert("RGB")
                W, H = im.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
                
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]
                
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cls_name = names[cls_id]
                x, y, w, h = (float(v) for v in parts[1:5])
                
                if w * W < 20 or h * H < 20:
                    continue
                    
                x0 = max(0, (x - w/2 - w*pad) * W)
                y0 = max(0, (y - h/2 - h*pad) * H)
                x1 = min(W, (x + w/2 + w*pad) * W)
                y1 = min(H, (y + h/2 + h*pad) * H)
                
                safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
                crop_name = f"{safe_stem[:30]}_{rf_hash[:8]}_b{i}.jpg"
                folder_cls = folder_map[cls_name]
                dst = out_p / sp / folder_cls / crop_name
                
                im.crop((round(x0), round(y0), round(x1), round(y1))).save(dst, quality=95)
                counts[(sp, folder_cls)] += 1
                rows.append({
                    "src": str(img_path), "dst": str(dst), "split": sp, "orig_group": orig_group,
                    "label": cls_name, "class": folder_cls,
                    "crop_w": round(x1 - x0), "crop_h": round(y1 - y0)
                })
                
    csv_path = out_p / "crops.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
        
    print(f"Done cropping Pineapple! Total crops: {len(rows)} in {time.time()-t0:.1f}s")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fruit", choices=["sawo", "pineapple", "all"], default="all")
    parser.add_argument("--video-stride", type=int, default=3)
    args = parser.parse_args()
    
    if args.fruit in ("sawo", "all"):
        crop_sawo()
    if args.fruit in ("pineapple", "all"):
        crop_pineapple(video_stride=args.video_stride)
