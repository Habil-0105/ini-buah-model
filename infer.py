#!/usr/bin/env python
"""Run fruit detection first, then classify detected fruit crops.

  python infer.py --det-ckpt runs/detect/fruit/weights/best.mlpackage \\
      --fruit-ckpt runs/fruit_mobilenet_v3_large/best.pth \\
      --ripeness-ckpt runs/ripeness_mobilenet_v3_large/best.pth --input photo.jpg

Video = sample frames -> detect and crop the most confident fruit per frame -> classify crops -> aggregate.
Native-label ripeness is conditioned on the predicted fruit. DragonFruit maturity and quality
are separate outputs and can both be returned for one input.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import eval_transform, load_rgb, pick_device
from models import build_model

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["arch"], ck["num_classes"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def load_detector(path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for detection-first inference; run this command in .venv-detection") from exc
    try:
        return YOLO(path)
    except Exception as exc:
        raise SystemExit(f"could not load detector checkpoint {path}: {exc}") from exc


def detect_crop(detector, image, device, conf, iou, imgsz, padding, coreml_detector=False):
    """Return the highest-confidence detected fruit crop and its box metadata."""
    predict_args = dict(source=image, conf=conf, iou=iou, imgsz=imgsz, max_det=1, verbose=False)
    if not coreml_detector:
        predict_args["device"] = str(device)
    results = detector.predict(**predict_args)
    result = results[0] if results else None
    boxes = getattr(result, "boxes", None) if result is not None else None
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy[0].detach().cpu().numpy().astype(float)
    confidence = float(boxes.conf[0].detach().cpu())
    class_id = int(boxes.cls[0].detach().cpu())
    names = getattr(detector, "names", {})
    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
    width, height = image.size
    pad_x, pad_y = (xyxy[2] - xyxy[0]) * padding, (xyxy[3] - xyxy[1]) * padding
    x1 = max(0, int(np.floor(xyxy[0] - pad_x)))
    y1 = max(0, int(np.floor(xyxy[1] - pad_y)))
    x2 = min(width, int(np.ceil(xyxy[2] + pad_x)))
    y2 = min(height, int(np.ceil(xyxy[3] + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None, None
    info = {"box_xyxy": [x1, y1, x2, y2], "confidence": confidence,
            "detector_class": label, "detector_class_id": class_id}
    return image.crop((x1, y1, x2, y2)), info


@torch.no_grad()
def probs_for(model, ck, pil_images, device, bs=32):
    tf = eval_transform(ck["img_size"], ck["mean"], ck["std"])
    out = []
    for i in range(0, len(pil_images), bs):
        x = torch.stack([tf(im) for im in pil_images[i:i + bs]]).to(device)
        out.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out)


def sample_frames(video, sample_fps, max_frames):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))
    wanted = list(range(0, n, step)) if n else None
    if wanted and len(wanted) > max_frames:
        wanted = [wanted[i] for i in np.linspace(0, len(wanted) - 1, max_frames).round().astype(int)]
    frames, i, want = [], 0, set(wanted) if wanted else None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if (want is None and i % step == 0) or (want is not None and i in want):
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def aggregate(fruit_p, ripe_p, fruit_ck, ripe_ck, frame_min_conf, accept_conf, quality_p=None, quality_ck=None):
    """Aggregate fruit, optional ripeness, and optional quality predictions across frames."""
    fclasses = fruit_ck["classes"]
    keep = fruit_p.max(1) >= frame_min_conf
    if not keep.any():
        keep[:] = True
    fp = fruit_p[keep]
    fmean = fp.mean(0)
    fi = int(fmean.argmax())
    fruit_candidate = fclasses[fi]
    fruit_confidence = float(fmean[fi])
    # This is a closed-set classifier: it always has a top class, even for
    # blank/background images. Only expose a fruit when it clears the same
    # acceptance threshold used to flag uncertain results.
    fruit = fruit_candidate if fruit_confidence >= accept_conf else None
    res = dict(
        frames_total=int(len(fruit_p)), frames_used=int(keep.sum()),
        fruit=fruit, fruit_confidence=fruit_confidence,
        fruit_frame_agreement=float((fp.argmax(1) == fi).mean()),
        fruit_distribution={c: float(v) for c, v in zip(fclasses, fmean)},
        status="ok" if fruit is not None else "uncertain_fruit",
    )
    res["ripeness"] = None
    if ripe_p is not None and fruit is not None:
        rclasses = ripe_ck["classes"]
        if fruit not in ripe_ck.get("ripeness_fruits", []):
            res["ripeness_note"] = f"no ripeness training data for '{fruit}', so ripeness is not predicted"
        else:
            rp = ripe_p[keep]
            if ripe_ck["label_space"] == "native":
                mask = np.array([c.startswith(fruit + "/") for c in rclasses])
                rp = rp[:, mask]
                names = [c for c, m in zip(rclasses, mask) if m]
                rp = rp / np.maximum(rp.sum(1, keepdims=True), 1e-9)
            else:
                names = list(rclasses)
            wts = fp[:, fi]
            rmean = (rp * wts[:, None]).sum(0) / wts.sum()
            ri = int(rmean.argmax())
            res["ripeness"] = dict(
                label=names[ri], confidence=float(rmean[ri]),
                frame_agreement=float((rp.argmax(1) == ri).mean()),
                distribution={n: float(v) for n, v in zip(names, rmean)},
                label_space=ripe_ck["label_space"],
            )
            if rmean[ri] < accept_conf and res["status"] == "ok":
                res["status"] = "uncertain_ripeness"

    res["quality"] = None
    if quality_p is not None and fruit is not None and fruit in quality_ck.get("quality_fruits", []):
        qp = quality_p[keep]
        qnames = quality_ck["classes"]
        wts = fp[:, fi]
        qmean = (qp * wts[:, None]).sum(0) / wts.sum()
        qi = int(qmean.argmax())
        res["quality"] = dict(
            label=qnames[qi], confidence=float(qmean[qi]),
            frame_agreement=float((qp.argmax(1) == qi).mean()),
            distribution={n: float(v) for n, v in zip(qnames, qmean)},
        )
        if qmean[qi] < accept_conf and res["status"] == "ok":
            res["status"] = "uncertain_quality"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-ckpt", default=None, help="YOLO detector .pt or Core ML .mlpackage; required unless --skip-detection")
    ap.add_argument("--fruit-ckpt", required=True)
    ap.add_argument("--ripeness-ckpt", default=None)
    ap.add_argument("--quality-ckpt", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--frame-min-conf", type=float, default=0.4, help="drop frames whose top fruit prob is below this")
    ap.add_argument("--accept-conf", type=float, default=0.6, help="below this the result is flagged uncertain")
    ap.add_argument("--det-conf", type=float, default=0.25, help="minimum detector confidence")
    ap.add_argument("--det-iou", type=float, default=0.7, help="detector NMS IoU threshold")
    ap.add_argument("--det-imgsz", type=int, default=640, help="detector inference image size")
    ap.add_argument("--crop-padding", type=float, default=0.05, help="fractional padding around each detected box")
    ap.add_argument("--skip-detection", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.skip_detection and args.det_ckpt:
        ap.error("do not combine --skip-detection with --det-ckpt")
    if not args.skip_detection and not args.det_ckpt:
        ap.error("provide --det-ckpt for detection-first inference; --skip-detection is reserved for synthetic smoke tests")
    if not 0 <= args.crop_padding <= 1:
        ap.error("--crop-padding must be between 0 and 1")

    device = pick_device(args.device)
    detector = load_detector(args.det_ckpt) if args.det_ckpt else None
    fm, fck = load_ckpt(args.fruit_ckpt, device)
    rm = rck = None
    if args.ripeness_ckpt:
        rm, rck = load_ckpt(args.ripeness_ckpt, device)
    qm = qck = None
    if args.quality_ckpt:
        qm, qck = load_ckpt(args.quality_ckpt, device)

    p = Path(args.input)
    is_video = p.suffix.lower() in VIDEO_EXT
    images = sample_frames(p, args.sample_fps, args.max_frames) if is_video else [load_rgb(p)]
    if not images:
        raise SystemExit("no frames could be read")
    crops, used_indices, detections = [], [], []
    if detector:
        for frame_index, image in enumerate(images):
            crop, info = detect_crop(detector, image, device, args.det_conf, args.det_iou,
                                     args.det_imgsz, args.crop_padding,
                                     coreml_detector=args.det_ckpt.lower().endswith((".mlpackage", ".mlmodel")))
            if crop is None:
                detections.append({"frame": frame_index, "detected": False})
                continue
            crops.append(crop)
            used_indices.append(frame_index)
            detections.append({"frame": frame_index, "detected": True, **info})
    else:
        crops = images
        used_indices = list(range(len(images)))

    if not crops:
        res = dict(frames_total=len(images), frames_used=0, fruit=None, fruit_confidence=None,
                   fruit_frame_agreement=0.0, fruit_distribution={}, status="no_fruit_detected",
                   ripeness=None, quality=None)
        res["detection"] = {"enabled": detector is not None, "frames_with_fruit": 0,
                            "confidence_threshold": args.det_conf, "frames": detections}
        res["input"] = str(p)
        res["type"] = "video" if is_video else "image"
        print(json.dumps(res, indent=2))
        return

    fp = probs_for(fm, fck, crops, device)
    rp = probs_for(rm, rck, crops, device) if rm else None
    qp = probs_for(qm, qck, crops, device) if qm else None
    res = aggregate(fp, rp, fck, rck, args.frame_min_conf if is_video else 0.0, args.accept_conf,
                    quality_p=qp, quality_ck=qck)
    res["frames_total"] = len(images)
    res["frames_with_fruit"] = len(crops)
    res["detection"] = {"enabled": detector is not None, "confidence_threshold": args.det_conf,
                        "frames": detections}
    res["input"] = str(p)
    res["type"] = "video" if is_video else "image"
    if args.per_frame:
        frame_results = []
        for crop_i, frame_index in enumerate(used_indices):
            fidx = int(fp[crop_i].argmax())
            fconf = float(fp[crop_i, fidx])
            ffruit = fck["classes"][fidx] if fconf >= args.accept_conf else None

            frame_ripeness = None
            if rp is not None and ffruit is not None and ffruit in rck.get("ripeness_fruits", []):
                if rck["label_space"] == "native":
                    mask = np.array([c.startswith(ffruit + "/") for c in rck["classes"]])
                    frame_rp = rp[crop_i, mask]
                    frame_rp = frame_rp / max(float(frame_rp.sum()), 1e-9)
                    frame_rnames = [c for c, m in zip(rck["classes"], mask) if m]
                else:
                    frame_rp = rp[crop_i]
                    frame_rnames = rck["classes"]
                frame_ripeness = frame_rnames[int(frame_rp.argmax())]

            frame_quality = None
            if qp is not None and ffruit is not None and ffruit in qck.get("quality_fruits", []):
                frame_quality = qck["classes"][int(qp[crop_i].argmax())]

            detection = detections[frame_index] if detector else {"detected": True}
            frame_results.append(dict(i=frame_index, detection=detection, fruit=ffruit, fruit_conf=fconf,
                                      ripeness=frame_ripeness, quality=frame_quality))
        res["per_frame"] = frame_results
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
