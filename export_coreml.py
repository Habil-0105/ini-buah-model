#!/usr/bin/env python
"""Export a trained checkpoint to Core ML (.mlpackage).

  python export_coreml.py --ckpt runs/fruit_mobilenet_v3_large/best.pth
Input: RGB image (named "image"), softmax + class labels are baked in. Normalisation happens inside the model.
Full verification (running the Core ML model) only works on macOS.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import eval_transform, load_rgb, write_json
from models import build_model


class Wrapped(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x in [0, 1]
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output .mlpackage path (default: next to the checkpoint)")
    ap.add_argument("--fp32", action="store_true", help="keep float32 weights (default float16)")
    ap.add_argument("--verify-image", default=None, help="image used to compare PyTorch vs Core ML (macOS only)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    ckpt_path = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt_path.with_name(f"{ck['task']}_classifier.mlpackage")
    report = dict(ckpt=str(ckpt_path), out=str(out), ok=False)
    try:
        import coremltools as ct
        S = ck["img_size"]
        base = build_model(ck["arch"], ck["num_classes"], pretrained=False)
        base.load_state_dict(ck["state_dict"])
        net = Wrapped(base, ck["mean"], ck["std"]).eval()
        example = torch.rand(1, 3, S, S)
        traced = torch.jit.trace(net, example)
        ml = ct.convert(
            traced,
            inputs=[ct.ImageType(name="image", shape=(1, 3, S, S), scale=1 / 255.0,
                                 color_layout=ct.colorlayout.RGB)],
            classifier_config=ct.ClassifierConfig(list(ck["classes"])),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
        )
        ml.short_description = f"Fruit {ck['task']} classifier ({ck['arch']}, {S}px)"
        ml.user_defined_metadata["classes"] = json.dumps(ck["classes"])
        ml.user_defined_metadata["label_space"] = ck["label_space"]
        ml.user_defined_metadata["ripeness_fruits"] = json.dumps(ck.get("ripeness_fruits", []))
        ml.save(str(out))
        report.update(ok=True, converted=True)
        print("saved", out)

        if args.verify_image:
            if sys.platform != "darwin":
                report["verify"] = "skipped: Core ML prediction requires macOS"
            else:
                img = load_rgb(args.verify_image, S * 2)
                with torch.no_grad():
                    p_t = net(tfm_to_01(img, S)).numpy()[0]
                pred = ml.predict({"image": center_crop(img, S)})
                probs = pred[[k for k in pred if k != "classLabel"][0]]
                p_c = np.array([probs[c] for c in ck["classes"]])
                report["verify"] = dict(torch_top1=ck["classes"][int(p_t.argmax())], coreml_top1=pred["classLabel"],
                                        max_abs_prob_diff=float(np.abs(p_t - p_c).max()))
        else:
            report.setdefault("verify", "not requested (--verify-image)")
    except Exception as e:  # document failures instead of hiding them
        report.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print("Core ML export FAILED:", e, file=sys.stderr)
    write_json(out.with_suffix(".export_report.json"), report)
    sys.exit(0 if report["ok"] else 1)


def center_crop(img, S):
    from PIL import Image
    w, h = img.size
    s = S / min(w, h)
    img = img.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - S) // 2, (h - S) // 2
    return img.crop((l, t, l + S, t + S))


def tfm_to_01(img, S):
    arr = np.asarray(center_crop(img, S), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


if __name__ == "__main__":
    main()
