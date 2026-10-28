#!/usr/bin/env python
"""Train + evaluate one classifier (task: fruit | ripeness | quality) from artifacts/splits.csv.

  python train.py --task fruit
  python train.py --task ripeness
  python train.py --task quality
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset, Sampler

from common import (IMAGENET_MEAN, IMAGENET_STD, ensure_dir, eval_transform, load_config, load_rgb, pick_device,
                    load_label_map, seed_everything, train_transform, write_json)
from models import ARCHS, backbone_head, build_model


class ImageDS(Dataset):
    def __init__(self, paths, targets, tf, size):
        self.paths, self.targets, self.tf, self.size = list(paths), list(targets), tf, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_rgb(self.paths[i], min_side=self.size * 2)), self.targets[i]


class CappedSampler(Sampler):
    """Each epoch draws up to `cap` random images per class (different draw every epoch)."""

    def __init__(self, targets, cap, seed):
        t = np.asarray(targets)
        self.by_class = {c: np.where(t == c)[0] for c in np.unique(t)}
        self.cap, self.seed, self.epoch = cap, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return sum(min(len(v), self.cap) if self.cap else len(v) for v in self.by_class.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = []
        for v in self.by_class.values():
            idx.extend(v if (not self.cap or len(v) <= self.cap) else rng.choice(v, self.cap, replace=False))
        rng.shuffle(idx)
        return iter(int(i) for i in idx)


def gen_worker(_):
    """Seed numpy in each data-loading worker. Must be top-level so it can be pickled on macOS (spawn)."""
    np.random.seed(torch.initial_seed() % 2**32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1).float().cpu().numpy()
        probs.append(p)
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(probs)


def metrics_for(y, probs, classes):
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    rep = classification_report(y, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    return dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        per_class={c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
                   for c in classes},
    ), confusion_matrix(y, pred, labels=labels), pred


def save_confusion(cm, classes, path_png, path_csv, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(path_csv)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    s = max(5, 0.55 * len(classes))
    fig, ax = plt.subplots(figsize=(s + 2, s + 1))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["fruit", "ripeness", "quality"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--early-stop-patience", type=int,
                    help="epochs without validation macro-F1 improvement after warmup (default from config.yaml)")
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--ripeness-labels", choices=["native", "normalized"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default=None, help="run directory (default runs/<task>_<arch>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = dict(cfg["train"])
    for k in ("arch", "epochs", "workers", "device", "batch_size"):
        if getattr(args, k) is not None:
            tc[k] = getattr(args, k)
    if args.img_size:
        tc["img_size"] = args.img_size
    if args.early_stop_patience is not None:
        if args.early_stop_patience < 1:
            ap.error("--early-stop-patience must be at least 1")
        tc["early_stop_patience"] = args.early_stop_patience
    if args.ripeness_labels:
        tc["ripeness_labels"] = args.ripeness_labels
    if args.no_pretrained:
        tc["pretrained"] = False

    art = Path(cfg["prepare"]["artifacts_dir"])
    splits_path = Path(args.splits or art / "splits.csv")
    meta = json.loads((art / "split_meta.json").read_text())
    root = Path(args.data_root or meta["data_root"])
    df = pd.read_csv(splits_path, dtype=str).fillna("")
    fruit_types = set(load_label_map()["fruit_types"])

    if args.task == "fruit":
        label_col, label_space = "fruit", "fruit"
        # Filter old split manifests too: an earlier manifest may still contain
        # fruit classes that are no longer part of the configured project.
        df = df[df["fruit"].isin(fruit_types)].reset_index(drop=True)
    elif args.task == "quality":
        label_col, label_space = "dragon_quality", "quality"
    else:
        # Ripeness labels are fruit-prefixed in the native label space. Filter
        # by the fruit column so stale splits cannot reintroduce Banana labels.
        df = df[df["fruit"].isin(fruit_types)].reset_index(drop=True)
        if tc["ripeness_labels"] == "native":
            label_col, label_space = "ripeness_native", "native"
        else:
            label_col, label_space = "ripeness_norm", "normalized"
    df = df[df[label_col] != ""].reset_index(drop=True)
    classes = sorted(df[label_col].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"task={args.task} label_col={label_col} classes={len(classes)} images={len(df)}")

    seed_everything(cfg["seed"])
    device = pick_device(tc["device"])
    S = tc["img_size"]
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, p in parts.items():
        missing = [c for c in classes if c not in set(p[label_col])]
        if missing:
            raise SystemExit(f"ERROR: {s} split has no samples for {missing}. The split is unusable; "
                             f"re-run prepare_splits.py and read artifacts/data_report.md (largest groups).")
    tgt = {s: [cidx[x] for x in p[label_col]] for s, p in parts.items()}
    paths = {s: [str(root / x) for x in p["path"]] for s, p in parts.items()}

    nw = tc["workers"]
    common_dl = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                     worker_init_fn=gen_worker)
    sampler = CappedSampler(tgt["train"], tc["cap_per_class_per_epoch"], cfg["seed"])
    train_dl = DataLoader(ImageDS(paths["train"], tgt["train"], train_transform(S), S), batch_size=tc["batch_size"],
                          sampler=sampler, drop_last=len(sampler) > tc["batch_size"], **common_dl)
    val_dl = DataLoader(ImageDS(paths["val"], tgt["val"], eval_transform(S), S), batch_size=tc["batch_size"],
                        shuffle=False, **common_dl)
    test_dl = DataLoader(ImageDS(paths["test"], tgt["test"], eval_transform(S), S), batch_size=tc["batch_size"],
                         shuffle=False, **common_dl)

    # class weights from the effective (capped) per-epoch counts
    cnt = np.bincount(tgt["train"], minlength=len(classes)).astype(float)
    eff = np.minimum(cnt, tc["cap_per_class_per_epoch"] or np.inf)
    w = (eff[eff > 0].mean() / np.maximum(eff, 1)) ** 0.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                    label_smoothing=tc["label_smoothing"])

    try:
        model = build_model(tc["arch"], len(classes), tc["pretrained"]).to(device)
    except Exception as e:
        if tc["pretrained"] and ("urlopen" in repr(e) or "URLError" in repr(e) or "HTTP" in repr(e)):
            raise SystemExit(f"Could not download the ImageNet weights ({e}).\n"
                             "Fix the network/DNS, or download the .pth file manually and put it in "
                             "~/.cache/torch/hub/checkpoints/ (see README). --no-pretrained trains from scratch and "
                             "will be much worse.")
        raise
    feats, head = backbone_head(model)
    E, W = tc["epochs"], tc["warmup_epochs"]

    def make_opt(frozen):
        if frozen:
            return torch.optim.AdamW(head.parameters(), lr=tc["lr_head"], weight_decay=tc["weight_decay"])
        return torch.optim.AdamW([{"params": feats.parameters(), "lr": tc["lr_backbone"]},
                                  {"params": head.parameters(), "lr": tc["lr_head"]}],
                                 weight_decay=tc["weight_decay"])

    run = ensure_dir(args.out or Path(tc["runs_dir"]) / f"{args.task}_{tc['arch']}")
    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    opt, sched = None, None
    for epoch in range(1, E + 1):
        frozen = epoch <= W
        if epoch == 1 or epoch == W + 1:
            for p in feats.parameters():
                p.requires_grad_(not frozen)
            opt = make_opt(frozen)
            sched = None if frozen else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, E - W))
        model.train()
        if frozen:
            feats.eval()
        sampler.set_epoch(epoch)
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if sched:
            sched.step()
        yv, pv = predict(model, val_dl, device)
        mv, _, _ = metrics_for(yv, pv, classes)
        history.append(dict(epoch=epoch, train_loss=tot / max(nb, 1), val_acc=mv["accuracy"],
                            val_macro_f1=mv["macro_f1"], frozen_backbone=frozen, seconds=round(time.time() - t0, 1)))
        print(history[-1])
        if mv["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = mv["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif not frozen:
            bad += 1
            if bad >= tc["early_stop_patience"]:
                print(f"early stopping at epoch {epoch} (best {best_epoch}, val macro-F1 {best_f1:.4f})")
                break
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)

    model.load_state_dict(best_state)
    ripeness_fruits = (sorted({c.split("/")[0] for c in classes})
                       if args.task == "ripeness" and label_space == "native" else
                       sorted(df["fruit"].unique()) if args.task == "ripeness" else [])
    quality_fruits = sorted(df["fruit"].unique()) if args.task == "quality" else []
    torch.save(dict(arch=tc["arch"], num_classes=len(classes), classes=classes, img_size=S, mean=IMAGENET_MEAN,
                    std=IMAGENET_STD, task=args.task, label_col=label_col, label_space=label_space,
                    ripeness_fruits=ripeness_fruits, quality_fruits=quality_fruits,
                    best_epoch=best_epoch, state_dict=best_state),
               run / "best.pth")
    write_json(run / "labels.json", dict(task=args.task, label_space=label_space, classes=classes,
                                         ripeness_fruits=ripeness_fruits, quality_fruits=quality_fruits))

    result = dict(task=args.task, arch=tc["arch"], label_space=label_space, best_epoch=best_epoch,
                  device=str(device), seed=cfg["seed"], config=tc, classes=classes)
    for s, dl in (("val", val_dl), ("test", test_dl)):
        y, p = predict(model, dl, device)
        m, cm, pred = metrics_for(y, p, classes)
        if args.task == "ripeness" and label_space == "native":
            byf = {}
            for f in ripeness_fruits:
                sel = np.array([classes[t].startswith(f + "/") for t in y])
                if sel.any():
                    byf[f] = dict(n=int(sel.sum()), accuracy=float((pred[sel] == y[sel]).mean()))
            m["accuracy_by_fruit"] = byf
        result[s] = m
        save_confusion(cm, classes, run / f"confusion_{s}.png", run / f"confusion_{s}.csv",
                       f"{args.task} {s} (best epoch {best_epoch})")
        pd.DataFrame(dict(path=parts[s]["path"], true=[classes[i] for i in y], pred=[classes[i] for i in pred],
                          confidence=p.max(1))).to_csv(run / f"predictions_{s}.csv", index=False)
        print(f"[{s}] n={m['n']} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    result["caveats"] = [
        "Splits are group-aware only where identifiers exist (avocado fruit_id, papaya frame blocks, pineapple/sawo source groups).",
        "Mango/DragonFruit have no sample IDs: same-fruit photos may straddle splits, so scores are optimistic.",
        "Papaya classes each come from a single video: papaya scores reflect scene identity as much as ripeness.",
        "DragonFruit quality predicts Defect/Fresh from the supplied quality dataset; it is separate from maturity and needs independent validation.",
        "Fruit-type data comes from different sources per fruit: the model may exploit background/camera cues.",
        "Not production-ready evidence: validate on your own held-out phone photos/videos.",
    ]
    write_json(run / "metrics.json", result)
    print("saved ->", run)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # same behaviour on macOS (default) and Linux
    main()
