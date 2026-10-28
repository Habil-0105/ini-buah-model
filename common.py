"""Shared helpers: config, seeding, device, image loading, transforms."""
import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HERE = Path(__file__).resolve().parent


def load_config(path=None):
    with open(path or HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def load_label_map():
    with open(HERE / "label_map.yaml") as f:
        return yaml.safe_load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref="auto"):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb(path, min_side=0):
    """Open an image as RGB with EXIF orientation applied. JPEG draft mode speeds up big photos."""
    im = Image.open(path)
    if min_side and im.format == "JPEG":
        try:
            im.draft("RGB", (min_side, min_side))
        except Exception:
            pass
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def train_transform(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T
    # No hue jitter: colour is the main ripeness cue.
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T
    # Resize shorter side + centre crop = Vision's .centerCrop on iOS.
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
