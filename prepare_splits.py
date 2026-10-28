#!/usr/bin/env python
"""Scan the dataset root (read-only), dedupe, build leakage-aware groups, and write one split manifest.

Outputs (in artifacts/): splits.csv, split_meta.json, data_report.md, hash_cache.csv
Original dataset files are only ever read.
"""
import argparse
import hashlib
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common import IMG_EXT, ensure_dir, load_config, load_label_map, write_json

PAPAYA_RE = re.compile(r"^(?P<video>[A-Za-z]+)_f(?P<frame>\d+)_aug(?P<aug>\d+)_b(?P<box>\d+)$")
AVOCADO_FID_RE = re.compile(r"^([A-Za-z0-9]+_\d+)__")


# ----------------------------------------------------------------------------- scanning
def list_images(d: Path):
    out = []
    for p in d.rglob("*"):
        if (p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith(".")
                and "__MACOSX" not in p.parts):
            out.append(p)
    return sorted(out)


def collect(root: Path, cfg, lm):
    P, pr = cfg["paths"], cfg["prepare"]
    recs, missing, notes = [], [], {}

    def add(path, **kw):
        base = dict(abs=str(path), rel=path.relative_to(root).as_posix(), fruit="", ripeness_native="",
                    dragon_maturity="", dragon_quality="", group_key="", collection="")
        base.update(kw)
        recs.append(base)

    def need(key):
        d = root / P[key]
        if not d.is_dir():
            missing.append(f"{key}: {d}")
            return None
        return d

    # Dragon fruit
    d = need("dragon_maturity")
    if d:
        for p in list_images(d):
            add(p, fruit="dragon_fruit", dragon_maturity=p.parent.name, collection="dragon_maturity")
    d = need("dragon_quality")
    if d:
        for p in list_images(d):
            add(p, fruit="dragon_fruit", dragon_quality=p.parent.name, collection="dragon_quality")
    # Mango
    d = need("mango")
    if d:
        # Only scan the top-level native ripeness folders. The directory also
        # contains Mango_Dataset, a separate YOLO detection dataset whose images
        # must not enter the classification split.
        mango_labels = set(lm.get("ripeness_native", {}).get("mango", []))
        for cd in sorted(x for x in d.iterdir() if x.is_dir() and x.name in mango_labels):
            for p in list_images(cd):
                add(p, fruit="mango", ripeness_native=f"mango/{cd.name}", collection="mango")
    # Avocado (derived layout; group = fruit_id so all photos of one fruit stay together)
    d = need("avocado")
    if d:
        fid_map = {}
        csvp = d / "labels_matched.csv"
        if csvp.exists():
            df = pd.read_csv(csvp, dtype=str).fillna("")
            for r in df.itertuples():
                fid_map[Path(str(r.dst).replace("\\", "/")).name] = r.fruit_id
        n_csv = n_re = n_none = 0
        for split in ("Training", "Testing"):
            sd = d / split
            if not sd.is_dir():
                continue
            for cd in sorted(x for x in sd.iterdir() if x.is_dir()):
                for p in list_images(cd):
                    fid = fid_map.get(p.name)
                    if fid:
                        n_csv += 1
                    else:
                        m = AVOCADO_FID_RE.match(p.stem)
                        fid = m.group(1) if m else None
                        n_re += bool(m)
                        n_none += not m
                    gk = f"avocado|{fid}" if fid else f"avocado|{p.stem}"
                    add(p, fruit="avocado", ripeness_native=f"avocado/{cd.name}", group_key=gk,
                        collection="avocado")
        notes["avocado_fruit_id_source"] = f"csv={n_csv}, filename_regex={n_re}, none={n_none}"
    # Papaya crops (group = video + frame block; class == video per the handoff, so true video split is impossible)
    d = need("papaya_crops")
    if d:
        n_re = n_none = 0
        for split in ("Training", "Testing"):
            sd = d / split
            if not sd.is_dir():
                continue
            for cd in sorted(x for x in sd.iterdir() if x.is_dir()):
                label = re.sub(r"^\d+_", "", cd.name)
                for p in list_images(cd):
                    m = PAPAYA_RE.match(p.stem)
                    if m:
                        n_re += 1
                        blk = int(m.group("frame")) // pr["papaya_frame_block"]
                        gk = f"papaya|{m.group('video')}|{blk}"
                    else:
                        n_none += 1
                        gk = f"papaya|{p.stem}"
                    add(p, fruit="papaya", ripeness_native=f"papaya/{label}", group_key=gk, collection="papaya")
        notes["papaya_filename_parsed"] = f"parsed={n_re}, unparsed={n_none}"
    # Pineapple crops
    d = need("pineapple_crops")
    if d:
        for split in ("Training", "Validation", "Testing"):
            sd = d / split
            if not sd.is_dir():
                continue
            for cd in sorted(x for x in sd.iterdir() if x.is_dir()):
                label = re.sub(r"^\d+_", "", cd.name)
                for p in list_images(cd):
                    stem = p.stem
                    if stem.startswith("Pineapple"):
                        m = re.match(r"^Pineapple(\d+)", stem)
                        blk = int(m.group(1)) // 30 if m else 0
                        gk = f"pineapple|video|{blk}"
                    else:
                        orig = stem.split("_")[0]
                        gk = f"pineapple|img|{orig}"
                    add(p, fruit="pineapple", ripeness_native=f"pineapple/{label}", group_key=gk, collection="pineapple")
    # Sawo crops
    d = need("sawo_crops")
    if d:
        for split in ("Training", "Testing"):
            sd = d / split
            if not sd.is_dir():
                continue
            for cd in sorted(x for x in sd.iterdir() if x.is_dir()):
                label = re.sub(r"^\d+_", "", cd.name)
                for p in list_images(cd):
                    parts = p.stem.rsplit("_", 2)
                    orig_stem = parts[0] if len(parts) >= 3 else p.stem
                    gk = f"sawo|{orig_stem}"
                    add(p, fruit="sawo", ripeness_native=f"sawo/{label}", group_key=gk, collection="sawo")
    return recs, missing, notes


# ----------------------------------------------------------------------------- hashing
def hash_one(abs_path):
    try:
        h = hashlib.md5()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        im = Image.open(abs_path)
        try:
            im.draft("L", (128, 128))
        except Exception:
            pass
        g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return abs_path, h.hexdigest(), str(v), True
    except Exception:
        return abs_path, "", "", False


def compute_hashes(df, cache_path, workers):
    cache = {}
    if cache_path.exists():
        c = pd.read_csv(cache_path, dtype=str).fillna("")
        cache = {r.abs: (r.size, r.mtime, r.md5, r.dhash, r.ok) for r in c.itertuples()}
    todo, rows = [], {}
    for p in df["abs"]:
        st = Path(p).stat()
        key = (str(st.st_size), str(st.st_mtime_ns))
        c = cache.get(p)
        if c and (c[0], c[1]) == key:
            rows[p] = (c[2], c[3], c[4] == "True")
        else:
            todo.append(p)
    if todo:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x
        print(f"hashing {len(todo)} images ({len(rows)} cached) ...")
        if workers > 1:
            with Pool(workers) as pool:
                it = pool.imap_unordered(hash_one, todo, chunksize=32)
                for p, md5, dh, ok in tqdm(it, total=len(todo)):
                    rows[p] = (md5, dh, ok)
        else:
            for p in tqdm(todo):
                _, md5, dh, ok = hash_one(p)
                rows[p] = (md5, dh, ok)
        out = []
        for p, (md5, dh, ok) in rows.items():
            st = Path(p).stat()
            out.append((p, st.st_size, st.st_mtime_ns, md5, dh, ok))
        pd.DataFrame(out, columns=["abs", "size", "mtime", "md5", "dhash", "ok"]).to_csv(cache_path, index=False)
    df = df.copy()
    df["md5"] = df["abs"].map(lambda p: rows[p][0])
    df["dhash"] = df["abs"].map(lambda p: rows[p][1])
    df["ok"] = df["abs"].map(lambda p: rows[p][2])
    return df


# ----------------------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.size = [1] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = min(ra, rb), max(ra, rb)
            self.p[hi] = lo
            self.size[lo] += self.size[hi]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lm = load_label_map()
    pr = cfg["prepare"]
    root = Path(args.data_root or cfg["data_root"]).expanduser().resolve()
    art = ensure_dir(Path(pr["artifacts_dir"]))
    workers = args.workers if args.workers is not None else pr["workers"]

    recs, missing, notes = collect(root, cfg, lm)
    if missing:
        print("ERROR: expected collection folders not found under", root, file=sys.stderr)
        for m in missing:
            print("  -", m, file=sys.stderr)
        print("Fix --data-root or the `paths:` section of config.yaml.", file=sys.stderr)
        sys.exit(1)
    raw = pd.DataFrame(recs)
    print(f"found {len(raw)} image files in {raw['collection'].nunique()} collections")

    raw = compute_hashes(raw, art / "hash_cache.csv", workers)
    unreadable = raw[~raw["ok"]]
    raw = raw[raw["ok"]].reset_index(drop=True)

    # ---- exact-duplicate merge (identical bytes), with conflict handling
    merged, fruit_conflicts, rip_conflicts = [], [], []
    for md5, g in raw.groupby("md5", sort=False):
        if g["fruit"].nunique() > 1:
            fruit_conflicts.append(list(g["rel"]))
            continue
        rips = sorted({x for x in g["ripeness_native"] if x})
        if len(rips) > 1:
            rip_conflicts.append((list(g["rel"]), rips))
        first = g.iloc[0]
        merged.append(dict(
            rel=first["rel"], abs=first["abs"], fruit=first["fruit"], md5=md5, dhash=int(first["dhash"]),
            ripeness_native=rips[0] if len(rips) == 1 else "",
            dragon_maturity=next((x for x in g["dragon_maturity"] if x), ""),
            dragon_quality=next((x for x in g["dragon_quality"] if x), ""),
            collections="|".join(sorted(set(g["collection"]))),
            all_paths="|".join(g["rel"]),
            group_keys=sorted({k for k in g["group_key"] if k}),
        ))
    m = pd.DataFrame(merged)
    n_exact_removed = len(raw) - len(m) - sum(len(c) for c in fruit_conflicts)

    # dragon maturity as ripeness is opt-in
    if pr["include_dragon_maturity_in_ripeness"]:
        sel = (m["fruit"] == "dragon_fruit") & (m["dragon_maturity"] != "") & (m["ripeness_native"] == "")
        m.loc[sel, "ripeness_native"] = "dragon_fruit/" + m.loc[sel, "dragon_maturity"]

    # ---- groups: filename-derived keys + exact dups + near-duplicates
    n = len(m)
    uf = UF(n)
    key_first = {}
    for i, keys in enumerate(m["group_keys"]):
        for k in keys:
            if k in key_first:
                uf.union(i, key_first[k])
            else:
                key_first[k] = i
    near_pairs = cross_fruit = cross_rip = skipped_buckets = capped = 0
    thr = pr["near_dup_hamming"]
    cap = pr["near_dup_max_group"]
    has_key = [bool(k) for k in m["group_keys"]]
    if thr > 0:
        h = m["dhash"].tolist()
        for band in range(4):
            buckets = {}
            for i, v in enumerate(h):
                buckets.setdefault((v >> (16 * band)) & 0xFFFF, []).append(i)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                if len(members) > pr["near_dup_max_bucket"]:
                    skipped_buckets += 1
                    continue
                for ai in range(len(members)):
                    for bi in range(ai + 1, len(members)):
                        a, b = members[ai], members[bi]
                        if bin(h[a] ^ h[b]).count("1") > thr:
                            continue
                        if has_key[a] and has_key[b]:
                            continue  # both have filename/CSV identifiers: trust them, don't chain by appearance
                        ra, rb = uf.find(a), uf.find(b)
                        if ra == rb:
                            continue
                        if uf.size[ra] + uf.size[rb] > cap:
                            capped += 1  # prevents one giant "everything looks alike" group
                            continue
                        near_pairs += 1
                        cross_fruit += m.at[a, "fruit"] != m.at[b, "fruit"]
                        xa, xb = m.at[a, "ripeness_native"], m.at[b, "ripeness_native"]
                        cross_rip += bool(xa and xb and xa != xb)
                        uf.union(a, b)
    m["group_id"] = [f"g{uf.find(i):06d}" for i in range(n)]

    # ---- normalized stage column (name-identical stages only; see label_map.yaml)
    def norm(label):
        if not label or "/" not in label:
            return ""
        fruit, lab = label.split("/", 1)
        return (lm["normalized_stage"].get(fruit, {}) or {}).get(lab) or ""
    m["ripeness_norm"] = m["ripeness_native"].map(norm)

    # ---- group-aware stratified 3-way split
    fr = pr["split_fractions"]
    strat_label = m["ripeness_native"].where(m["ripeness_native"] != "", "")
    quality_label = "quality/" + m["dragon_quality"]
    strat_label = strat_label.where(strat_label != "", quality_label.where(m["dragon_quality"] != "", "-"))
    strat = (m["fruit"] + "|" + strat_label).values
    groups = m["group_id"].values
    idx = np.arange(n)
    k_test = max(2, round(1 / fr["test"]))
    k_val = max(2, round(1 / (fr["val"] / (1 - fr["test"]))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rest, test = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=cfg["seed"]).split(idx, strat, groups))
        tr, va = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=cfg["seed"]).split(
            rest, strat[rest], groups[rest]))
    split = np.empty(n, dtype=object)
    split[test] = "test"
    split[rest[va]] = "val"
    split[rest[tr]] = "train"
    m["split"] = split

    # ---- leakage checks (hard failures)
    assert m.groupby("group_id")["split"].nunique().max() == 1, "a group spans several splits"
    assert m.groupby("md5")["split"].nunique().max() == 1, "an exact duplicate spans several splits"

    out = m[["rel", "fruit", "ripeness_native", "ripeness_norm", "dragon_maturity",
             "dragon_quality", "group_id", "md5", "collections", "all_paths", "split"]].rename(columns={"rel": "path"})

    # ---- every class must appear in train, val AND test, otherwise the split is unusable
    bad = []
    for col in ("fruit", "ripeness_native", "dragon_quality"):
        sub = m[m[col] != ""]
        ct = pd.crosstab(sub[col], sub["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
        for lab, row in ct.iterrows():
            if (row == 0).any():
                bad.append(f"{col}={lab}: " + ", ".join(f"{k}={v}" for k, v in row.items()))
    gsz = m.groupby("group_id").agg(n=("rel", "size"), fruit=("fruit", "first"),
                                    n_fruits=("fruit", "nunique"), example=("rel", "first")
                                    ).sort_values("n", ascending=False)
    top_groups = gsz.head(8)
    out_path = art / ("splits.csv" if not bad else "splits_REJECTED.csv")
    out.to_csv(out_path, index=False)
    if not bad and (art / "splits_REJECTED.csv").exists():
        (art / "splits_REJECTED.csv").unlink()
    write_json(art / "split_meta.json", dict(data_root=str(root), seed=cfg["seed"], n_images=n,
                                            fractions_requested=fr, k_test=k_test, k_val=k_val))

    # ---- report
    def table(df, rows, cols="split"):
        return pd.crosstab(df[rows], df[cols]).reindex(columns=["train", "val", "test"], fill_value=0).to_string()
    L = []
    L.append("# Data report\n")
    L.append(f"Dataset root: `{root}`  |  seed {cfg['seed']}\n")
    L.append("## Scan and dedupe\n```")
    L.append(f"image files found:            {len(recs)}")
    L.append(f"unreadable (skipped):         {len(unreadable)}")
    L.append(f"exact-duplicate files merged: {n_exact_removed}")
    L.append(f"dropped, same bytes under different fruits: {sum(len(c) for c in fruit_conflicts)}")
    L.append(f"conflicting ripeness labels on identical bytes (ripeness blanked): {len(rip_conflicts)}")
    L.append(f"unique images kept:           {n}")
    L.append(f"near-duplicate pairs merged (dHash<= {thr}): {near_pairs}  "
             f"[cross-fruit: {cross_fruit}, cross-ripeness-label: {cross_rip}, skipped big buckets: {skipped_buckets}, "
             f"blocked by group-size cap {cap}: {capped}]")
    for k, v in notes.items():
        L.append(f"{k}: {v}")
    L.append("```\n")
    L.append("## Leakage control\n```")
    L.append(f"groups: {m['group_id'].nunique()} over {n} images; max group size {m['group_id'].value_counts().max()}")
    L.append("largest groups:\n" + top_groups.to_string())
    L.append("checked: no group and no md5 appears in more than one split")
    L.append("```\n")
    L.append("## Fruit-type split\n```\n" + table(m, "fruit") + "\n```\n")
    r = m[m["ripeness_native"] != ""]
    L.append("## Ripeness split (native labels)\n```\n" + table(r, "ripeness_native") + "\n```\n")
    q = m[m["dragon_quality"] != ""]
    L.append("## DragonFruit quality split\n```\n" + table(q, "dragon_quality") + "\n```\n")
    rn = m[m["ripeness_norm"] != ""]
    L.append("## Ripeness split (normalized, name-identical stages only)\n```\n" + table(rn, "ripeness_norm") + "\n```\n")
    L.append("## Known limitations (cannot be fixed with the available metadata)\n")
    L.append("- **Mango, DragonFruit**: no sample/fruit ID exists, so only exact and near-duplicate images are grouped. "
             "Different photos of the same fruit can still land in different splits -> scores for these are optimistic.\n"
             "- **Papaya**: per the handoff each class comes from a single video. Groups are (video, frame block), "
             "so train/val/test are different time blocks of the same videos and class is confounded with scene. "
             "Papaya ripeness scores are optimistic and say little about real-world papayas.\n"
             "- **Fruit-type task**: every fruit comes from a different dataset/camera/background, so the model can "
             "partly learn dataset identity. Test on your own phone photos before trusting it.\n"
             "- No 'not a fruit / other' class exists; use a confidence threshold in the app.\n")
    if fruit_conflicts:
        L.append("## Same bytes under different fruits (dropped)\n" + "\n".join("- " + " | ".join(c) for c in fruit_conflicts[:50]) + "\n")
    if rip_conflicts:
        L.append("## Conflicting ripeness labels (ripeness blanked)\n" +
                 "\n".join(f"- {c[0]} -> {c[1]}" for c in rip_conflicts[:50]) + "\n")
    if len(unreadable):
        L.append("## Unreadable files\n" + "\n".join("- " + p for p in unreadable["rel"].head(50)) + "\n")
    (art / "data_report.md").write_text("\n".join(L))
    print("\n".join(L[:12]))
    print("largest groups:\n" + top_groups.to_string())
    if bad:
        print("\nERROR: the split is UNUSABLE, these classes are missing from at least one split:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        print(f"Written to {out_path} (not splits.csv) so training cannot pick it up. See {art/'data_report.md'}.",
              file=sys.stderr)
        sys.exit(2)
    print(f"\nwrote {out_path} and {art/'data_report.md'}")


if __name__ == "__main__":
    main()
