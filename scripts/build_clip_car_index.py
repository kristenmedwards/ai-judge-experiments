"""Build the compact per-car CLIP index used by the ``rag_clip`` context strategy.

Joins the raw pooled embeddings to the 2000 uploaded cars and writes
``data/car_clip_embeddings.npz`` (car_ids "car_N.png" + L2-normalizable vectors).

Inputs
------
--embeddings   clip_embeddings.npy         (18053 x 768, row order == clip_dataset_map.csv)
--map          clip_dataset_map.csv        (aligned rows; needs an image_path and/or car_name col)
--car-mapping  car_name_mapping.csv        (car_id, upload_filename, original_filename, actual_car_name)
--out          data/car_clip_embeddings.npz

Join logic (robust, in priority order):
  1. basename(image_path in map)  ==  original_filename in car_name_mapping   [exact image]
  2. fallback: car_name (map) == actual_car_name (mapping), first matching row  [car-level]

Run once (paths are wherever your embeddings were generated — likely alongside
clip_embeddings.npy / on OneDrive):

    python scripts/build_clip_car_index.py \
      --embeddings data/clip_embeddings.npy \
      --map <path>/clip_dataset_map.csv \
      --car-mapping ../selected_2000_isometric_upload_chunks_renamed/car_name_mapping.csv \
      --out data/car_clip_embeddings.npz
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def _basename(path: str) -> str:
    if not isinstance(path, str):
        return ""
    return re.split(r"[\\/]", path.strip())[-1].strip().lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build per-car CLIP index (car_N.png -> vector).")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--map", required=True, help="clip_dataset_map.csv aligned to the embeddings")
    ap.add_argument("--car-mapping", required=True, help="car_name_mapping.csv")
    ap.add_argument("--out", default="data/car_clip_embeddings.npz")
    a = ap.parse_args(argv)

    emb = np.load(a.embeddings).astype("float32")
    dmap = pd.read_csv(a.map)
    cmap = pd.read_csv(a.car_mapping)
    if len(dmap) != len(emb):
        print(f"!! map rows ({len(dmap)}) != embeddings rows ({len(emb)}); "
              "these must be aligned.", file=sys.stderr)
        return 2

    # index rows of the pooled embeddings
    path_col = "image_path" if "image_path" in dmap.columns else (
        "output_path" if "output_path" in dmap.columns else None)
    base_to_row = {}
    if path_col:
        for i, p in enumerate(dmap[path_col].tolist()):
            base_to_row.setdefault(_basename(p), i)
    name_to_row = {}
    if "car_name" in dmap.columns:
        for i, nm in enumerate(dmap["car_name"].astype(str).tolist()):
            name_to_row.setdefault(nm.strip().lower(), i)

    car_ids, vectors = [], []
    n_img, n_name, n_miss = 0, 0, 0
    for _, r in cmap.iterrows():
        upload = str(r.get("upload_filename") or "").strip()  # car_N.png
        if not upload:
            cid = str(r.get("car_id") or "").strip()
            upload = f"{cid}.png" if cid else ""
        if not upload:
            continue
        row = base_to_row.get(_basename(str(r.get("original_filename") or "")))
        if row is not None:
            n_img += 1
        else:
            row = name_to_row.get(str(r.get("actual_car_name") or "").strip().lower())
            if row is not None:
                n_name += 1
        if row is None:
            n_miss += 1
            continue
        car_ids.append(upload)
        vectors.append(emb[row])

    if not car_ids:
        print("!! no cars matched; check that image_path basenames match "
              "original_filename in car_name_mapping.csv.", file=sys.stderr)
        return 3

    vectors = np.vstack(vectors).astype("float32")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, car_ids=np.array(car_ids), vectors=vectors)
    print(f"[build] matched {len(car_ids)}/{len(cmap)} cars "
          f"(by image={n_img}, by car_name={n_name}, missing={n_miss}); dim={vectors.shape[1]}")
    print(f"[build] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
