"""Per-car CLIP embedding index for RAG context selection.

The raw ``clip_embeddings.npy`` is (18053, 768) over the FULL candidate isometric
pool, in the row order of ``clip_dataset_map.csv``. ``scripts/build_clip_car_index.py``
joins that to the 2000 uploaded cars (via ``car_name_mapping.csv``) and writes a
compact ``data/car_clip_embeddings.npz`` holding only the cars we rate:

    car_ids : array of "car_N.png"  (matches the ratings/image keys)
    vectors : float32 [N, D], L2-normalized

This module loads that compact file and answers nearest-neighbour queries for the
``rag_clip`` context strategy. If the file is absent, ``load_default()`` returns
None and callers fall back to the lightweight pixel feature.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

DEFAULT_PATH = os.environ.get("CLIP_CAR_INDEX", "data/car_clip_embeddings.npz")


class CarClipIndex:
    def __init__(self, car_ids: List[str], vectors: np.ndarray):
        # store normalized vectors for cosine == dot product
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = v / norms
        self.car_ids = [str(c) for c in car_ids]
        self._row = {c: i for i, c in enumerate(self.car_ids)}

    @classmethod
    def load(cls, path: str) -> "CarClipIndex":
        data = np.load(path, allow_pickle=True)
        return cls(list(data["car_ids"]), data["vectors"])

    def has(self, car_png: str) -> bool:
        return car_png in self._row

    def vec(self, car_png: str) -> Optional[np.ndarray]:
        i = self._row.get(car_png)
        return None if i is None else self.vectors[i]

    def nearest(self, target: str, pool: List[str], k: int) -> List[str]:
        """Return up to k pool cars most similar to target (cosine desc). Pool cars
        without an embedding are dropped; if the target has no embedding, returns
        an empty list so the caller can fall back."""
        t = self.vec(target)
        if t is None:
            return []
        cand = [c for c in pool if c in self._row]
        if not cand:
            return []
        rows = np.array([self._row[c] for c in cand])
        scores = self.vectors[rows] @ t
        order = np.argsort(-scores)[:k]
        return [cand[i] for i in order]

    def coverage(self, cars: List[str]) -> float:
        if not cars:
            return 0.0
        return sum(c in self._row for c in cars) / len(cars)


@lru_cache(maxsize=4)
def load_default(path: str = DEFAULT_PATH) -> Optional[CarClipIndex]:
    """Load the compact per-car index if it exists, else None (cached)."""
    if not path or not os.path.exists(path):
        return None
    try:
        return CarClipIndex.load(path)
    except Exception:
        return None
