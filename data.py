from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pandas as pd
from datasets import load_dataset


@dataclass
class MensaSplit:
    name: str
    scenes: List[str]
    labels: List[int]
    movie_ids: List[int]
    scene_indices: List[int]


def load_mensa_split(split: str = "train") -> MensaSplit:
    """Load MENSA from Hugging Face and flatten to scene-level rows.

    Expects the MENSA dataset schema as used in the existing repo loader:
    item["scenes"] -> list[str]
    item["labels"] -> list[int]
    item["name"] or item["title"] (optional)
    """
    # The existing codebase uses this identifier successfully
    ds = load_dataset("rohitsaxena/MENSA", split=split)
    all_scenes: List[str] = []
    all_labels: List[int] = []
    movie_index: List[int] = []
    scene_indices: List[int] = []
    for movie_idx in range(len(ds)):
        item = ds[movie_idx]
        scenes = item.get("scenes") or item.get("script_scenes", [])
        labels = item.get("labels", [0] * len(scenes))
        for i, sc in enumerate(scenes):
            all_scenes.append(sc)
            all_labels.append(int(labels[i]))
            movie_index.append(movie_idx)
            scene_indices.append(i)

    return MensaSplit(name=split, scenes=all_scenes, labels=all_labels, movie_ids=movie_index, scene_indices=scene_indices)


def load_mensa_dataframe(split: str = "train") -> pd.DataFrame:
    ms = load_mensa_split(split)
    return pd.DataFrame({
        "scene_text": ms.scenes,
        "label": ms.labels,
        "movie_id": ms.movie_ids,
        "scene_index": ms.scene_indices,
    })


