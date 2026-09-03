import os
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Column name aliases
_IMAGE_COLS = ["image_path", "filepath", "image", "path", "file_name"]
_LAT_COLS   = ["Latitude",   "latitude",  "lat"]
_LON_COLS   = ["Longitude",  "longitude", "lon"]


def _find_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find any of {candidates} in CSV columns: {list(df.columns)}"
    )


def prepare_data(csv_path: str) -> Tuple[List[torch.Tensor], List[List[float]]]:
    df = pd.read_csv(csv_path)

    img_col = _find_col(df, _IMAGE_COLS)
    lat_col = _find_col(df, _LAT_COLS)
    lon_col = _find_col(df, _LON_COLS)

    csv_dir = Path(csv_path).parent

    X: List[torch.Tensor]  = []
    y: List[List[float]]   = []

    for _, row in df.iterrows():
        img_path = str(row[img_col])
        if not os.path.isabs(img_path):
            img_path = str(csv_dir / img_path)

        img = Image.open(img_path).convert("RGB")
        tensor = _transform(img)

        X.append(tensor)
        y.append([float(row[lat_col]), float(row[lon_col])])

    return X, y