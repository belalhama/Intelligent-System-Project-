"""
MNIST Downloader, IDX Parser, and Reproducible Preprocessor (Digits 1-9).

Follows P7 specification:
- Downloads the 4 original MNIST gzip files (~11 MiB total) and keeps a local copy in data/raw/.
- Splits the 60,000 training set into 50,000 train / 10,000 validation with a fixed seed;
  keeps the 10,000 test set untouched.
- Drops digit 0 (keeps digits 1..9).
- Scales pixels to [0.0, 1.0].
- Provides the brightness rule for blank cell detection (not using the NN).
"""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MNIST_FILES: Dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

# Simple brightness rule threshold: blank cells have mean pixel intensity < 0.025
# whereas real handwritten digits (1-9) have mean intensity ~0.06 to 0.35.
BRIGHTNESS_THRESHOLD: float = 0.025


@dataclass
class MNISTSubset:
    """Processed MNIST dataset filtered to digits 1..9 with pixels scaled to [0, 1]."""

    X_train: np.ndarray  # (N_train, 28, 28) float32 in [0, 1]
    y_train: np.ndarray  # (N_train,) int64 in 1..9
    X_val: np.ndarray    # (N_val, 28, 28) float32 in [0, 1]
    y_val: np.ndarray    # (N_val,) int64 in 1..9
    X_test: np.ndarray   # (N_test, 28, 28) float32 in [0, 1]
    y_test: np.ndarray   # (N_test,) int64 in 1..9
    seed: int
    raw_counts: Dict[str, int]


def detect_blank_cell(
    cell_image: np.ndarray,
    threshold: float = BRIGHTNESS_THRESHOLD,
) -> bool:
    """
    Detect whether a 28x28 Sudoku cell image is blank using a simple brightness rule
    (mean pixel intensity < threshold), as required by P7 (not using the NN).
    """
    return float(np.mean(cell_image)) < threshold


def _download_file(filename: str, raw_dir: Path) -> Path:
    """Download a single MNIST gzip file from available mirrors if not already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    last_err = None
    for base_url in MNIST_MIRRORS:
        url = base_url + filename
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (SudokuCSPProject/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            dest.write_bytes(data)
            return dest
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Failed to download {filename} from all mirrors: {last_err}")


def _read_idx_images(gz_path: Path) -> np.ndarray:
    """Parse an IDX3 gzip file into a (N, 28, 28) uint8 numpy array."""
    with gzip.open(gz_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {gz_path}")
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def _read_idx_labels(gz_path: Path) -> np.ndarray:
    """Parse an IDX1 gzip file into a (N,) uint8 numpy array."""
    with gzip.open(gz_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {gz_path}")
        buf = f.read(num_labels)
        return np.frombuffer(buf, dtype=np.uint8)


def download_and_preprocess_mnist(
    data_dir: Path = Path("data"),
    seed: int = 42,
) -> MNISTSubset:
    """
    Download raw MNIST files, split 60,000 training images into 50,000 train / 10,000 val
    using `seed`, drop digit 0 (keep 1..9), scale pixels to [0, 1], and save to disk.
    """
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    paths = {k: _download_file(fname, raw_dir) for k, fname in MNIST_FILES.items()}

    raw_train_x = _read_idx_images(paths["train_images"])
    raw_train_y = _read_idx_labels(paths["train_labels"])
    raw_test_x = _read_idx_images(paths["test_images"])
    raw_test_y = _read_idx_labels(paths["test_labels"])

    assert len(raw_train_x) == 60_000 and len(raw_train_y) == 60_000
    assert len(raw_test_x) == 10_000 and len(raw_test_y) == 10_000

    # Split 60,000 training set into 50,000 train and 10,000 validation reproducibly
    rng = np.random.default_rng(seed)
    perm = rng.permutation(60_000)
    train_idx = perm[:50_000]
    val_idx = perm[50_000:]

    x_tr_full, y_tr_full = raw_train_x[train_idx], raw_train_y[train_idx]
    x_va_full, y_va_full = raw_train_x[val_idx], raw_train_y[val_idx]

    # Filter out digit 0 (keep digits 1..9) and scale pixels to [0.0, 1.0]
    tr_mask = y_tr_full != 0
    va_mask = y_va_full != 0
    te_mask = raw_test_y != 0

    X_train = (x_tr_full[tr_mask].astype(np.float32)) / 255.0
    y_train = y_tr_full[tr_mask].astype(np.int64)

    X_val = (x_va_full[va_mask].astype(np.float32)) / 255.0
    y_val = y_va_full[va_mask].astype(np.int64)

    X_test = (raw_test_x[te_mask].astype(np.float32)) / 255.0
    y_test = raw_test_y[te_mask].astype(np.int64)

    raw_counts = {
        "raw_train_total": 60_000,
        "raw_train_split": 50_000,
        "raw_val_split": 10_000,
        "raw_test_total": 10_000,
        "filtered_train_1to9": int(len(y_train)),
        "filtered_val_1to9": int(len(y_val)),
        "filtered_test_1to9": int(len(y_test)),
    }

    npz_path = proc_dir / "mnist_1to9.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        seed=np.array([seed], dtype=np.int64),
    )

    summary_path = proc_dir / "dataset_summary.json"
    summary_data = {
        "seed": seed,
        "brightness_threshold": BRIGHTNESS_THRESHOLD,
        "counts": raw_counts,
        "train_class_distribution": {
            str(d): int(np.sum(y_train == d)) for d in range(1, 10)
        },
        "val_class_distribution": {
            str(d): int(np.sum(y_val == d)) for d in range(1, 10)
        },
        "test_class_distribution": {
            str(d): int(np.sum(y_test == d)) for d in range(1, 10)
        },
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    return MNISTSubset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        raw_counts=raw_counts,
    )


def load_processed_mnist(
    data_dir: Path = Path("data"),
    seed: int = 42,
) -> MNISTSubset:
    """Load preprocessed MNIST (1..9) from disk, or run download_and_preprocess_mnist if missing."""
    npz_path = data_dir / "processed" / "mnist_1to9.npz"
    summary_path = data_dir / "processed" / "dataset_summary.json"
    if not npz_path.exists():
        return download_and_preprocess_mnist(data_dir=data_dir, seed=seed)

    loaded = np.load(npz_path)
    raw_counts = {}
    if summary_path.exists():
        raw_counts = json.loads(summary_path.read_text(encoding="utf-8")).get("counts", {})

    return MNISTSubset(
        X_train=loaded["X_train"],
        y_train=loaded["y_train"],
        X_val=loaded["X_val"],
        y_val=loaded["y_val"],
        X_test=loaded["X_test"],
        y_test=loaded["y_test"],
        seed=int(loaded["seed"][0]),
        raw_counts=raw_counts,
    )

