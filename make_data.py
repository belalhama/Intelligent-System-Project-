"""
Reproducible Data Download, Preprocessing, and Sudoku Mosaic Generation Script (make_data.py).

Deliverable #2:
- Downloads the 4 original MNIST gzip files and stores a local copy in data/raw/.
- Scales pixel intensities to [0.0, 1.0].
- Splits the 60,000 training set into 50,000 train / 10,000 validation (fixed seed=42),
  leaving the 10,000 test set untouched.
- Drops digit 0 (retains digits 1..9).
- Generates a reproducible benchmark sample of Sudoku puzzles and 9x9 handwritten
  image mosaics with injected hard digits in data/processed/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.mnist_loader import BRIGHTNESS_THRESHOLD, download_and_preprocess_mnist
from data.sudoku_generator import generate_sudoku_mosaic_dataset
from nn.digit_classifier import load_or_train_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess MNIST (digits 1-9) and generate seeded Sudoku mosaics."
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Target data directory (default: data)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seed = args.seed

    print(f"[make_data] Step 1/3: Downloading and preprocessing MNIST (seed={seed})...")
    dataset = download_and_preprocess_mnist(data_dir=data_dir, seed=seed)
    print(
        f"  Raw MNIST split: 50,000 train / 10,000 val / 10,000 test (untouched)\n"
        f"  Filtered (digits 1..9): "
        f"train={len(dataset.y_train):,}, val={len(dataset.y_val):,}, test={len(dataset.y_test):,}\n"
        f"  Pixel range: [{dataset.X_train.min():.2f}, {dataset.X_train.max():.2f}], "
        f"Blank brightness threshold: {BRIGHTNESS_THRESHOLD}"
    )

    print("[make_data] Step 2/3: Training/loading calibrated NN and Logistic Regression baseline...")
    nn_clf, lr_clf, metrics = load_or_train_models(dataset=dataset, seed=seed)
    nn_m = metrics["nn"]
    lr_m = metrics["baseline_logreg"]
    print(
        f"  NN (MLP + TempScaling T={nn_m.temperature:.3f}): "
        f"Test Acc = {nn_m.accuracy * 100:.2f}%, "
        f"ECE (uncal -> cal) = {nn_m.ece_uncalibrated:.4f} -> {nn_m.ece_calibrated:.4f}, "
        f"NLL = {nn_m.nll_calibrated:.4f}"
    )
    print(
        f"  Baseline (Logistic Regression): "
        f"Test Acc = {lr_m.accuracy * 100:.2f}%, "
        f"ECE (uncal -> cal) = {lr_m.ece_uncalibrated:.4f} -> {lr_m.ece_calibrated:.4f}, "
        f"NLL = {lr_m.nll_calibrated:.4f}"
    )

    print("[make_data] Step 3/3: Generating reproducible sample Sudoku 9x9 mosaics...")
    test_probs = nn_clf.predict_proba(dataset.X_test)
    mosaics = generate_sudoku_mosaic_dataset(
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        num_puzzles=10,
        num_givens=30,
        num_injected_misreads=2,
        seed=seed,
        classifier_probs_on_test=test_probs,
        predict_proba_fn=nn_clf.predict_proba,
    )

    proc_dir = data_dir / "processed"
    np.savez_compressed(
        proc_dir / "sample_sudoku_mosaics.npz",
        puzzle_grids=np.array([m.puzzle_grid for m in mosaics], dtype=np.int64),
        solution_grids=np.array([m.solution_grid for m in mosaics], dtype=np.int64),
        cell_images=np.array([m.cell_images for m in mosaics], dtype=np.float32),
    )

    meta_records = []
    for m in mosaics:
        meta_records.append(
            {
                "puzzle_id": m.puzzle_id,
                "seed": m.seed,
                "num_givens": m.num_givens,
                "num_injected_misreads": m.num_injected_misreads,
                "injected_cells": [
                    {
                        "cell": list(ic.cell),
                        "true_digit": ic.true_digit,
                        "intended_read_digit": ic.intended_read_digit,
                        "alpha_blend": ic.alpha_blend,
                        "source_test_idx_true": ic.source_test_idx_true,
                        "source_test_idx_confuser": ic.source_test_idx_confuser,
                    }
                    for ic in m.injected_cells
                ],
            }
        )
    (proc_dir / "sample_puzzles_metadata.json").write_text(
        json.dumps(meta_records, indent=2), encoding="utf-8"
    )
    print(f"[make_data] Done! Saved processed files to {proc_dir.resolve()}.")


if __name__ == "__main__":
    main()

