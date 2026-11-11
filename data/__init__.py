"""Data loading, preprocessing, and Sudoku mosaic generation module."""

from .mnist_loader import (
    BRIGHTNESS_THRESHOLD,
    MNISTSubset,
    detect_blank_cell,
    download_and_preprocess_mnist,
    load_processed_mnist,
)
from .sudoku_generator import (
    SudokuMosaicInstance,
    generate_full_grid,
    generate_puzzle_with_unique_solution,
    generate_sudoku_mosaic_dataset,
    render_mosaic_image,
)

__all__ = [
    "BRIGHTNESS_THRESHOLD",
    "MNISTSubset",
    "SudokuMosaicInstance",
    "detect_blank_cell",
    "download_and_preprocess_mnist",
    "generate_full_grid",
    "generate_puzzle_with_unique_solution",
    "generate_sudoku_mosaic_dataset",
    "load_processed_mnist",
    "render_mosaic_image",
]

