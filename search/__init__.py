"""Search module for finding minimum-cost digit corrections in Sudoku."""

from .correction_search import (
    CorrectionEntry,
    SearchResult,
    SearchStepTrace,
    compute_conflict_heuristic,
    run_correction_search,
    run_small_known_example,
)

__all__ = [
    "CorrectionEntry",
    "SearchResult",
    "SearchStepTrace",
    "compute_conflict_heuristic",
    "run_correction_search",
    "run_small_known_example",
]

