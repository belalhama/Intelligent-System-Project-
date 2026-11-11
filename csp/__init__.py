"""CSP module for Sudoku solving with Backtracking, Forward Checking, and AC-3."""

from .sudoku_csp import (
    CSPStats,
    SudokuCSP,
    ac3,
    find_duplicate_conflicts,
    is_valid_complete_grid,
    solve_sudoku,
    count_solutions,
)

__all__ = [
    "CSPStats",
    "SudokuCSP",
    "ac3",
    "find_duplicate_conflicts",
    "is_valid_complete_grid",
    "solve_sudoku",
    "count_solutions",
]

