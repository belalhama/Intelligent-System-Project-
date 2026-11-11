"""
Comprehensive Test Suite for P7: Handwritten Sudoku Reader and Solver.

Deliverable #9:
- At least 5 normal cases
- At least 3 edge cases
- At least 2 impossible cases (each verified to produce a clear "No solution" message
  and list the least confident cells)
- Plus verification of Search counters on the small known-answer example and
  switchable CSP components (BT, FC, AC-3).
"""

from __future__ import annotations

import numpy as np
import pytest

from csp.sudoku_csp import (
    Grid,
    SudokuCSP,
    ac3,
    count_solutions,
    find_duplicate_conflicts,
    init_domains,
    is_valid_complete_grid,
    solve_sudoku,
)
from data.mnist_loader import BRIGHTNESS_THRESHOLD, detect_blank_cell, load_processed_mnist
from data.sudoku_generator import (
    generate_puzzle_with_unique_solution,
    generate_sudoku_mosaic_dataset,
)
from integration.pipeline import SudokuReaderSolverPipeline
from nn.digit_classifier import load_or_train_models
from search.correction_search import (
    run_correction_search,
    run_small_known_example,
)


def _sample_valid_solution() -> Grid:
    return [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]


def _sample_35_given_puzzle() -> Grid:
    sol = _sample_valid_solution()
    mask = [
        [1, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, 1],
    ]
    return [[sol[r][c] if mask[r][c] else 0 for c in range(9)] for r in range(9)]


def _make_synthetic_probs(
    grid: Grid,
    overrides=None,
):
    overrides = overrides or {}
    probs_map = {}
    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if val == 0:
                continue
            if (r, c) in overrides:
                probs_map[(r, c)] = overrides[(r, c)]
            else:
                p = np.full(9, 0.001, dtype=np.float64)
                p[val - 1] = 0.992
                p /= p.sum()
                probs_map[(r, c)] = p
    return probs_map


# ==============================================================================
# NORMAL TEST CASES (6 normal cases >= 5 required)
# ==============================================================================


def test_normal_1_zero_misreads_clean_puzzle():
    """Normal Case 1: Clean 35-given puzzle with 0 misreads solves in 1 search node."""
    puzzle = _sample_35_given_puzzle()
    expected_sol = _sample_valid_solution()
    probs = _make_synthetic_probs(puzzle)

    res = run_correction_search(puzzle, probs, algorithm="astar", max_corrections=3)
    assert res.solved is True
    assert res.nodes_expanded == 1
    assert res.total_cost == 0.0
    assert len(res.corrections) == 0
    assert res.solution_grid == expected_sol


def test_normal_2_single_misread_known_example():
    """Normal Case 2: Small known-answer example (1 misread 7->1 with 52% vs 45% prob)."""
    results = run_small_known_example()
    for alg in ("ucs", "greedy", "astar"):
        r = results[alg]
        assert r.solved is True
        assert len(r.corrections) == 1
        assert r.corrections[0].cell == (1, 1)
        assert r.corrections[0].read_digit == 1
        assert r.corrections[0].new_digit == 7
        assert r.nodes_expanded >= 2
        assert r.nodes_generated > r.nodes_expanded
    # A* expands fewer nodes than UCS because heuristic skips the non-conflicting distractor
    assert results["astar"].nodes_expanded < results["ucs"].nodes_expanded


def test_normal_3_two_misreads_row_and_box_conflicts():
    """Normal Case 3: Two misreads causing simultaneous row and box conflicts."""
    puzzle = _sample_35_given_puzzle()
    expected_sol = _sample_valid_solution()
    corrupted = [row[:] for row in puzzle]
    # Cell (0, 0) true=5 -> read as 7 (conflicts with (0, 4)=7)
    # Cell (4, 3) true=8 -> read as 4 (conflicts with (4, 0)=4)
    corrupted[0][0] = 7
    corrupted[4][3] = 4

    p1 = np.full(9, 0.002, dtype=np.float64)
    p1[7 - 1] = 0.54
    p1[5 - 1] = 0.44
    p1 /= p1.sum()

    p2 = np.full(9, 0.002, dtype=np.float64)
    p2[4 - 1] = 0.53
    p2[8 - 1] = 0.45
    p2 /= p2.sum()

    probs = _make_synthetic_probs(corrupted, overrides={(0, 0): p1, (4, 3): p2})
    res_astar = run_correction_search(corrupted, probs, algorithm="astar", max_corrections=3)
    res_ucs = run_correction_search(corrupted, probs, algorithm="ucs", max_corrections=3)

    assert res_astar.solved is True
    assert res_astar.solution_grid == expected_sol
    assert len(res_astar.corrections) == 2
    assert abs(res_astar.total_cost - res_ucs.total_cost) < 1e-7
    assert res_astar.nodes_expanded <= res_ucs.nodes_expanded


def test_normal_4_three_misreads_at_max_budget():
    """Normal Case 4: Three misread digits (at maximum budget of 3 corrections)."""
    puzzle = _sample_35_given_puzzle()
    expected_sol = _sample_valid_solution()
    corrupted = [row[:] for row in puzzle]
    corrupted[0][0] = 7  # true 5
    corrupted[1][1] = 1  # true 7
    corrupted[8][8] = 8  # true 9

    overrides = {}
    for cell, read_d, true_d in [((0, 0), 7, 5), ((1, 1), 1, 7), ((8, 8), 8, 9)]:
        p = np.full(9, 0.002, dtype=np.float64)
        p[read_d - 1] = 0.52
        p[true_d - 1] = 0.46
        p /= p.sum()
        overrides[cell] = p

    probs = _make_synthetic_probs(corrupted, overrides=overrides)
    res = run_correction_search(corrupted, probs, algorithm="astar", max_corrections=3)
    assert res.solved is True
    assert len(res.corrections) == 3
    assert res.solution_grid == expected_sol


def test_normal_5_csp_switchable_bt_fc_ac3():
    """Normal Case 5: Switchable CSP components (BT, BT+FC, BT+FC+AC3) all solve accurately."""
    puzzle = _sample_35_given_puzzle()
    expected_sol = _sample_valid_solution()

    sol_bt, st_bt = solve_sudoku(puzzle, use_fc=False, use_ac3=False)
    sol_fc, st_fc = solve_sudoku(puzzle, use_fc=True, use_ac3=False)
    sol_ac3, st_ac3 = solve_sudoku(puzzle, use_fc=True, use_ac3=True)

    assert sol_bt == expected_sol and is_valid_complete_grid(sol_bt)
    assert sol_fc == expected_sol and is_valid_complete_grid(sol_fc)
    assert sol_ac3 == expected_sol and is_valid_complete_grid(sol_ac3)

    # Verify progressive pruning effectiveness: BT+FC+AC3 <= BT+FC <= BT
    assert st_ac3.assignments <= st_fc.assignments <= st_bt.assignments
    assert st_ac3.backtracks <= st_fc.backtracks <= st_bt.backtracks


def test_normal_6_end_to_end_mnist_mosaic_pipeline():
    """Normal Case 6: Full end-to-end integration test on a 9x9 MNIST image mosaic."""
    dataset = load_processed_mnist(seed=42)
    nn_clf, _, _ = load_or_train_models(dataset=dataset, seed=42)
    test_probs = nn_clf.predict_proba(dataset.X_test)

    mosaics = generate_sudoku_mosaic_dataset(
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        num_puzzles=1,
        num_givens=30,
        num_injected_misreads=1,
        seed=42,
        classifier_probs_on_test=test_probs,
        predict_proba_fn=nn_clf.predict_proba,
    )
    m = mosaics[0]
    pipeline = SudokuReaderSolverPipeline(
        classifier=nn_clf, search_algorithm="astar", max_corrections=3
    )
    out = pipeline.solve_mosaic(
        cell_images=m.cell_images,
        mode="nn_guided",
        true_puzzle_grid=m.puzzle_grid,
        true_solution_grid=m.solution_grid,
    )
    assert out.solved is True
    assert out.exact_recovery is True
    assert is_valid_complete_grid(out.solution_grid)


# ==============================================================================
# EDGE TEST CASES (4 edge cases >= 3 required)
# ==============================================================================


def test_edge_1_sparse_25_givens_with_unique_solution():
    """Edge Case 1: Minimum-givens (25 clues) sparse puzzle solved with uniqueness check."""
    rng = np.random.default_rng(123)
    puzzle, expected_sol = generate_puzzle_with_unique_solution(rng, num_givens=25)
    num_clues = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)
    assert num_clues == 25
    assert count_solutions(puzzle, max_solutions=2) == 1

    sol, stats = solve_sudoku(puzzle, use_fc=True, use_ac3=True, require_unique=True)
    assert stats.solved is True
    assert stats.unique is True
    assert sol == expected_sol


def test_edge_2_near_tie_probabilities_non_negative_cost():
    """Edge Case 2: Near-tie top-2 probabilities (0.5001 vs 0.4999) give tiny non-negative cost."""
    puzzle = _sample_35_given_puzzle()
    corrupted = [row[:] for row in puzzle]
    corrupted[0][0] = 7  # true is 5, conflicts with (0, 4)=7

    p_tie = np.zeros(9, dtype=np.float64)
    p_tie[7 - 1] = 0.5001
    p_tie[5 - 1] = 0.4999
    probs = _make_synthetic_probs(corrupted, overrides={(0, 0): p_tie})

    res = run_correction_search(corrupted, probs, algorithm="astar", max_corrections=3)
    assert res.solved is True
    assert len(res.corrections) == 1
    assert 0.0 <= res.corrections[0].cost < 0.001


def test_edge_3_brightness_rule_threshold_boundary():
    """Edge Case 3: Blank cell detection precisely around BRIGHTNESS_THRESHOLD (0.025)."""
    below_img = np.full((28, 28), BRIGHTNESS_THRESHOLD - 1e-4, dtype=np.float32)
    above_img = np.full((28, 28), BRIGHTNESS_THRESHOLD + 1e-4, dtype=np.float32)
    assert detect_blank_cell(below_img) is True
    assert detect_blank_cell(above_img) is False


def test_edge_4_underconstrained_grid_rejected_when_require_unique():
    """Edge Case 4: Grid with only 5 givens has multiple solutions and fails require_unique=True."""
    sparse_grid = [[0] * 9 for _ in range(9)]
    sparse_grid[0][0] = 1
    sparse_grid[1][1] = 2
    sparse_grid[2][2] = 3
    sparse_grid[3][3] = 4
    sparse_grid[4][4] = 5

    sol_any, st_any = solve_sudoku(sparse_grid, require_unique=False)
    assert sol_any is not None and st_any.solved is True

    sol_uniq, st_uniq = solve_sudoku(sparse_grid, require_unique=True)
    assert sol_uniq is None
    assert st_uniq.solved is False
    assert "No solution" in st_uniq.message


# ==============================================================================
# IMPOSSIBLE TEST CASES (3 impossible cases >= 2 required)
# ==============================================================================


def test_impossible_1_five_conflicts_exceed_correction_budget():
    """
    Impossible Case 1: 5 disjoint duplicate-given conflicts exceed max_corrections=3.
    Must return solved=False, a clear 'No solution: unreadable puzzle' message,
    and list the least confident cells.
    """
    puzzle = _sample_35_given_puzzle()
    corrupted = [row[:] for row in puzzle]
    # Inject 5 independent duplicate conflicts in rows 0, 1, 3, 4, 5
    corrupted[0][0] = 7  # conflicts with (0, 4)=7
    corrupted[1][0] = 9  # conflicts with (1, 4)=9
    corrupted[3][0] = 6  # conflicts with (3, 4)=6
    corrupted[4][0] = 3  # conflicts with (4, 5)=3
    corrupted[5][0] = 2  # conflicts with (5, 4)=2

    probs = _make_synthetic_probs(corrupted)
    res = run_correction_search(corrupted, probs, algorithm="astar", max_corrections=3)

    assert res.solved is False
    assert res.solution_grid is None
    assert "No solution" in res.message and "unreadable" in res.message.lower()
    assert len(res.least_confident_cells) > 0


def test_impossible_2_unfixable_contradiction_locked_alternatives():
    """
    Impossible Case 2: A puzzle where the conflicting cells only have alternative
    digits that ALSO conflict in the same row/column/box, making it unsolvable.
    """
    puzzle = _sample_35_given_puzzle()
    corrupted = [row[:] for row in puzzle]
    # Make cell (0, 0) read as 7 (conflicts with (0,4)=7), and give all cells top
    # alternatives that are also already present in row 0 / col 0 / box 0!
    corrupted[0][0] = 7
    bad_p = np.full(9, 1e-9, dtype=np.float64)
    # Row 0 / Col 0 / Box 0 already contain 3, 4, 7, 1, 6, 8, 9:
    bad_p[7 - 1] = 0.50
    bad_p[3 - 1] = 0.25  # conflicts with (0, 1)=3
    bad_p[4 - 1] = 0.15  # conflicts with (0, 2)=4
    bad_p[6 - 1] = 0.10  # conflicts with (1, 0)=6
    probs = _make_synthetic_probs(corrupted, overrides={(0, 0): bad_p})

    res = run_correction_search(
        corrupted,
        probs,
        algorithm="ucs",
        max_corrections=2,
        max_uncertain_cells=1,
        top_alts_per_cell=3,
    )
    assert res.solved is False
    assert res.solution_grid is None
    assert "No solution" in res.message


def test_impossible_3_direct_csp_conflict_all_modes():
    """
    Impossible Case 3: Direct CSP call on a grid with two equal givens in row 0
    must fail cleanly with a 'No solution' message under BT, BT+FC, and BT+FC+AC3.
    """
    bad_grid = _sample_35_given_puzzle()
    bad_grid[0][0] = 7  # Duplicates (0, 4) = 7 in row 0

    for use_fc, use_ac3 in [(False, False), (True, False), (True, True)]:
        sol, stats = solve_sudoku(bad_grid, use_fc=use_fc, use_ac3=use_ac3)
        assert sol is None
        assert stats.solved is False
        assert "No solution" in stats.message

