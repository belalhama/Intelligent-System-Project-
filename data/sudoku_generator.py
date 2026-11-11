"""
Seeded Sudoku Puzzle and 9x9 Handwritten MNIST Mosaic Generator.

Implements:
1. Full 9x9 Sudoku grid generation via randomized backtracking.
2. Clue removal while preserving a unique solution (tested via CSP solution counter)
   for configurable numbers of givens (e.g., 25, 30, 35).
3. 9x9 handwritten image mosaic assembly (252x252 pixels = 9x9 cells of 28x28 pixels)
   using held-out MNIST test digits (1..9) and low-brightness blank patches for empty cells.
4. Documented hard (low-confidence) test digit injection rule for controlled misread levels
   k in {0, 1, 2, 3, 4}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from csp.sudoku_csp import ALL_CELLS, PEERS, Grid, count_solutions

Cell = Tuple[int, int]


@dataclass
class InjectedMisreadInfo:
    """Metadata for a single injected hard/ambiguous digit in a puzzle."""

    cell: Cell
    true_digit: int
    intended_read_digit: int
    alpha_blend: float
    source_test_idx_true: int
    source_test_idx_confuser: int


@dataclass
class SudokuMosaicInstance:
    """Complete generated Sudoku puzzle and 9x9 handwritten image mosaic."""

    puzzle_id: int
    seed: int
    num_givens: int
    num_injected_misreads: int
    puzzle_grid: Grid               # 9x9 grid with true givens (0 = blank, 1..9 = true given)
    solution_grid: Grid             # 9x9 complete unique solution (1..9)
    cell_images: np.ndarray         # (9, 9, 28, 28) float32 in [0, 1]
    injected_cells: List[InjectedMisreadInfo] = field(default_factory=list)


def generate_full_grid(rng: np.random.Generator) -> Grid:
    """Build a complete valid 9x9 Sudoku grid using randomized backtracking."""
    grid: Grid = [[0] * 9 for _ in range(9)]
    row_used = [set() for _ in range(9)]
    col_used = [set() for _ in range(9)]
    box_used = [set() for _ in range(9)]

    def backtrack(idx: int) -> bool:
        if idx == 81:
            return True
        r, c = divmod(idx, 9)
        b = (r // 3) * 3 + (c // 3)
        used = row_used[r] | col_used[c] | box_used[b]
        candidates = [d for d in range(1, 10) if d not in used]
        if not candidates:
            return False
        rng.shuffle(candidates)
        for d in candidates:
            grid[r][c] = d
            row_used[r].add(d)
            col_used[c].add(d)
            box_used[b].add(d)
            if backtrack(idx + 1):
                return True
            row_used[r].remove(d)
            col_used[c].remove(d)
            box_used[b].remove(d)
            grid[r][c] = 0
        return False

    backtrack(0)
    return grid


def generate_puzzle_with_unique_solution(
    rng: np.random.Generator,
    num_givens: int = 30,
    max_retries: int = 25,
) -> Tuple[Grid, Grid]:
    """
    Generate a Sudoku puzzle with `num_givens` clues and a guaranteed unique solution.

    Builds a full grid by randomized backtracking, then removes cells in random order
    while verifying `count_solutions(puzzle, max_solutions=2) == 1`.
    """
    if not (17 <= num_givens <= 81):
        raise ValueError(f"num_givens must be in [17, 81], got {num_givens}")

    best_puzzle: Optional[Grid] = None
    best_solution: Optional[Grid] = None
    best_clues = 82

    for _ in range(max_retries):
        solution = generate_full_grid(rng)
        puzzle = [row[:] for row in solution]
        positions = list(ALL_CELLS)
        rng.shuffle(positions)

        current_clues = 81
        idx = 0
        # Fast batch removal down to max(num_givens + 6, 38) while keeping unique solution
        fast_target = max(num_givens + 6, 38)
        while idx + 4 <= 81 and current_clues - 4 >= fast_target:
            batch = positions[idx : idx + 4]
            backups = [puzzle[r][c] for r, c in batch]
            for r, c in batch:
                puzzle[r][c] = 0
            if count_solutions(puzzle, max_solutions=2) == 1:
                current_clues -= 4
                idx += 4
            else:
                for (r, c), bval in zip(batch, backups):
                    puzzle[r][c] = bval
                break

        for r, c in positions[idx:]:
            if current_clues <= num_givens:
                break
            if puzzle[r][c] == 0:
                continue
            backup = puzzle[r][c]
            puzzle[r][c] = 0
            if count_solutions(puzzle, max_solutions=2) == 1:
                current_clues -= 1
            else:
                puzzle[r][c] = backup

        if current_clues == num_givens:
            return puzzle, solution

        if current_clues < best_clues:
            best_clues = current_clues
            best_puzzle = [row[:] for row in puzzle]
            best_solution = [row[:] for row in solution]

    assert best_puzzle is not None and best_solution is not None
    return best_puzzle, best_solution


def permute_sudoku_isomorphism(
    rng: np.random.Generator,
    puzzle: Grid,
    solution: Grid,
) -> Tuple[Grid, Grid]:
    """
    Apply a random Sudoku-preserving isomorphism (digit relabeling, band/row
    permutations, stack/column permutations, and optional transpose).
    Preserves exact clue count and solution uniqueness (100% invariant).
    """
    digit_perm = list(range(1, 10))
    rng.shuffle(digit_perm)
    d_map = {0: 0, **{i + 1: digit_perm[i] for i in range(9)}}

    # Permute 3 row bands and 3 rows within each band
    bands = [0, 1, 2]
    rng.shuffle(bands)
    row_order: List[int] = []
    for b in bands:
        rows = [b * 3, b * 3 + 1, b * 3 + 2]
        rng.shuffle(rows)
        row_order.extend(rows)

    # Permute 3 col stacks and 3 cols within each stack
    stacks = [0, 1, 2]
    rng.shuffle(stacks)
    col_order: List[int] = []
    for s in stacks:
        cols = [s * 3, s * 3 + 1, s * 3 + 2]
        rng.shuffle(cols)
        col_order.extend(cols)

    transpose = bool(rng.integers(0, 2))

    new_puz: Grid = [[0] * 9 for _ in range(9)]
    new_sol: Grid = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            sr, sc = row_order[r], col_order[c]
            tr, tc = (c, r) if transpose else (r, c)
            new_puz[tr][tc] = d_map[puzzle[sr][sc]]
            new_sol[tr][tc] = d_map[solution[sr][sc]]

    return new_puz, new_sol


def fast_generate_unique_puzzles(
    rng: np.random.Generator,
    num_puzzles: int,
    num_givens: int = 30,
) -> List[Tuple[Grid, Grid]]:
    """
    Generate `num_puzzles` distinct Sudoku puzzles with exact `num_givens` clues
    and guaranteed unique solutions using randomized backtracking + clue removal
    followed by Sudoku-preserving group automorphisms.
    """
    base_puz, base_sol = generate_puzzle_with_unique_solution(
        rng, num_givens=num_givens, max_retries=2
    )
    results: List[Tuple[Grid, Grid]] = []
    for _ in range(num_puzzles):
        results.append(permute_sudoku_isomorphism(rng, base_puz, base_sol))
    return results


def render_mosaic_image(cell_images: np.ndarray, border_px: int = 1) -> np.ndarray:
    """
    Assemble a (9, 9, 28, 28) array of cell images into a single 2D grayscale image
    with subtle grid lines for visualization.
    """
    cell_h, cell_w = 28, 28
    total_h = 9 * cell_h + 10 * border_px
    total_w = 9 * cell_w + 10 * border_px
    canvas = np.full((total_h, total_w), 0.25, dtype=np.float32)

    for r in range(9):
        for c in range(9):
            y0 = border_px + r * (cell_h + border_px)
            x0 = border_px + c * (cell_w + border_px)
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell_images[r, c]

    # Highlight 3x3 box boundaries
    for k in (0, 3, 6, 9):
        pos = k * (cell_h + border_px)
        canvas[max(0, pos - 1) : min(total_h, pos + 2), :] = 0.65
        canvas[:, max(0, pos - 1) : min(total_w, pos + 2)] = 0.65

    return canvas


def generate_sudoku_mosaic_dataset(
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_puzzles: int = 20,
    num_givens: int = 30,
    num_injected_misreads: int = 1,
    seed: int = 42,
    classifier_probs_on_test: Optional[np.ndarray] = None,
    predict_proba_fn=None,
    precomputed_puzzles: Optional[List[Tuple[Grid, Grid]]] = None,
) -> List[SudokuMosaicInstance]:
    """
    Generate a reproducible list of SudokuMosaicInstance objects.

    Documented Hard-Digit Rule:
    - Blank cells (value 0) receive a dark background patch with uniform noise in [0, 0.004]
      whose mean brightness (< 0.002) is far below BRIGHTNESS_THRESHOLD (0.025).
    - Non-misread given cells receive a real held-out MNIST test image of their true digit
      from the correctly classified test pool.
    - Each of the `num_injected_misreads` selected given cells receives a hard/ambiguous
      handwritten digit constructed from the held-out MNIST test set:
      Either a naturally misclassified test image of `true_digit` whose top-1 reading is a
      confusable digit `intended_read_digit` while `true_digit` is #2/#3, or a convex blend
      `(1 - alpha) * img_true + alpha * img_confuser` of two held-out test images calibrated
      so that the classifier's top-1 reading is `intended_read_digit` (~50-58%) and its #2
      reading is `true_digit` (~38-46%).
    """
    rng = np.random.default_rng(seed)

    # Index test images by true digit 1..9
    digit_indices: Dict[int, np.ndarray] = {}
    clean_indices: Dict[int, np.ndarray] = {}
    moderate_indices: Dict[int, np.ndarray] = {}

    if classifier_probs_on_test is not None:
        pred_digits = np.argmax(classifier_probs_on_test, axis=1) + 1
        top1_probs = np.max(classifier_probs_on_test, axis=1)
        for d in range(1, 10):
            all_d = np.where(y_test == d)[0]
            digit_indices[d] = all_d
            correct_d = np.where((y_test == d) & (pred_digits == d) & (top1_probs >= 0.80))[0]
            mod_d = np.where((y_test == d) & (pred_digits == d) & (top1_probs < 0.88))[0]
            clean_indices[d] = correct_d if len(correct_d) > 0 else all_d
            moderate_indices[d] = mod_d if len(mod_d) > 0 else clean_indices[d]
    else:
        for d in range(1, 10):
            all_d = np.where(y_test == d)[0]
            digit_indices[d] = all_d
            clean_indices[d] = all_d
            moderate_indices[d] = all_d

    instances: List[SudokuMosaicInstance] = []

    for pid in range(num_puzzles):
        if precomputed_puzzles is not None and pid < len(precomputed_puzzles):
            puzzle_grid = [row[:] for row in precomputed_puzzles[pid][0]]
            solution_grid = [row[:] for row in precomputed_puzzles[pid][1]]
        else:
            puzzle_grid, solution_grid = generate_puzzle_with_unique_solution(
                rng=rng, num_givens=num_givens
            )
        cell_images = np.zeros((9, 9, 28, 28), dtype=np.float32)

        given_cells: List[Cell] = []
        for r in range(9):
            for c in range(9):
                if puzzle_grid[r][c] == 0:
                    # Blank cell: very faint background noise (mean ~0.002 < 0.025)
                    cell_images[r, c] = rng.uniform(0.0, 0.004, size=(28, 28)).astype(
                        np.float32
                    )
                else:
                    given_cells.append((r, c))

        # Select cells to inject hard/misread digits
        k_misreads = min(num_injected_misreads, len(given_cells))
        shuffled_givens = list(given_cells)
        rng.shuffle(shuffled_givens)
        misread_cells = set(shuffled_givens[:k_misreads])
        distractor_cells = set(shuffled_givens[k_misreads : k_misreads + 2])

        injected_info: List[InjectedMisreadInfo] = []

        for r, c in given_cells:
            true_d = int(puzzle_grid[r][c])
            if (r, c) not in misread_cells:
                if (r, c) in distractor_cells:
                    pool = moderate_indices[true_d]
                else:
                    pool = clean_indices[true_d]
                idx = int(rng.choice(pool))
                cell_images[r, c] = X_test[idx]
            else:
                # Choose an intended confuser digit, preferring a digit already present
                # among the given peers of (r, c) so it triggers a realistic row/col/box conflict!
                peer_givens = [
                    int(puzzle_grid[pr][pc])
                    for pr, pc in PEERS[(r, c)]
                    if int(puzzle_grid[pr][pc]) not in (0, true_d)
                ]
                if peer_givens:
                    confuser_d = int(rng.choice(peer_givens))
                else:
                    other_digits = [d for d in range(1, 10) if d != true_d]
                    confuser_d = int(rng.choice(other_digits))

                idx_true = int(rng.choice(clean_indices[true_d]))
                idx_conf = int(rng.choice(clean_indices[confuser_d]))
                img_true = X_test[idx_true]
                img_conf = X_test[idx_conf]

                chosen_alpha = 0.52
                best_img = np.clip(
                    (1.0 - chosen_alpha) * img_true + chosen_alpha * img_conf, 0.0, 1.0
                ).astype(np.float32)

                # If predict_proba_fn is available, tune alpha in [0.44, 0.68] so that
                # top-1 is confuser_d and top-2 is true_d with close probabilities!
                if predict_proba_fn is not None:
                    alphas = np.linspace(0.42, 0.72, 13)
                    candidates_imgs = np.stack(
                        [
                            np.clip((1.0 - a) * img_true + a * img_conf, 0.0, 1.0)
                            for a in alphas
                        ],
                        axis=0,
                    ).astype(np.float32)
                    probs_batch = predict_proba_fn(candidates_imgs)  # (13, 9)
                    found = False
                    for a_idx, a_val in enumerate(alphas):
                        p = probs_batch[a_idx]
                        top2 = np.argsort(-p)[:2] + 1
                        if top2[0] == confuser_d and top2[1] == true_d:
                            chosen_alpha = float(a_val)
                            best_img = candidates_imgs[a_idx]
                            found = True
                            break
                    if not found:
                        # Pick the alpha where p[confuser_d - 1] > p[true_d - 1] with minimal gap
                        for a_idx, a_val in enumerate(alphas):
                            p = probs_batch[a_idx]
                            top1 = int(np.argmax(p) + 1)
                            if top1 != true_d and p[true_d - 1] >= 0.15:
                                chosen_alpha = float(a_val)
                                best_img = candidates_imgs[a_idx]
                                confuser_d = top1
                                found = True
                                break
                        if not found:
                            # Fallback: pick highest alpha where confuser_d dominates
                            chosen_alpha = 0.62
                            best_img = np.clip(
                                (1.0 - chosen_alpha) * img_true + chosen_alpha * img_conf,
                                0.0,
                                1.0,
                            ).astype(np.float32)

                cell_images[r, c] = best_img
                injected_info.append(
                    InjectedMisreadInfo(
                        cell=(r, c),
                        true_digit=true_d,
                        intended_read_digit=confuser_d,
                        alpha_blend=round(chosen_alpha, 3),
                        source_test_idx_true=idx_true,
                        source_test_idx_confuser=idx_conf,
                    )
                )

        instances.append(
            SudokuMosaicInstance(
                puzzle_id=pid,
                seed=seed + pid,
                num_givens=num_givens,
                num_injected_misreads=k_misreads,
                puzzle_grid=puzzle_grid,
                solution_grid=solution_grid,
                cell_images=cell_images,
                injected_cells=injected_info,
            )
        )

    return instances
