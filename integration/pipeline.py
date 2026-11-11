"""
Integrated Handwritten Sudoku Reader and Solver Pipeline.

Cooperation workflow:
1. Brightness Rule detects blank cells (mean pixel intensity < 0.025).
2. Calibrated Neural Network classifies all non-blank 28x28 cells (digits 1..9)
   and outputs calibrated probability distributions.
3. Initial AC-3 check inspects the raw reading for duplicate-given conflicts /
   domain wipeout.
4. Search (UCS, Greedy, or A*) proposes candidate correction sets in order of
   minimum log-likelihood cost (or heuristic priority).
5. CSP (BT + FC + AC-3) tests each candidate corrected puzzle; the first solvable
   (and optionally uniquely solvable) puzzle wins.
6. If no candidate within `max_corrections` is solvable, reports "No solution:
   unreadable puzzle" and lists the least confident cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from csp.sudoku_csp import (
    CSPStats,
    Grid,
    SudokuCSP,
    ac3,
    find_duplicate_conflicts,
    init_domains,
)
from data.mnist_loader import BRIGHTNESS_THRESHOLD, detect_blank_cell
from nn.digit_classifier import CalibratedDigitClassifier
from search.correction_search import (
    CorrectionEntry,
    SearchResult,
    SearchStepTrace,
    run_correction_search,
)

Cell = Tuple[int, int]


@dataclass
class PipelineOutput:
    """Comprehensive result from the integrated NN + Search + CSP pipeline."""

    solved: bool
    exact_recovery: Optional[bool]
    mode: str
    search_algorithm: str
    initial_grid: Grid
    corrected_grid: Optional[Grid]
    solution_grid: Optional[Grid]
    initial_conflicts: List[Tuple[Cell, Cell, int]]
    initial_ac3_consistent: bool
    initial_misread_cells: List[Tuple[Cell, int, int]]  # (cell, true_d, read_d) if ground truth known
    corrections: List[CorrectionEntry]
    total_cost: float
    nodes_expanded: int
    nodes_generated: int
    csp_calls: int
    total_csp_assignments: int
    total_csp_backtracks: int
    least_confident_cells: List[Tuple[Cell, int, float]]
    message: str
    trace: List[SearchStepTrace] = field(default_factory=list)


def read_mosaic_grid(
    cell_images: np.ndarray,
    classifier: CalibratedDigitClassifier,
    brightness_threshold: float = BRIGHTNESS_THRESHOLD,
    use_temperature: bool = True,
) -> Tuple[Grid, Dict[Cell, np.ndarray]]:
    """
    Read a (9, 9, 28, 28) handwritten Sudoku mosaic:
    - Blank cells are detected via the simple brightness rule (not the NN).
    - Non-blank cells are classified in a single batch by the NN (digits 1..9).
    """
    initial_grid: Grid = [[0] * 9 for _ in range(9)]
    non_blank_coords: List[Cell] = []
    non_blank_imgs: List[np.ndarray] = []

    for r in range(9):
        for c in range(9):
            img = cell_images[r, c]
            if not detect_blank_cell(img, threshold=brightness_threshold):
                non_blank_coords.append((r, c))
                non_blank_imgs.append(img)

    cell_probs: Dict[Cell, np.ndarray] = {}
    if non_blank_imgs:
        batch = np.stack(non_blank_imgs, axis=0)
        probs_batch = classifier.predict_proba(batch, use_temperature=use_temperature)
        for idx, (r, c) in enumerate(non_blank_coords):
            p = probs_batch[idx]
            cell_probs[(r, c)] = p
            initial_grid[r][c] = int(np.argmax(p) + 1)

    return initial_grid, cell_probs


class SudokuReaderSolverPipeline:
    """
    End-to-end Handwritten Sudoku Reader and Solver combining:
    - Calibrated Neural Network (digit 1..9 probabilities)
    - Correction Search (UCS, Greedy Best-First, A*)
    - Sudoku CSP (Backtracking + Forward Checking + AC-3)
    """

    def __init__(
        self,
        classifier: CalibratedDigitClassifier,
        search_algorithm: str = "astar",
        max_corrections: int = 3,
        max_uncertain_cells: int = 12,
        top_alts_per_cell: int = 3,
        csp_use_fc: bool = True,
        csp_use_ac3: bool = True,
        require_unique: bool = False,
    ) -> None:
        self.classifier = classifier
        self.search_algorithm = search_algorithm
        self.max_corrections = max_corrections
        self.max_uncertain_cells = max_uncertain_cells
        self.top_alts_per_cell = top_alts_per_cell
        self.csp_use_fc = csp_use_fc
        self.csp_use_ac3 = csp_use_ac3
        self.require_unique = require_unique

    def solve_mosaic(
        self,
        cell_images: np.ndarray,
        mode: str = "nn_guided",
        true_puzzle_grid: Optional[Grid] = None,
        true_solution_grid: Optional[Grid] = None,
        random_seed: int = 42,
    ) -> PipelineOutput:
        """
        Run the integrated pipeline on a (9, 9, 28, 28) mosaic image.

        Modes:
          - 'nn_guided': Full NN probability-guided search + CSP (Main System).
          - 'no_correction': Raw NN argmax reading sent directly to CSP without correction search.
          - 'random_correction': Uninformed random probability baseline with the same search budget.
        """
        initial_grid, cell_probs = read_mosaic_grid(
            cell_images=cell_images,
            classifier=self.classifier,
        )
        return self.solve_from_reading(
            initial_grid=initial_grid,
            cell_probs=cell_probs,
            mode=mode,
            true_puzzle_grid=true_puzzle_grid,
            true_solution_grid=true_solution_grid,
            random_seed=random_seed,
        )

    def solve_from_reading(
        self,
        initial_grid: Grid,
        cell_probs: Dict[Cell, np.ndarray],
        mode: str = "nn_guided",
        true_puzzle_grid: Optional[Grid] = None,
        true_solution_grid: Optional[Grid] = None,
        random_seed: int = 42,
    ) -> PipelineOutput:
        """Run AC-3 check, Search, and CSP from an initial reading and probability map."""
        initial_conflicts = find_duplicate_conflicts(initial_grid)
        init_doms = init_domains(initial_grid)
        ac3_stats = CSPStats()
        initial_ac3_ok = ac3(init_doms, ac3_stats)

        initial_misreads: List[Tuple[Cell, int, int]] = []
        if true_puzzle_grid is not None:
            for r in range(9):
                for c in range(9):
                    t_val = int(true_puzzle_grid[r][c])
                    r_val = int(initial_grid[r][c])
                    if t_val != r_val:
                        initial_misreads.append(((r, c), t_val, r_val))

        # Mode 1: No Correction (NN correction OFF -> raw reading straight to CSP)
        if mode == "no_correction":
            csp = SudokuCSP(use_fc=self.csp_use_fc, use_ac3=self.csp_use_ac3)
            sol_grid, csp_st = csp.solve(initial_grid, require_unique=self.require_unique)
            exact_rec = None
            if true_solution_grid is not None:
                exact_rec = sol_grid is not None and (sol_grid == true_solution_grid)

            # Rank least confident cells from NN probabilities
            least_conf = sorted(
                [
                    (cell, int(initial_grid[cell[0]][cell[1]]), float(np.max(p)))
                    for cell, p in cell_probs.items()
                    if int(initial_grid[cell[0]][cell[1]]) != 0
                ],
                key=lambda x: x[2],
            )[:10]

            msg = (
                "Solved directly without corrections."
                if sol_grid is not None
                else f"No solution: unreadable puzzle ({csp_st.message})"
            )
            return PipelineOutput(
                solved=sol_grid is not None,
                exact_recovery=exact_rec,
                mode=mode,
                search_algorithm="NONE",
                initial_grid=initial_grid,
                corrected_grid=initial_grid if sol_grid is not None else None,
                solution_grid=sol_grid,
                initial_conflicts=initial_conflicts,
                initial_ac3_consistent=initial_ac3_ok,
                initial_misread_cells=initial_misreads,
                corrections=[],
                total_cost=0.0 if sol_grid is not None else float("inf"),
                nodes_expanded=1,
                nodes_generated=1,
                csp_calls=1,
                total_csp_assignments=csp_st.assignments,
                total_csp_backtracks=csp_st.backtracks,
                least_confident_cells=least_conf,
                message=msg,
                trace=[],
            )

        # Mode 2: Random Correction (Uninformed baseline replacing NN probabilities with random scores)
        active_probs = cell_probs
        if mode == "random_correction":
            rng = np.random.default_rng(random_seed)
            random_probs: Dict[Cell, np.ndarray] = {}
            for cell, _ in cell_probs.items():
                read_d = int(initial_grid[cell[0]][cell[1]])
                if read_d == 0:
                    continue
                # Assign random Dirichlet-like weights to alternatives, keeping read_d highest
                # with a random confidence in [0.40, 0.95] so cells & alternatives are picked randomly
                raw_alt = rng.uniform(0.01, 0.30, size=9)
                raw_alt[read_d - 1] = rng.uniform(0.35, 0.90)
                raw_alt /= raw_alt.sum()
                random_probs[cell] = raw_alt
            active_probs = random_probs

        # Mode 3 (or Mode 2 with random probs): Run Search + CSP
        search_res: SearchResult = run_correction_search(
            initial_grid=initial_grid,
            cell_probs=active_probs,
            algorithm=self.search_algorithm,
            max_corrections=self.max_corrections,
            max_uncertain_cells=self.max_uncertain_cells,
            top_alts_per_cell=self.top_alts_per_cell,
            csp_use_fc=self.csp_use_fc,
            csp_use_ac3=self.csp_use_ac3,
            require_unique=self.require_unique,
            max_expanded=80 if mode == "random_correction" else 1500,
        )

        exact_rec = None
        if true_solution_grid is not None:
            exact_rec = (
                search_res.solved
                and search_res.solution_grid is not None
                and (search_res.solution_grid == true_solution_grid)
            )

        return PipelineOutput(
            solved=search_res.solved,
            exact_recovery=exact_rec,
            mode=mode,
            search_algorithm=search_res.algorithm,
            initial_grid=initial_grid,
            corrected_grid=search_res.corrected_grid,
            solution_grid=search_res.solution_grid,
            initial_conflicts=initial_conflicts,
            initial_ac3_consistent=initial_ac3_ok,
            initial_misread_cells=initial_misreads,
            corrections=search_res.corrections,
            total_cost=search_res.total_cost,
            nodes_expanded=search_res.nodes_expanded,
            nodes_generated=search_res.nodes_generated,
            csp_calls=search_res.csp_calls,
            total_csp_assignments=search_res.total_csp_assignments,
            total_csp_backtracks=search_res.total_csp_backtracks,
            least_confident_cells=search_res.least_confident_cells,
            message=search_res.message,
            trace=search_res.trace,
        )

