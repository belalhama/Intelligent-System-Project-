"""
Search Algorithms (UCS, Greedy Best-First, A*) for Sudoku Digit Correction.

Formulation:
- State: Sorted tuple of corrections ((r, c), new_digit) applied to the initial reading.
- Action: Change one uncertain given cell (r, c) to one of its top alternative digits.
- Step Cost: log P(read_digit | image) - log P(new_digit | image) >= 0 (never negative).
- Goal Test: The corrected 9x9 grid is solvable by the Sudoku CSP (and optionally has a
  unique solution).
- A* Heuristic: Admissible lower bound based on vertex-disjoint duplicate-digit conflicts
  ("at least one cell of any duplicate-digit conflict must change").
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from csp.sudoku_csp import (
    PEERS,
    CSPStats,
    Grid,
    SudokuCSP,
    find_duplicate_conflicts,
)

Cell = Tuple[int, int]
StateTuple = Tuple[Tuple[Cell, int], ...]


@dataclass
class CorrectionEntry:
    """Single cell correction log entry."""

    cell: Cell
    read_digit: int
    new_digit: int
    cost: float
    read_prob: float
    new_prob: float


@dataclass
class SearchStepTrace:
    """Trace record of a single expanded search state."""

    step: int
    algorithm: str
    corrections: List[Tuple[Cell, int, int]]  # (cell, old_digit, new_digit)
    g_cost: float
    h_cost: float
    f_cost: float
    num_conflicts: int
    csp_tested: bool
    csp_solved: bool
    csp_backtracks: int
    csp_assignments: int
    note: str = ""


@dataclass
class SearchResult:
    """Complete output of a correction search run."""

    solved: bool
    algorithm: str
    corrected_grid: Optional[Grid]
    solution_grid: Optional[Grid]
    corrections: List[CorrectionEntry]
    total_cost: float
    nodes_expanded: int
    nodes_generated: int
    csp_calls: int
    total_csp_assignments: int
    total_csp_backtracks: int
    message: str
    least_confident_cells: List[Tuple[Cell, int, float]] = field(default_factory=list)
    trace: List[SearchStepTrace] = field(default_factory=list)


def _step_cost(orig_prob: float, new_prob: float, eps: float = 1e-6) -> float:
    """
    Non-negative cost of replacing orig_digit (prob orig_prob) with new_digit (prob new_prob).
    Defined as negative log-likelihood difference: max(0, log(orig_prob) - log(new_prob)).
    """
    diff = math.log(max(orig_prob, eps)) - math.log(max(new_prob, eps))
    return max(0.0, diff)


def build_candidate_actions(
    initial_grid: Grid,
    cell_probs: Dict[Cell, np.ndarray],
    max_uncertain_cells: int = 12,
    top_alts_per_cell: int = 3,
) -> Tuple[
    Dict[Cell, List[Tuple[int, float, float, float]]],
    Dict[Cell, float],
    List[Tuple[Cell, int, float]],
]:
    """
    Identify candidate uncertain cells and their top alternative digits with step costs.

    Returns:
      - cell_alternatives: {cell: [(alt_digit, step_cost, read_prob, alt_prob), ...]}
      - min_change_cost: {cell: minimum cost to change cell to any alternative}
      - least_confident_cells: sorted list of (cell, read_digit, top1_confidence)
    """
    given_cells: List[Tuple[Cell, int, float, float]] = []
    min_change_cost: Dict[Cell, float] = {}
    all_cell_alts: Dict[Cell, List[Tuple[int, float, float, float]]] = {}

    for (r, c), probs in cell_probs.items():
        read_digit = int(initial_grid[r][c])
        if read_digit == 0:
            continue
        p_arr = np.asarray(probs, dtype=np.float64)
        p_arr = np.clip(p_arr, 1e-8, 1.0)
        p_arr = p_arr / p_arr.sum()

        read_prob = float(p_arr[read_digit - 1])
        # Sort alternative digits 1..9 by descending probability
        alt_indices = [idx for idx in np.argsort(-p_arr) if (idx + 1) != read_digit]
        alts: List[Tuple[int, float, float, float]] = []
        for idx in alt_indices[:top_alts_per_cell]:
            alt_digit = int(idx + 1)
            alt_prob = float(p_arr[idx])
            cost = _step_cost(read_prob, alt_prob)
            alts.append((alt_digit, cost, read_prob, alt_prob))

        if alts:
            all_cell_alts[(r, c)] = alts
            min_change_cost[(r, c)] = min(a[1] for a in alts)
            margin = read_prob - alts[0][3]
            given_cells.append(((r, c), read_digit, read_prob, margin))

    # Rank cells by uncertainty (lowest read_prob / margin first)
    given_cells.sort(key=lambda x: (x[2], x[3]))
    least_confident = [(c, d, p) for (c, d, p, _) in given_cells]

    # Always include any cell involved in an initial duplicate conflict
    conflict_cells: Set[Cell] = set()
    for c1, c2, _ in find_duplicate_conflicts(initial_grid):
        conflict_cells.add(c1)
        conflict_cells.add(c2)

    selected_cells: Set[Cell] = set(conflict_cells)
    for cell, _, _, _ in given_cells:
        if len(selected_cells) >= max_uncertain_cells:
            break
        selected_cells.add(cell)

    filtered_alts = {
        cell: all_cell_alts[cell] for cell in selected_cells if cell in all_cell_alts
    }
    return filtered_alts, min_change_cost, least_confident


def apply_corrections(initial_grid: Grid, state: StateTuple) -> Grid:
    """Return a new 9x9 grid with the state's corrections applied."""
    grid = [row[:] for row in initial_grid]
    for (r, c), new_digit in state:
        grid[r][c] = new_digit
    return grid


def compute_conflict_heuristic(
    current_grid: Grid,
    min_change_cost: Dict[Cell, float],
    already_corrected: Set[Cell],
) -> Tuple[float, int]:
    """
    Compute the admissible A* heuristic and the number of duplicate-digit conflicts.

    Admissibility Proof:
    Every conflict (c1, c2, d) means peer cells c1 and c2 currently hold the same digit d.
    In any valid Sudoku grid, c1 != c2, so at least one of {c1, c2} MUST change its value.
    If neither c1 nor c2 has been corrected yet, changing at least one of them costs at
    least min(min_change_cost[c1], min_change_cost[c2]).
    By selecting a maximal set of vertex-disjoint conflict pairs, no single cell modification
    can resolve more than one pair in the disjoint set. Therefore, the sum of lower bounds
    across vertex-disjoint conflict pairs never overestimates the true remaining cost h*(s).
    """
    conflicts = find_duplicate_conflicts(current_grid)
    if not conflicts:
        return 0.0, 0

    # Score each conflict pair by its minimum resolution cost
    scored_conflicts: List[Tuple[float, Cell, Cell]] = []
    for c1, c2, _ in conflicts:
        # If a cell was already corrected, changing it again isn't a valid single-step
        # extension, or if we only consider uncorrected cells:
        cost1 = min_change_cost.get(c1, 0.0) if c1 not in already_corrected else float("inf")
        cost2 = min_change_cost.get(c2, 0.0) if c2 not in already_corrected else float("inf")
        lb = min(cost1, cost2)
        if math.isinf(lb):
            lb = 0.0
        scored_conflicts.append((lb, c1, c2))

    # Sort descending by lower bound to get the tightest admissible packing
    scored_conflicts.sort(key=lambda x: -x[0])

    used_cells: Set[Cell] = set()
    h_val = 0.0
    for lb, c1, c2 in scored_conflicts:
        if c1 not in used_cells and c2 not in used_cells:
            used_cells.add(c1)
            used_cells.add(c2)
            h_val += lb

    return h_val, len(conflicts)


def run_correction_search(
    initial_grid: Grid,
    cell_probs: Dict[Cell, np.ndarray],
    algorithm: str = "astar",
    max_corrections: int = 3,
    max_uncertain_cells: int = 12,
    top_alts_per_cell: int = 3,
    csp_use_fc: bool = True,
    csp_use_ac3: bool = True,
    require_unique: bool = False,
    max_expanded: int = 2000,
) -> SearchResult:
    """
    Search for the most likely set of up to `max_corrections` digit replacements
    that makes the Sudoku puzzle solvable by the CSP.

    Supported algorithms:
      - 'ucs': Uniform Cost Search, priority = g(s)
      - 'greedy': Greedy Best-First Search, priority = (num_conflicts, h(s), g(s))
      - 'astar': A* Search, priority = g(s) + h(s)
    """
    alg = algorithm.lower().strip()
    if alg not in ("ucs", "greedy", "astar", "a*"):
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    if alg == "a*":
        alg = "astar"

    cell_alts, min_change_cost, least_confident = build_candidate_actions(
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        max_uncertain_cells=max_uncertain_cells,
        top_alts_per_cell=top_alts_per_cell,
    )

    csp_solver = SudokuCSP(use_fc=csp_use_fc, use_ac3=csp_use_ac3, max_assignments=90)

    nodes_expanded = 0
    nodes_generated = 0
    csp_calls = 0
    total_csp_assignments = 0
    total_csp_backtracks = 0
    trace: List[SearchStepTrace] = []

    # Root state: empty tuple of corrections
    root_state: StateTuple = ()
    root_h, root_conflicts = compute_conflict_heuristic(
        initial_grid, min_change_cost, set()
    )

    def _priority(g: float, h: float, n_conf: int, depth: int) -> Tuple[float, ...]:
        if alg == "ucs":
            return (round(g, 9), depth, round(h, 9))
        elif alg == "greedy":
            return (float(n_conf), round(h, 9), round(g, 9), depth)
        else:  # astar
            return (round(g + h, 9), round(h, 9), depth, round(g, 9))

    tie_counter = 0
    open_heap: List[Tuple[Tuple[float, ...], int, float, float, int, StateTuple]] = []
    heapq.heappush(
        open_heap,
        (
            _priority(0.0, root_h, root_conflicts, 0),
            tie_counter,
            0.0,
            root_h,
            root_conflicts,
            root_state,
        ),
    )
    nodes_generated += 1
    best_g: Dict[StateTuple, float] = {root_state: 0.0}

    while open_heap and nodes_expanded < max_expanded:
        _, _, g_val, h_val, n_conf, state = heapq.heappop(open_heap)

        if g_val > best_g.get(state, float("inf")) + 1e-12:
            continue

        nodes_expanded += 1
        candidate_grid = apply_corrections(initial_grid, state)
        corrected_cells = {cell for cell, _ in state}

        # Check if AC-3 / direct conflict test passes, then run full CSP
        sol_grid: Optional[Grid] = None
        csp_stats = CSPStats()
        csp_tested = False

        if n_conf == 0:
            csp_tested = True
            csp_calls += 1
            sol_grid, csp_stats = csp_solver.solve(
                candidate_grid, require_unique=require_unique
            )
            total_csp_assignments += csp_stats.assignments
            total_csp_backtracks += csp_stats.backtracks
        else:
            # Even with direct conflicts, if AC-3 is enabled in CSP we can record that
            # AC-3 immediately rejects this grid due to duplicate givens.
            csp_stats.ac3_consistent = False
            csp_stats.message = f"AC-3 conflict: {n_conf} duplicate-digit conflict(s)."

        trace_corrs = [
            (cell, int(initial_grid[cell[0]][cell[1]]), new_d)
            for cell, new_d in state
        ]
        f_val = g_val + h_val if alg == "astar" else (g_val if alg == "ucs" else h_val)
        trace.append(
            SearchStepTrace(
                step=nodes_expanded,
                algorithm=alg.upper(),
                corrections=trace_corrs,
                g_cost=g_val,
                h_cost=h_val,
                f_cost=f_val,
                num_conflicts=n_conf,
                csp_tested=csp_tested,
                csp_solved=sol_grid is not None,
                csp_backtracks=csp_stats.backtracks,
                csp_assignments=csp_stats.assignments,
                note=csp_stats.message,
            )
        )

        if sol_grid is not None:
            # Build detailed correction log
            correction_entries: List[CorrectionEntry] = []
            for cell, new_d in state:
                orig_d = int(initial_grid[cell[0]][cell[1]])
                alts = cell_alts[cell]
                matched = [a for a in alts if a[0] == new_d]
                if matched:
                    _, c_cost, r_prob, a_prob = matched[0]
                else:
                    c_cost, r_prob, a_prob = 0.0, 0.0, 0.0
                correction_entries.append(
                    CorrectionEntry(
                        cell=cell,
                        read_digit=orig_d,
                        new_digit=new_d,
                        cost=c_cost,
                        read_prob=r_prob,
                        new_prob=a_prob,
                    )
                )

            return SearchResult(
                solved=True,
                algorithm=alg.upper(),
                corrected_grid=candidate_grid,
                solution_grid=sol_grid,
                corrections=correction_entries,
                total_cost=g_val,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                csp_calls=csp_calls,
                total_csp_assignments=total_csp_assignments,
                total_csp_backtracks=total_csp_backtracks,
                message=f"Solved with {len(correction_entries)} correction(s).",
                least_confident_cells=least_confident[:10],
                trace=trace,
            )

        # Expand children if within correction budget
        if len(state) >= max_corrections:
            continue

        for cell, alts in cell_alts.items():
            if cell in corrected_cells:
                continue
            for alt_digit, step_cost, _, _ in alts:
                new_state: StateTuple = tuple(
                    sorted(list(state) + [(cell, alt_digit)], key=lambda x: x[0])
                )
                new_g = g_val + step_cost
                if new_g + 1e-12 >= best_g.get(new_state, float("inf")):
                    continue

                best_g[new_state] = new_g
                next_grid = apply_corrections(initial_grid, new_state)
                next_corrected = corrected_cells | {cell}
                new_h, new_conf = compute_conflict_heuristic(
                    next_grid, min_change_cost, next_corrected
                )
                tie_counter += 1
                nodes_generated += 1
                heapq.heappush(
                    open_heap,
                    (
                        _priority(new_g, new_h, new_conf, len(new_state)),
                        tie_counter,
                        new_g,
                        new_h,
                        new_conf,
                        new_state,
                    ),
                )

    return SearchResult(
        solved=False,
        algorithm=alg.upper(),
        corrected_grid=None,
        solution_grid=None,
        corrections=[],
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        csp_calls=csp_calls,
        total_csp_assignments=total_csp_assignments,
        total_csp_backtracks=total_csp_backtracks,
        message=(
            f"No solution: unreadable puzzle (no valid Sudoku found within "
            f"max_corrections={max_corrections})."
        ),
        least_confident_cells=least_confident[:10],
        trace=trace,
    )


def run_small_known_example() -> Dict[str, SearchResult]:
    """
    Small deterministic example with a known exact answer (Checklist Deliverable #5).

    Scenario:
    Matches the PDF example on Page 14:
    "The network reads a 7 as a 1 (52% vs 45%). Two 1s appear in the same row, so the
    puzzle looks broken. The system changes that cell to 7 (cheap, since the network
    was nearly as sure about 7) and the puzzle becomes solvable."
    We also include a distractor uncertain cell (0, 2) where changing 4 -> 2 is slightly
    cheaper (cost 0.10 vs 0.144), so UCS tries the non-conflict cell first while A* and
    Greedy immediately jump to the conflicting cell (0, 1)!
    """
    base_puzzle: Grid = [
        [5, 7, 4, 0, 7, 0, 0, 0, 0],  # Wait: let's use a known valid 35-given puzzle!
    ]
    # Valid complete Sudoku grid:
    valid_solution: Grid = [
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
    # Create a puzzle by keeping 36 givens
    mask = [
        [1, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 6, 0],
        [1, 0, 0, 0, 6, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 2, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1],
    ]
    puzzle: Grid = [
        [valid_solution[r][c] if mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]

    # Inject the exact PDF example misread at cell (1, 1): true digit is 7, read as 1 (52% vs 45%)
    # Notice cell (1, 3) in the same row already has a given 1! So two 1s appear in row 1.
    corrupted_puzzle = [row[:] for row in puzzle]
    corrupted_puzzle[1][1] = 1  # Misread 7 -> 1

    # Construct synthetic calibrated probabilities for all given cells
    cell_probs: Dict[Cell, np.ndarray] = {}
    for r in range(9):
        for c in range(9):
            val = corrupted_puzzle[r][c]
            if val == 0:
                continue
            probs = np.full(9, 0.001, dtype=np.float64)
            probs[val - 1] = 0.992
            probs /= probs.sum()
            cell_probs[(r, c)] = probs

    # Distractor uncertain cell at (0, 2) where true value is 4 (51% vs 47% for 2, cost ~0.081)
    p_distractor = np.full(9, 0.0028, dtype=np.float64)
    p_distractor[4 - 1] = 0.51
    p_distractor[2 - 1] = 0.47
    p_distractor /= p_distractor.sum()
    cell_probs[(0, 2)] = p_distractor

    # Conflicting misread cell at (1, 1): read as 1 (52%), true digit 7 (45%), cost = log(0.52/0.45) = 0.1446
    p_misread = np.full(9, 0.0042, dtype=np.float64)
    p_misread[1 - 1] = 0.52
    p_misread[7 - 1] = 0.45
    p_misread /= p_misread.sum()
    cell_probs[(1, 1)] = p_misread

    results: Dict[str, SearchResult] = {}
    for alg in ("ucs", "greedy", "astar"):
        results[alg] = run_correction_search(
            initial_grid=corrupted_puzzle,
            cell_probs=cell_probs,
            algorithm=alg,
            max_corrections=3,
        )
    return results


if __name__ == "__main__":
    print("Running Small Deterministic Search Example with Known Answer:")
    res_map = run_small_known_example()
    for alg_key, r in res_map.items():
        corrs = [(c.cell, c.read_digit, c.new_digit, round(c.cost, 4)) for c in r.corrections]
        print(
            f"  {alg_key.upper():<6} | solved={r.solved} | "
            f"nodes_expanded={r.nodes_expanded} | nodes_generated={r.nodes_generated} | "
            f"cost={r.total_cost:.4f} | corrections={corrs}"
        )

