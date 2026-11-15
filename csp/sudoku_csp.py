"""
Sudoku Constraint Satisfaction Problem (CSP) Solver.

Implements:
- 81 variables (cells (r, c) for r, c in 0..8), domains 1..9 (givens fixed).
- All-different constraints across rows, columns, and 3x3 boxes (1,620 directed arcs).
- Switchable components:
  1. Backtracking (BT)
  2. Forward Checking (FC)
  3. Arc Consistency (AC-3)
- Detailed counters: assignments, backtracks, arc_revisions, values_pruned.
- Uniqueness checking (exact solution count up to a limit).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
Grid = List[List[int]]
Domains = Dict[Cell, Set[int]]


def _build_peers() -> Tuple[List[Cell], Dict[Cell, List[Cell]], List[Tuple[Cell, Cell]]]:
    """Precompute all 81 cells, their 20 peers each, and all 1,620 directed arcs."""
    cells: List[Cell] = [(r, c) for r in range(9) for c in range(9)]
    peers: Dict[Cell, List[Cell]] = {}
    arcs: List[Tuple[Cell, Cell]] = []

    for r, c in cells:
        cell_peers: Set[Cell] = set()
        for k in range(9):
            if k != c:
                cell_peers.add((r, k))
            if k != r:
                cell_peers.add((k, c))
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if (nr, nc) != (r, c):
                    cell_peers.add((nr, nc))
        sorted_peers = sorted(cell_peers)
        peers[(r, c)] = sorted_peers
        for p in sorted_peers:
            arcs.append(((r, c), p))

    return cells, peers, arcs


ALL_CELLS, PEERS, ALL_ARCS = _build_peers()


@dataclass
class CSPStats:
    """Counters and diagnostic metrics for a CSP run."""

    assignments: int = 0
    backtracks: int = 0
    arc_revisions: int = 0
    values_pruned: int = 0
    ac3_consistent: bool = True
    solved: bool = False
    unique: Optional[bool] = None
    solution_count: int = 0
    message: str = ""
    initial_domain_sizes: int = 0
    post_ac3_domain_sizes: int = 0


def init_domains(grid: Grid) -> Domains:
    """Initialize variable domains from a 9x9 grid (0 = blank, 1..9 = given)."""
    domains: Domains = {}
    for r in range(9):
        for c in range(9):
            val = int(grid[r][c])
            if 1 <= val <= 9:
                domains[(r, c)] = {val}
            else:
                domains[(r, c)] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    return domains


def find_duplicate_conflicts(grid: Grid) -> List[Tuple[Cell, Cell, int]]:
    """
    Return all pairs of peer cells (c1, c2, digit) with c1 < c2 that have the
    same non-zero given digit in the same row, column, or 3x3 box.
    """
    conflicts: List[Tuple[Cell, Cell, int]] = []
    for r, c in ALL_CELLS:
        val = int(grid[r][c])
        if val == 0:
            continue
        for pr, pc in PEERS[(r, c)]:
            if (r, c) < (pr, pc) and int(grid[pr][pc]) == val:
                conflicts.append(((r, c), (pr, pc), val))
    return conflicts


def ac3(
    domains: Domains,
    stats: Optional[CSPStats] = None,
    initial_arcs: Optional[List[Tuple[Cell, Cell]]] = None,
) -> bool:
    """
    Enforce Arc Consistency (AC-3) on the given domains in-place.

    If two peer cells have identical singleton domains (e.g. two equal givens
    in the same row/col/box), revising their arc empties one domain and AC-3
    immediately returns False.
    """
    queue = deque(initial_arcs if initial_arcs is not None else ALL_ARCS)
    in_queue = set(queue)

    while queue:
        xi, xj = queue.popleft()
        in_queue.discard((xi, xj))
        if stats is not None:
            stats.arc_revisions += 1

        if _revise(domains, xi, xj, stats):
            if len(domains[xi]) == 0:
                if stats is not None:
                    stats.ac3_consistent = False
                return False
            for xk in PEERS[xi]:
                if xk != xj and (xk, xi) not in in_queue:
                    queue.append((xk, xi))
                    in_queue.add((xk, xi))
    return True


def _revise(
    domains: Domains,
    xi: Cell,
    xj: Cell,
    stats: Optional[CSPStats] = None,
) -> bool:
    """
    Revise domain of xi with respect to xj under the inequality constraint xi != xj.
    A value v in D(xi) has no support in D(xj) iff D(xj) == {v}.
    """
    dj = domains[xj]
    if len(dj) != 1:
        return False
    sole_val = next(iter(dj))
    if sole_val in domains[xi]:
        domains[xi].remove(sole_val)
        if stats is not None:
            stats.values_pruned += 1
        return True
    return False


def is_valid_complete_grid(grid: Grid) -> bool:
    """Verify that a 9x9 grid is completely filled with 1..9 and satisfies all constraints."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    for r, c in ALL_CELLS:
        val = int(grid[r][c])
        if val < 1 or val > 9:
            return False
        for pr, pc in PEERS[(r, c)]:
            if int(grid[pr][pc]) == val:
                return False
    return True


class SudokuCSP:
    """
    Configurable Sudoku CSP solver supporting Backtracking (BT),
    Forward Checking (FC), and Arc Consistency (AC-3) as switchable parts.
    """

    def __init__(
        self,
        use_fc: bool = True,
        use_ac3: bool = True,
        maintain_ac3: Optional[bool] = None,
        max_assignments: int = 500_000,
    ) -> None:
        self.use_fc = use_fc
        self.use_ac3 = use_ac3
        self.maintain_ac3 = use_ac3 if maintain_ac3 is None else maintain_ac3
        self.max_assignments = max_assignments

    def solve(
        self,
        grid: Grid,
        require_unique: bool = False,
    ) -> Tuple[Optional[Grid], CSPStats]:
        """
        Solve the Sudoku puzzle and return (solution_grid_or_None, stats).

        If require_unique is True, checks that the puzzle has *exactly* one solution.
        """
        stats = CSPStats()
        domains = init_domains(grid)
        stats.initial_domain_sizes = sum(len(d) for d in domains.values())

        # Step 1: Optional AC-3 preprocessing
        if self.use_ac3:
            consistent = ac3(domains, stats)
            stats.post_ac3_domain_sizes = sum(len(d) for d in domains.values())
            if not consistent:
                stats.solved = False
                stats.message = "No solution: AC-3 detected domain wipeout (conflict)."
                return None, stats
        else:
            stats.post_ac3_domain_sizes = stats.initial_domain_sizes
            if self.use_fc:
                # 1-step Forward Checking from initial givens
                for r, c in ALL_CELLS:
                    val = int(grid[r][c])
                    if val != 0:
                        for pr, pc in PEERS[(r, c)]:
                            if int(grid[pr][pc]) == val:
                                stats.solved = False
                                stats.message = (
                                    f"No solution: conflicting givens at {(r, c)} and {(pr, pc)}."
                                )
                                return None, stats
                            if int(grid[pr][pc]) == 0 and val in domains[(pr, pc)]:
                                domains[(pr, pc)].remove(val)
                                stats.values_pruned += 1
                                if len(domains[(pr, pc)]) == 0:
                                    stats.solved = False
                                    stats.message = (
                                        f"No solution: Forward Checking emptied domain at {(pr, pc)}."
                                    )
                                    return None, stats
            else:
                # Pure Backtracking: check if initial givens directly conflict
                conflicts = find_duplicate_conflicts(grid)
                if conflicts:
                    c1, c2, v = conflicts[0]
                    stats.backtracks += 1
                    stats.solved = False
                    stats.message = f"No solution: duplicate given {v} at {c1} and {c2}."
                    return None, stats

        # Build initial assignment dictionary from givens (and AC-3 deduced singletons)
        assignment: Dict[Cell, int] = {}
        for cell in ALL_CELLS:
            val = int(grid[cell[0]][cell[1]])
            if val != 0:
                assignment[cell] = val
            elif self.use_ac3 and len(domains[cell]) == 1:
                only_val = next(iter(domains[cell]))
                if all(assignment.get(p) != only_val for p in PEERS[cell]):
                    assignment[cell] = only_val
                else:
                    stats.solved = False
                    stats.message = f"No solution: AC-3 singleton conflict at {cell}."
                    return None, stats

        solutions: List[Grid] = []
        target_solutions = 2 if require_unique else 1

        self._backtrack(
            assignment=assignment,
            domains=domains,
            stats=stats,
            solutions=solutions,
            max_solutions=target_solutions,
        )

        if not solutions:
            stats.solved = False
            if not stats.message:
                stats.message = "No solution: search exhausted without valid assignment."
            return None, stats

        stats.solution_count = len(solutions)
        if require_unique:
            stats.unique = len(solutions) == 1
            if not stats.unique:
                stats.solved = False
                stats.message = "No solution: puzzle has multiple solutions (not unique)."
                return None, stats

        stats.solved = True
        stats.message = "Solved successfully."
        return solutions[0], stats

    def _select_unassigned_variable(
        self,
        assignment: Dict[Cell, int],
        domains: Domains,
    ) -> Cell:
        """
        Select next unassigned variable:
        - When FC or AC-3 is active, use Minimum Remaining Values (MRV) on the shrunk domains.
        - For pure BT, select in fixed row-major order.
        """
        if self.use_fc or self.use_ac3:
            best_cell: Optional[Cell] = None
            best_len = 10
            for cell in ALL_CELLS:
                if cell not in assignment:
                    d_len = len(domains[cell])
                    if d_len < best_len:
                        best_len = d_len
                        best_cell = cell
                        if best_len <= 1:
                            break
            assert best_cell is not None
            return best_cell
        else:
            for cell in ALL_CELLS:
                if cell not in assignment:
                    return cell
            raise RuntimeError("No unassigned cell found")

    def _is_consistent(self, cell: Cell, val: int, assignment: Dict[Cell, int]) -> bool:
        """Check whether assigning cell = val conflicts with any already-assigned peer."""
        for peer in PEERS[cell]:
            if assignment.get(peer) == val:
                return False
        return True

    def _backtrack(
        self,
        assignment: Dict[Cell, int],
        domains: Domains,
        stats: CSPStats,
        solutions: List[Grid],
        max_solutions: int,
    ) -> bool:
        """Recursive backtracking search with optional FC and AC-3."""
        if len(assignment) == 81:
            sol = [[assignment[(r, c)] for c in range(9)] for r in range(9)]
            solutions.append(sol)
            return len(solutions) >= max_solutions

        if stats.assignments >= self.max_assignments:
            stats.message = "No solution: assignment budget exceeded."
            return True

        var = self._select_unassigned_variable(assignment, domains)
        candidate_values = sorted(domains[var])

        for val in candidate_values:
            stats.assignments += 1
            if not self._is_consistent(var, val, assignment):
                continue

            assignment[var] = val

            if self.use_fc:
                pruned: List[Tuple[Cell, int]] = []
                wipeout = False
                newly_singleton: List[Cell] = []
                for peer in PEERS[var]:
                    if peer not in assignment and val in domains[peer]:
                        domains[peer].remove(val)
                        pruned.append((peer, val))
                        stats.values_pruned += 1
                        if len(domains[peer]) == 0:
                            wipeout = True
                            break
                        elif len(domains[peer]) == 1:
                            newly_singleton.append(peer)

                if not wipeout and self.use_ac3 and self.maintain_ac3 and newly_singleton:
                    saved_domains = {c: set(d) for c, d in domains.items()}
                    saved_domains[var] = {val}
                    arcs = [
                        (nb, s_cell)
                        for s_cell in newly_singleton
                        for nb in PEERS[s_cell]
                        if nb not in assignment
                    ]
                    if ac3(saved_domains, stats, initial_arcs=arcs):
                        if self._backtrack(
                            assignment, saved_domains, stats, solutions, max_solutions
                        ):
                            for peer, pval in pruned:
                                domains[peer].add(pval)
                            return True
                    for peer, pval in pruned:
                        domains[peer].add(pval)
                elif not wipeout:
                    old_var_dom = domains[var]
                    domains[var] = {val}
                    if self._backtrack(assignment, domains, stats, solutions, max_solutions):
                        domains[var] = old_var_dom
                        for peer, pval in pruned:
                            domains[peer].add(pval)
                        return True
                    domains[var] = old_var_dom
                    for peer, pval in pruned:
                        domains[peer].add(pval)
                else:
                    for peer, pval in pruned:
                        domains[peer].add(pval)
            else:
                if self._backtrack(assignment, domains, stats, solutions, max_solutions):
                    return True

            del assignment[var]

        stats.backtracks += 1
        return False


def solve_sudoku(
    grid: Grid,
    use_fc: bool = True,
    use_ac3: bool = True,
    require_unique: bool = False,
) -> Tuple[Optional[Grid], CSPStats]:
    """Convenience function to solve a Sudoku grid with configurable FC and AC-3."""
    solver = SudokuCSP(use_fc=use_fc, use_ac3=use_ac3)
    return solver.solve(grid, require_unique=require_unique)


def count_solutions(grid: Grid, max_solutions: int = 2) -> int:
    """Count the number of valid solutions to a Sudoku grid, capped at max_solutions."""
    stats = CSPStats()
    domains = init_domains(grid)
    if not ac3(domains, stats):
        return 0
    assignment: Dict[Cell, int] = {}
    for cell in ALL_CELLS:
        if int(grid[cell[0]][cell[1]]) != 0:
            assignment[cell] = int(grid[cell[0]][cell[1]])
    solver = SudokuCSP(use_fc=True, use_ac3=True)
    solutions: List[Grid] = []
    solver._backtrack(assignment, domains, stats, solutions, max_solutions=max_solutions)
    return len(solutions)


if __name__ == "__main__":
    sample_puzzle: Grid = [
        [5, 3, 4, 0, 7, 0, 0, 1, 0],
        [6, 7, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 3, 0],
        [0, 4, 0, 0, 8, 0, 1, 7, 9],
    ]
    print("CSP Switchable Components Verification (35-Given Puzzle):")
    for label, fc_flag, ac3_flag in [
        ("BT (Pure Backtracking)", False, False),
        ("BT + FC (Forward Checking)", True, False),
        ("BT + AC-3 (Arc Consistency)", False, True),
        ("BT + FC + AC-3 (Full)", True, True),
    ]:
        sol, st = solve_sudoku(sample_puzzle, use_fc=fc_flag, use_ac3=ac3_flag)
        print(
            f"  {label:<30} | solved={st.solved} | assignments={st.assignments:4d} | "
            f"backtracks={st.backtracks:3d} | pruned={st.values_pruned:3d}"
        )

