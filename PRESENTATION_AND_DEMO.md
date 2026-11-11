# 10-Minute Presentation & 5-Minute Live Demo Script
**Project P7: Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)**

---

## Part 1: 10-Minute Presentation Deck (Slide-by-Slide Outline & Speaker Notes)

### Slide 1 — Title & Motivation (0:00 – 1:00)
- **Title**: Neuro-Symbolic Handwritten Sudoku Reader and Solver
- **Real-Life Application**: Educational math apps and camera-based puzzle solvers.
- **Core Challenge**: Even a 98%-accurate handwritten digit classifier has a $1 - 0.98^{30} \approx 45.5\%$ probability of misreading **at least one digit** across a 30-given Sudoku grid!
- **Key Insight**: A pure neural network ignores global Sudoku constraints, while a pure CSP fails immediately if a single given digit is misread. Combining **Calibrated NN + Informed Search + CSP** recovers the exact puzzle.

### Slide 2 — System Architecture & Cooperation Loop (1:00 – 2:30)
- **Stage 1 (Optical Reading)**:
  - Blank cells detected via a deterministic **Brightness Rule** ($\text{mean}(I) < 0.025$), avoiding unnecessary neural inference on empty squares.
  - Non-blank $28 \times 28$ cells classified into digits $1\dots 9$ by a **Temperature-Scaled Neural Network**.
- **Stage 2 (AC-3 Conflict Detection)**:
  - **AC-3** checks the initial reading; two equal givens in the same row, column, or $3 \times 3$ box empty a variable's domain immediately.
- **Stage 3 (Minimum-Cost Correction Search)**:
  - **UCS**, **Greedy Best-First**, or **A\*** searches over sets of up to 3 cell replacements ordered by negative log-likelihood step cost.
- **Stage 4 (CSP Verification & Solving)**:
  - **Backtracking + Forward Checking + AC-3** solves each candidate grid and optionally verifies solution uniqueness.

### Slide 3 — Data Pipeline & Documented Hard-Digit Rule (2:30 – 3:45)
- **Dataset**: MNIST (`train`: 50k $\to$ 45,073 digits 1–9; `val`: 10k $\to$ 9,004; untouched `test`: 10k $\to$ 9,020).
- **Seeded Puzzle Generator**: Builds full grids via randomized backtracking and removes clues while verifying `count_solutions(grid) == 1`.
- **Documented Hard-Digit Injection**: To systematically test $k \in \{0, 1, 2, 3, 4\}$ misreads across 5 seeds (100 puzzles per level), $k$ clue cells receive ambiguous held-out test digits (convex blends of two test images $(1-\alpha)x_{\text{true}} + \alpha x_{\text{confuser}}$ where the NN's top-1 prediction is the confuser digit and its #2 prediction is the true digit).

### Slide 4 — Neural Network & Temperature Scaling Calibration (3:45 – 5:00)
- **Architecture**: 3-layer MLP (`784 -> 256 -> 128 -> 9`, ReLU, Adam) vs. Multinomial Logistic Regression baseline.
- **Why Calibration Matters for Search**: Search step costs are $\Delta c = \log P(d_{\text{read}} \mid x) - \log P(d_{\text{new}} \mid x) \ge 0$. Uncalibrated networks are overconfident, distorting step costs.
- **Results on Untouched Test Set (9,020 images)**:
  - **MLP + Temperature Scaling ($T = 1.765$)**: **98.04% accuracy**, ECE reduced by **74%** (`0.0107 -> 0.0028`), NLL = `0.0643`.
  - **Logistic Regression Baseline**: **92.29% accuracy**, NLL = `0.2797`.

### Slide 5 — Search Formulation & Admissible A* Heuristic (5:00 – 6:30)
- **State**: Canonical sorted tuple of cell modifications $((r, c), d_{\text{new}})$.
- **Action**: Replace an uncertain given cell with one of its top alternative digits.
- **Step Cost**: $\log P(d_{\text{read}} \mid x) - \log P(d_{\text{new}} \mid x) \ge 0$ (never negative).
- **Admissible A\* Heuristic**:
  - For every duplicate-digit conflict $(c_1, c_2, d)$ in the same row, column, or box, at least one of $c_1$ or $c_2$ **must** change, costing at least $\min(\Delta_{\min}(c_1), \Delta_{\min}(c_2))$.
  - Summing over a maximal packing of **vertex-disjoint** conflict pairs guarantees $0 \le h(s) \le h^*(s)$ (zero overestimates across all tested instances).

### Slide 6 — CSP Design: Backtracking, Forward Checking, and AC-3 (6:30 – 7:45)
- **81 variables**, domains $\{1\dots 9\}$, 1,620 directed binary inequality arcs.
- **Switchable Architecture**:
  - **Pure BT**: Row-major backtracking checking consistency against assigned peers.
  - **BT + FC**: Prunes assigned values from unassigned peers and uses Minimum Remaining Values (MRV); backtracks immediately on domain wipeout.
  - **BT + FC + AC-3**: Runs full AC-3 preprocessing (detecting duplicate givens in 0 backtracks and cascading naked singles) and maintains arc consistency on newly created singletons during tree search.

### Slide 7 — Three Experimental Benchmarks Across 5 Seeds (7:45 – 9:15)
- **Experiment 1 (UCS vs. Greedy vs. A\*, 0–4 misreads, 100 puzzles/level)**:
  - A* achieves identical optimal recovery to UCS while expanding significantly fewer nodes by pruning branches that fail to resolve active duplicate conflicts.
- **Experiment 2 (BT vs. BT+FC vs. BT+FC+AC-3 on 25, 30, 35 givens)**:
  - Pure BT requires thousands of assignments and backtracks; BT+FC cuts assignments by >90%; BT+FC+AC-3 solves 30- and 35-given puzzles with near-zero backtracks.
- **Experiment 3 (NN-Guided vs. No Correction vs. Random Correction)**:
  - Without correction, any misread breaks the puzzle (0% recovery). Random correction fails on multi-misread puzzles. NN-guided correction recovers >95% of 1–3 misread puzzles.

### Slide 8 — Limitations & Conclusions (9:15 – 10:00)
- **Limitations**:
  - Synthesized $9 \times 9$ MNIST mosaics assume clean cell segmentation (real camera photos have perspective skew, grid-line bleed, and illumination gradients).
  - If a misread accidentally yields a *different* valid Sudoku, a solvability-only goal accepts it (mitigated by our `require_unique=True` extension).
- **Conclusion**: Neuro-symbolic integration turns an imperfect 98% classifier into a reliable puzzle solver with provable constraint satisfaction.

---

## Part 2: 5-Minute Live Demo Script (`python run_demo.py`)

### Minute 0:00 – 1:00: Run the Demo & Show Optical Reading (`[Part A]` – `[Part C]`)
1. Run command in terminal:
   ```powershell
   .\.venv\Scripts\python.exe run_demo.py
   ```
2. Point out **`[Part A]`**: On the small known-answer example (where a `7` is misread as `1` with 52% vs 45% probability and a distractor cell has a cheaper non-conflict change), **UCS expands 3 nodes** whereas **A\* and Greedy expand only 2 nodes**.
3. Point out **`[Part C]`**: On the 30-given $9 \times 9$ mosaic, the NN misreads two cells:
   - `Cell (0, 4)`: True digit is `2`, NN reads `1` ($P=59.1\%$) with `2` as second choice ($P=40.1\%$).
   - `Cell (6, 2)`: True digit is `6`, NN reads `8` ($P=35.8\%$) with `6` as second choice ($P=28.5\%$).

### Minute 1:00 – 2:30: Live Walkthrough of the Search Trace (`[Part D]`)
1. Highlight **`Initial AC-3 Consistent: False`** and the 3 duplicate conflicts detected immediately:
   - `((0, 4), (1, 3), 1)`, `((0, 4), (6, 4), 1)`, and `((6, 2), (6, 6), 8)`.
2. Walk through the **A\* Search Trace** line by line:
   - `Step 1`: $g=0.0000, h=0.4523, f=0.4523$ (3 conflicts $\to$ rejected by AC-3).
   - `Step 2`: Corrects `(0, 4): 1 -> 2` ($g=0.3875, h=0.0649, f=0.4523$, 1 conflict left).
   - `Step 3`: Corrects `(0, 4): 1 -> 2, (6, 2): 8 -> 6` ($g=0.6148, h=0.0000, f=0.6148$, 0 conflicts $\to$ **CSP Solved = True**!).
3. Compare **UCS (6 nodes expanded, 164 generated)** vs. **A\* (3 nodes expanded, 70 generated)**.

### Minute 2:30 – 3:45: CSP Pruning With and Without FC / AC-3 (`[Part E]`)
1. Point to **`[Part E]`** in the terminal output:
   - **Pure `BT`**: `5,710 assignments`, `605 backtracks`, `0 pruned values`.
   - **`BT+FC`**: `51 assignments`, `0 backtracks`, `408 pruned values`.
   - **`BT+FC+AC3`**: `0 tree-search assignments`, `0 backtracks` — initial AC-3 shrinks total domain size from `489 -> 81` (solving all 51 empty cells purely by constraint propagation!).

### Minute 3:45 – 5:00: System With NN ON vs. OFF, Impossible Case, & Visual Plot (`[Part F]` – `[Part G]`)
1. Point to **`[Part F]`**:
   - **NN Correction OFF**: Fails immediately (`Solved=False`, `"No solution: AC-3 detected domain wipeout"`).
   - **NN Correction ON**: Recovers both misreads (`Solved=True`, `Exact Recovery=True`).
2. Point to **`[Part G]`**: When 5 conflicts are injected (exceeding `max_corrections=3`), the system outputs:
   `No solution: unreadable puzzle (no valid Sudoku found within max_corrections=3)` and lists the least confident cells.
3. Open `results/demo_cooperation.png` to show the 4-panel visual summary of the mosaic, NN probabilities, search/CSP counters, and solved board.

