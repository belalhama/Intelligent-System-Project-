# Data Dictionary — P7: Handwritten Sudoku Reader and Solver

## 1. Source Dataset Overview
- **Dataset Name**: MNIST Database of Handwritten Digits (LeCun, Cortes, & Burges)
- **Primary Source URL**: `http://yann.lecun.com/exdb/mnist/` (Mirror: `https://storage.googleapis.com/cvdf-datasets/mnist/`)
- **Licence & Access**: Public domain / CC BY-SA 3.0; no registration required; ~11 MiB across 4 `.gz` IDX files stored locally in `data/raw/`.
- **Split & Preprocessing Rule (`make_data.py`, fixed `seed=42`)**:
  1. Raw training set (60,000 images) is split reproducibly using `np.random.default_rng(42).permutation(60000)` into **50,000 training** and **10,000 validation** examples.
  2. The **10,000 raw test set** is kept **100% untouched** (different writers from training/validation).
  3. All examples with label `0` are dropped (`y != 0`), retaining digits **1 through 9**:
     - Filtered Training Set (`X_train`, `y_train`): **45,073** images
     - Filtered Validation Set (`X_val`, `y_val`): **9,004** images
     - Filtered Test Set (`X_test`, `y_test`): **9,020** images
  4. Pixel values are scaled from `uint8` `[0, 255]` to `float32` `[0.0, 1.0]` by dividing by `255.0`.

---

## 2. Field-by-Field Data Dictionary

### A. Processed MNIST Dataset (`data/processed/mnist_1to9.npz`)

| Field / Column Name | Shape & Dtype | Range / Domain | Real vs. Generated | How Made / Preprocessing Rule |
| :--- | :--- | :--- | :--- | :--- |
| `X_train` | `(45073, 28, 28)` `float32` | `[0.0, 1.0]` | **Real** | Real handwritten digit images from the 50,000-image training split (`y != 0`), normalized via `pixel / 255.0`. |
| `y_train` | `(45073,)` `int64` | `{1, 2, ..., 9}` | **Real** | Real ground-truth human writer labels from MNIST (`0` dropped). |
| `X_val` | `(9004, 28, 28)` `float32` | `[0.0, 1.0]` | **Real** | Real handwritten digit images from the 10,000-image validation split (`y != 0`), normalized via `pixel / 255.0`. Used for early stopping and Temperature Scaling ($T$) calibration. |
| `y_val` | `(9004,)` `int64` | `{1, 2, ..., 9}` | **Real** | Real ground-truth labels for the validation split (`0` dropped). |
| `X_test` | `(9020, 28, 28)` `float32` | `[0.0, 1.0]` | **Real** | Real handwritten digit images from the untouched 10,000-image MNIST test set (`y != 0`), normalized via `pixel / 255.0`. |
| `y_test` | `(9020,)` `int64` | `{1, 2, ..., 9}` | **Real** | Real ground-truth labels for the untouched MNIST test set (`0` dropped). |
| `seed` | `(1,)` `int64` | `42` | **Generated** | Fixed PRNG seed (`42`) governing the 50k/10k train/validation split. |

---

### B. Sudoku Puzzle & 9x9 Image Mosaic Dataset (`data/processed/sample_sudoku_mosaics.npz` & `SudokuMosaicInstance`)

| Field / Column Name | Shape & Dtype | Range / Domain | Real vs. Generated | How Made / Generation Rule |
| :--- | :--- | :--- | :--- | :--- |
| `puzzle_id` | `int` | `0 .. N-1` | **Generated** | Sequential integer identifier for each generated Sudoku puzzle instance. |
| `seed` | `int` | `int >= 0` | **Generated** | Deterministic seed (`base_seed + puzzle_id`) passed to `np.random.default_rng` to guarantee exact reproducibility. |
| `num_givens` | `int` | `{25, 30, 35}` | **Generated** | Number of non-blank clue cells in `puzzle_grid`. |
| `num_injected_misreads` | `int` | `{0, 1, 2, 3, 4}` | **Generated** | Controlled number of given cells assigned a hard/ambiguous test digit image. |
| `solution_grid` | `(9, 9)` `int64` | `{1, 2, ..., 9}` | **Generated** | Complete valid $9 \times 9$ Sudoku grid built by seeded randomized backtracking (`generate_full_grid`). |
| `puzzle_grid` | `(9, 9)` `int64` | `{0, 1, ..., 9}` | **Generated** | Derived from `solution_grid` by shuffling all 81 cell coordinates and removing clues (`0` = blank) one by one while verifying via CSP (`count_solutions == 1`) that the puzzle retains a **unique solution** until `num_givens` clues remain. |
| `cell_images` | `(9, 9, 28, 28)` `float32` | `[0.0, 1.0]` | **Hybrid (Real MNIST Test Digits + Generated Layout/Blends)** | 81 cell patches of size $28 \times 28$ (forming a $252 \times 252$ mosaic):<br>1. **Blank cells (`puzzle_grid[r,c] == 0`)**: Generated low-brightness background noise $\mathcal{U}(0, 0.004)$ with mean intensity $\approx 0.002 < 0.025$ (detected by `detect_blank_cell` brightness rule, not by the NN).<br>2. **Standard given cells**: Sampled directly from real held-out `X_test` images matching `true_digit = puzzle_grid[r,c]`.<br>3. **Injected hard cells (`k` cells)**: Constructed from two real held-out `X_test` images `img_true` (label $d_{\text{true}}$) and `img_conf` (label $d_{\text{confuser}}$, preferring a digit already present among the cell's row/col/box peers) via convex blending $\text{clip}((1-\alpha) x_{\text{true}} + \alpha x_{\text{conf}}, 0, 1)$ with $\alpha \in [0.42, 0.72]$ chosen so the NN's top-1 reading is $d_{\text{confuser}}$ and its #2 reading is $d_{\text{true}}$. |
| `injected_cells` | `List[InjectedMisreadInfo]` | Structured Dict/Object | **Generated** | Records `cell` `(r, c)`, `true_digit`, `intended_read_digit`, `alpha_blend`, `source_test_idx_true`, and `source_test_idx_confuser` for full auditability. |

---

### C. Pipeline, Search, and CSP Output Fields (`results/experiment_results.json` & Correction Log)

| Field / Column Name | Type | Range / Domain | Real vs. Generated | Definition |
| :--- | :--- | :--- | :--- | :--- |
| `cell` | `Tuple[int, int]` | `(0..8, 0..8)` | **Generated** | Row and column coordinates `(r, c)` of a corrected cell in the Correction Log. |
| `read_digit` | `int` | `{1, ..., 9}` | **Generated (Model Output)** | Initial top-1 digit predicted by the calibrated NN on `cell_images[r, c]`. |
| `new_digit` | `int` | `{1, ..., 9}` | **Generated (Search Output)** | Alternative digit selected by Search (UCS, Greedy, or A*) to replace `read_digit`. |
| `cost` | `float` | `[0.0, \infty)` | **Generated (Search Metric)** | Non-negative log-likelihood step cost: $\max(0, \log P(d_{\text{read}} \mid x) - \log P(d_{\text{new}} \mid x))$. |
| `nodes_expanded` | `int` | `>= 1` | **Generated (Counter)** | Number of search states popped from the priority queue and evaluated. |
| `nodes_generated` | `int` | `>= 1` | **Generated (Counter)** | Total number of search states pushed onto the priority queue. |
| `assignments` | `int` | `>= 0` | **Generated (Counter)** | Number of candidate variable-value assignments tested during CSP backtracking. |
| `backtracks` | `int` | `>= 0` | **Generated (Counter)** | Number of dead-end nodes where CSP search exhausted a variable's candidate values and backtracked. |
| `values_pruned` | `int` | `>= 0` | **Generated (Counter)** | Number of domain values eliminated by Forward Checking (FC) and Arc Consistency (AC-3). |

