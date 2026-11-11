# P7: Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

An end-to-end neuro-symbolic system that reads a $9 \times 9$ Sudoku puzzle of handwritten digits (MNIST 1–9), detects conflicts caused by optical misreads using **Arc Consistency (AC-3)**, searches over the most likely digit corrections using **Uniform Cost Search (UCS)**, **Greedy Best-First Search**, and **A\*** guided by **Temperature-Scaled Neural Network probabilities**, and solves/verifies each candidate puzzle using a switchable **Constraint Satisfaction Problem (CSP)** solver (**Backtracking**, **Forward Checking**, and **AC-3**).

---

## 1. Repository Structure (Deliverable #1)

```text
e:\ISP\
├── data/                       # Data module: MNIST downloader, preprocessor, Sudoku mosaic generator
│   ├── __init__.py
│   ├── mnist_loader.py         # Downloads 4 MNIST .gz files, scales to [0,1], splits 50k/10k/10k, drops 0
│   └── sudoku_generator.py     # Seeded puzzle generator (unique solution) + hard-digit mosaic injector
├── nn/                         # Neural Network module: MLP + Temperature Scaling & LogReg baseline
│   ├── __init__.py
│   └── digit_classifier.py     # CalibratedDigitClassifier, ECE/NLL/Brier metrics, confusion matrix
├── search/                     # Search module: UCS, Greedy Best-First, and A* with conflict heuristic
│   ├── __init__.py
│   └── correction_search.py    # Graph search with counters (nodes_expanded, nodes_generated) & small example
├── csp/                        # CSP module: Switchable Backtracking (BT), Forward Checking (FC), and AC-3
│   ├── __init__.py
│   └── sudoku_csp.py           # 81-variable Sudoku CSP with counters (assignments, backtracks) & uniqueness check
├── integration/                # Integration module: Connects Brightness Rule + NN + Search + CSP
│   ├── __init__.py
│   └── pipeline.py             # SudokuReaderSolverPipeline (nn_guided, no_correction, random_correction)
├── eval/                       # Evaluation module: Runs 3 experiments across 5 seeds (100 puzzles/level)
│   ├── __init__.py
│   └── run_experiments.py      # Produces mean ± std tables and publication plots in results/
├── tests/                      # Test suite: 6 normal, 4 edge, 3 impossible cases + small known-answer check
│   ├── __init__.py
│   └── test_all.py
├── make_data.py                # Reproducible data download, preprocessing, and model training script
├── run_demo.py                 # Integrated end-to-end demo script and 4-panel cooperation visualizer
├── DATA_DICTIONARY.md          # Complete column/field data dictionary (real vs. generated)
├── REPORT.md                   # Comprehensive project report (<= 12 pages)
├── PRESENTATION_AND_DEMO.md    # 10-minute presentation slides + 5-minute live demo script
├── CONTRIBUTION_AND_GITLOG.md  # Individual contribution statement (<= 200 words) + git log
└── requirements.txt            # Pinned Python package versions
```

---

## 2. Environment Setup (Deliverable #8)

### Prerequisites
- Python 3.10+ (tested with Python 3.14.6 on Windows)

### Create Virtual Environment and Install Pinned Requirements
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Exact Run Commands

### Step 1: Download & Preprocess Data and Train/Calibrate Classifier (`make_data.py`)
Downloads the 4 MNIST `.gz` files into `data/raw/`, splits the 60,000 training set into 50,000 train / 10,000 validation (`seed=42`), keeps the 10,000 test set untouched, drops digit `0` (retaining `1..9`), scales pixels to `[0, 1]`, trains the calibrated Neural Network + Logistic Regression baseline, and generates sample $9 \times 9$ Sudoku mosaics:
```powershell
.\.venv\Scripts\python.exe make_data.py --seed 42
```

### Step 2: Run the Integrated System Demo (`run_demo.py`)
Runs the small known-answer search test, reads a $9 \times 9$ handwritten Sudoku mosaic with 2 injected hard misreads, prints the initial reading and AC-3 conflict detection, displays the step-by-step A* search trace and correction log, compares CSP pruning (`BT` vs `BT+FC` vs `BT+FC+AC3`), compares the system with NN correction `ON` vs `OFF`, demonstrates an impossible/unreadable puzzle case, and saves the 4-panel cooperation plot to `results/demo_cooperation.png`:
```powershell
.\.venv\Scripts\python.exe run_demo.py
```

### Step 3: Run the Test Suite (`tests/test_all.py`)
Executes 13 unit and integration tests covering **6 normal cases**, **4 edge cases**, and **3 impossible cases** (verifying clear `"No solution: unreadable puzzle"` messages and least-confident cell reporting):
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_all.py
```

### Step 4: Reproduce All Three Experiments Across 5 Seeds (`eval/run_experiments.py`)
Runs all 3 required experiments across 5 seeds (`42, 101, 202, 303, 404`, 20 puzzles/seed = **100 puzzles per misread/given level**), saves mean ± std results to `results/experiment_results.json`, and generates all plots (`fig_nn_vs_baseline.png`, `fig_exp1_search.png`, `fig_exp2_csp.png`, `fig_exp3_ablation.png`):
```powershell
.\.venv\Scripts\python.exe -m eval.run_experiments
```

