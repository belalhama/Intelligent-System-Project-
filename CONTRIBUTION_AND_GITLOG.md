# Individual Contribution Statement & Git Log
**Project P7: Handwritten Sudoku Reader and Solver**

---

## 1. Individual Contribution Statement (<= 200 Words Each)

### Option A — Single-Author / Individual Submission (148 words)
I designed, implemented, and evaluated the complete neuro-symbolic Handwritten Sudoku Reader and Solver (`P7`). Specifically, I built the reproducible data pipeline (`make_data.py`, `data/mnist_loader.py`, `data/sudoku_generator.py`) that downloads MNIST, splits the 60,000 training images into 50,000 train and 10,000 validation examples (`seed=42`), removes digit `0`, applies the brightness blank-detection rule, generates unique-solution Sudoku puzzles, and injects controlled ambiguous test digits. I implemented and trained the 2-layer MLP classifier with post-hoc Temperature Scaling ($T=1.765$, reducing ECE from `0.0107` to `0.0028`) and compared it against the Logistic Regression baseline (`nn/digit_classifier.py`). I developed the switchable Sudoku CSP solver (`csp/sudoku_csp.py`) with Backtracking, Forward Checking, and AC-3, and the correction search module (`search/correction_search.py`) supporting UCS, Greedy Best-First, and A\* with a vertex-disjoint conflict heuristic. Finally, I built `integration/pipeline.py`, `run_demo.py`, the 13-test suite (`tests/test_all.py`), the 5-seed experimental evaluation (`eval/run_experiments.py`), and the technical report.

---

### Option B — Multi-Member Team Template (Each <= 200 Words, if submitting as a group)
- **Member 1 (Data & Neural Network — 86 words)**: Implemented `make_data.py`, `data/mnist_loader.py`, and `DATA_DICTIONARY.md` to reproducibly download MNIST, filter digits 1–9, and detect blank cells via pixel brightness. Developed `nn/digit_classifier.py` to train the 2-layer MLP (`784->256->128->9`) and Multinomial Logistic Regression baseline, implemented validation-set Temperature Scaling ($T=1.765$), and generated the confusion matrix and reliability plots (`fig_nn_vs_baseline.png`).
- **Member 2 (CSP & Puzzle Generator — 84 words)**: Implemented `csp/sudoku_csp.py` featuring 81 variables, 1,620 binary inequality arcs, and switchable Backtracking (`BT`), Forward Checking (`FC`), and Arc Consistency (`AC-3`) with assignment/backtrack/pruning counters and uniqueness verification. Built `data/sudoku_generator.py` to generate unique-solution Sudoku puzzles (25, 30, 35 givens) and $9 \times 9$ handwritten mosaics with documented hard-digit injection, and ran Experiment 2 (`fig_exp2_csp.png`).
- **Member 3 (Search, Integration, Demo & Testing — 92 words)**: Implemented `search/correction_search.py` with Uniform Cost Search (UCS), Greedy Best-First Search, and A\* using the admissible vertex-disjoint duplicate-conflict heuristic, verified on the small known-answer example. Built the end-to-end neuro-symbolic pipeline (`integration/pipeline.py`), the integrated visual demo (`run_demo.py`), the 13-case unit/integration test suite (`tests/test_all.py` covering 6 normal, 4 edge, and 3 impossible cases), and ran Experiments 1 and 3 (`eval/run_experiments.py`).

---

## 2. Repository Git Log (`git log --stat`)

```text
commit b9a346dd406c1bdf8c92384b3fcf306453be7ba2
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(eval,tests,docs): add 5-seed experiments, 13-case pytest suite, technical report, and presentation

 PRESENTATION_AND_DEMO.md            |  102 +++
 README.md                           |   82 ++
 REPORT.md                           |  188 ++++
 eval/__init__.py                    |    6 +
 eval/run_experiments.py             |  613 +++++++++++++
 results/demo_cooperation.png        |  Bin 0 -> 296431 bytes
 results/experiment_results.json     | 1724 +++++++++++++++++++++++++++++++++++
 results/fig_exp1_search.png         |  Bin 0 -> 188171 bytes
 results/fig_exp2_csp.png            |  Bin 0 -> 82266 bytes
 results/fig_exp3_ablation.png       |  Bin 0 -> 79567 bytes
 results/fig_nn_vs_baseline.png      |  Bin 0 -> 207572 bytes
 results/nn_vs_baseline_metrics.json |  225 +++++
 tests/__init__.py                   |    2 +
 tests/test_all.py                   |  366 ++++++++
 14 files changed, 3308 insertions(+)

commit 722f207d81ead515ad0a90ae37e908ba626bab6d
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(integration): add end-to-end neuro-symbolic pipeline and 4-panel cooperation demo

 integration/__init__.py |  14 ++
 integration/pipeline.py | 293 +++++++++++++++++++++++++++++++
 run_demo.py             | 459 ++++++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 766 insertions(+)

commit 741e83e0b5b377ec6d47ed336cf5b4dc958a90df
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(search): implement UCS, Greedy, and A* correction search with admissible conflict heuristic

 search/__init__.py          |  20 ++
 search/correction_search.py | 520 ++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 540 insertions(+)

commit c4962c67e6f9d2d31786196f6bcf1973489bb052
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(csp): implement switchable Sudoku CSP with Backtracking, Forward Checking, and AC-3

 csp/__init__.py   |  22 +++
 csp/sudoku_csp.py | 433 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 455 insertions(+)

commit 42bf880fea53478f60b040e5faf9c20f217e407d
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(nn): implement calibrated MLP digit classifier with temperature scaling and LogReg baseline

 nn/__init__.py         |  18 +++
 nn/digit_classifier.py | 298 +++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 316 insertions(+)

commit c99b13638726f8129a8c99c1c5fe44b8acae2903
Author: Student Developer <student@university.edu>
Date:   Wed Nov 11 20:47:02 2026 +0200

    feat(data): add reproducible MNIST 1-9 loader, seeded Sudoku mosaic generator, and data dictionary

 .gitignore                                  |   9 +
 DATA_DICTIONARY.md                          |  62 ++++
 data/__init__.py                            |  30 ++
 data/mnist_loader.py                        | 235 ++++++++++++++++
 data/processed/dataset_summary.json         |  46 +++
 data/processed/sample_puzzles_metadata.json | 302 ++++++++++++++++++++
 data/sudoku_generator.py                    | 423 ++++++++++++++++++++++++++++
 make_data.py                                | 117 ++++++++
 requirements.txt                            |   8 +
 9 files changed, 1232 insertions(+)
```
