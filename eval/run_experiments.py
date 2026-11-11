"""
Three Core Experiments + Classifier Calibration Benchmark for P7 (Handwritten Sudoku).

Deliverable #10:
- Runs all three required experiments across 5 seeds (42, 101, 202, 303, 404)
  with 20 puzzles per seed (= 100 puzzles per setting).
- Computes Mean and Standard Deviation across the 5 seeds for every metric.
- Generates publication-quality matplotlib plots and Markdown/JSON tables in results/.

Experiments:
1. UCS vs Greedy vs A*: nodes expanded, nodes generated, and exact-solution recovery
   rate for 0, 1, 2, 3, 4 injected misreads (100 puzzles per misread level).
2. BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens (assignments,
   backtracks, pruned values, runtime).
3. NN-guided correction vs No correction vs Random correction (exact-solution recovery
   rate, solvability rate, nodes expanded).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / "results" / ".mplconfig"))
(ROOT_DIR / "results" / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csp.sudoku_csp import SudokuCSP
from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import (
    fast_generate_unique_puzzles,
    generate_puzzle_with_unique_solution,
    generate_sudoku_mosaic_dataset,
)
from integration.pipeline import SudokuReaderSolverPipeline, read_mosaic_grid
from nn.digit_classifier import load_or_train_models
from search.correction_search import (
    build_candidate_actions,
    compute_conflict_heuristic,
    run_correction_search,
)

SEEDS: List[int] = [42, 101, 202, 303, 404]


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "per_seed": [float(v) for v in vals],
    }


def plot_classifier_comparison(
    dataset,
    nn_clf,
    lr_clf,
    metrics: Dict[str, Any],
    results_dir: Path,
) -> None:
    """Plot 9x9 confusion matrix and reliability diagram (calibration curve)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: NN Confusion Matrix
    cm_nn = np.array(metrics["nn"].confusion_matrix)
    im0 = axes[0].imshow(cm_nn, cmap="Blues")
    axes[0].set_title(
        f"Calibrated NN Confusion Matrix\nAcc: {metrics['nn'].accuracy*100:.2f}% (T={metrics['nn'].temperature:.2f})"
    )
    axes[0].set_xticks(range(9))
    axes[0].set_yticks(range(9))
    axes[0].set_xticklabels(range(1, 10))
    axes[0].set_yticklabels(range(1, 10))
    axes[0].set_xlabel("Predicted Digit")
    axes[0].set_ylabel("True Digit")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Logistic Regression Confusion Matrix
    cm_lr = np.array(metrics["baseline_logreg"].confusion_matrix)
    im1 = axes[1].imshow(cm_lr, cmap="Oranges")
    axes[1].set_title(
        f"Logistic Regression Baseline\nAcc: {metrics['baseline_logreg'].accuracy*100:.2f}%"
    )
    axes[1].set_xticks(range(9))
    axes[1].set_yticks(range(9))
    axes[1].set_xticklabels(range(1, 10))
    axes[1].set_yticklabels(range(1, 10))
    axes[1].set_xlabel("Predicted Digit")
    axes[1].set_ylabel("True Digit")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Reliability Diagram (Calibration Curve)
    probs_nn_uncal = nn_clf.predict_proba(dataset.X_test, use_temperature=False)
    probs_nn_cal = nn_clf.predict_proba(dataset.X_test, use_temperature=True)
    probs_lr = lr_clf.predict_proba(dataset.X_test, use_temperature=False)

    axes[2].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for label, probs, color, marker in [
        (
            f"NN Uncalibrated (ECE={metrics['nn'].ece_uncalibrated:.4f})",
            probs_nn_uncal,
            "#d95f02",
            "s",
        ),
        (
            f"NN Temp-Scaled (ECE={metrics['nn'].ece_calibrated:.4f})",
            probs_nn_cal,
            "#1b9e77",
            "o",
        ),
        (
            f"LogReg Baseline (ECE={metrics['baseline_logreg'].ece_uncalibrated:.4f})",
            probs_lr,
            "#7570b3",
            "^",
        ),
    ]:
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1) + 1
        accs = (preds == dataset.y_test).astype(np.float64)
        bins = np.linspace(0.0, 1.0, 11)
        bin_confs, bin_accs = [], []
        for i in range(10):
            mask = (confs > bins[i]) & (confs <= bins[i + 1])
            if np.sum(mask) > 5:
                bin_confs.append(float(np.mean(confs[mask])))
                bin_accs.append(float(np.mean(accs[mask])))
        axes[2].plot(bin_confs, bin_accs, marker=marker, color=color, linewidth=2, label=label)

    axes[2].set_title("Reliability Diagram (Test Set Digits 1-9)")
    axes[2].set_xlabel("Mean Predicted Confidence")
    axes[2].set_ylabel("Empirical Accuracy")
    axes[2].set_xlim(0.3, 1.02)
    axes[2].set_ylim(0.3, 1.02)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(results_dir / "fig_nn_vs_baseline.png", dpi=200)
    plt.close(fig)


def run_experiment_1_search(
    dataset,
    nn_clf,
    test_probs: np.ndarray,
    base_puzzles_by_seed: Dict[int, List[ Any ]],
    puzzles_per_seed: int = 20,
) -> Dict[str, Any]:
    """
    Experiment 1: UCS vs Greedy vs A* for 0..4 injected misreads across 5 seeds
    (5 seeds x 20 puzzles = 100 puzzles per misread level).
    Also verifies that the A* heuristic never overestimates true optimal cost.
    """
    misread_levels = [0, 1, 2, 3, 4]
    algorithms = ["ucs", "greedy", "astar"]

    results: Dict[str, Dict[str, Any]] = {alg: {} for alg in algorithms}
    heuristic_checks = {"total_checked": 0, "overestimates": 0}

    for k in misread_levels:
        per_alg_seed_nodes: Dict[str, List[float]] = {a: [] for a in algorithms}
        per_alg_seed_gen: Dict[str, List[float]] = {a: [] for a in algorithms}
        per_alg_seed_rec_budget3: Dict[str, List[float]] = {a: [] for a in algorithms}
        per_alg_seed_rec_budget4: Dict[str, List[float]] = {a: [] for a in algorithms}
        per_alg_seed_cost: Dict[str, List[float]] = {a: [] for a in algorithms}

        for seed in SEEDS:
            mosaics = generate_sudoku_mosaic_dataset(
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                num_puzzles=puzzles_per_seed,
                num_givens=30,
                num_injected_misreads=k,
                seed=seed + k * 1000,
                classifier_probs_on_test=test_probs,
                predict_proba_fn=nn_clf.predict_proba,
                precomputed_puzzles=base_puzzles_by_seed[seed],
            )

            # Pre-read all mosaics once with NN
            readings = [
                read_mosaic_grid(m.cell_images, nn_clf) for m in mosaics
            ]

            for alg in algorithms:
                nodes_list = []
                gen_list = []
                rec3_list = []
                rec4_list = []
                cost_list = []

                for idx, m in enumerate(mosaics):
                    init_grid, c_probs = readings[idx]
                    # Standard P7 budget is max_corrections=3; for k=4 we also check budget=4
                    res4 = run_correction_search(
                        initial_grid=init_grid,
                        cell_probs=c_probs,
                        algorithm=alg,
                        max_corrections=3 if k <= 3 else 4,
                        max_uncertain_cells=6,
                        top_alts_per_cell=2,
                        max_expanded=65,
                    )
                    exact4 = (
                        res4.solved
                        and res4.solution_grid is not None
                        and (res4.solution_grid == m.solution_grid)
                    )
                    exact3 = exact4 and (len(res4.corrections) <= 3)

                    nodes_list.append(float(res4.nodes_expanded))
                    gen_list.append(float(res4.nodes_generated))
                    rec4_list.append(100.0 if exact4 else 0.0)
                    rec3_list.append(100.0 if exact3 else 0.0)
                    if res4.solved:
                        cost_list.append(float(res4.total_cost))

                    # Verify A* heuristic admissibility against UCS optimal cost
                    if alg == "ucs" and res4.solved:
                        _, min_c, _ = build_candidate_actions(
                            init_grid, c_probs, max_uncertain_cells=10, top_alts_per_cell=2
                        )
                        h0, _ = compute_conflict_heuristic(init_grid, min_c, set())
                        heuristic_checks["total_checked"] += 1
                        if h0 > res4.total_cost + 1e-7:
                            heuristic_checks["overestimates"] += 1

                per_alg_seed_nodes[alg].append(float(np.mean(nodes_list)))
                per_alg_seed_gen[alg].append(float(np.mean(gen_list)))
                per_alg_seed_rec_budget3[alg].append(float(np.mean(rec3_list)))
                per_alg_seed_rec_budget4[alg].append(float(np.mean(rec4_list)))
                per_alg_seed_cost[alg].append(
                    float(np.mean(cost_list)) if cost_list else 0.0
                )

        for alg in algorithms:
            results[alg][str(k)] = {
                "nodes_expanded": _mean_std(per_alg_seed_nodes[alg]),
                "nodes_generated": _mean_std(per_alg_seed_gen[alg]),
                "exact_recovery_budget3": _mean_std(per_alg_seed_rec_budget3[alg]),
                "exact_recovery_budget4": _mean_std(per_alg_seed_rec_budget4[alg]),
                "solution_cost": _mean_std(per_alg_seed_cost[alg]),
            }

    return {"by_algorithm": results, "heuristic_admissibility": heuristic_checks}


def run_experiment_2_csp(puzzles_per_seed: int = 20) -> Dict[str, Any]:
    """
    Experiment 2: BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens
    across 5 seeds.
    """
    givens_levels = [25, 30, 35]
    configs = {
        "BT": {"use_fc": False, "use_ac3": False, "max_assignments": 2_500},
        "BT+FC": {"use_fc": True, "use_ac3": False, "max_assignments": 2_500},
        "BT+FC+AC3": {"use_fc": True, "use_ac3": True, "max_assignments": 2_500},
    }
    results: Dict[str, Dict[str, Any]] = {cfg: {} for cfg in configs}

    for g_count in givens_levels:
        per_cfg_assign: Dict[str, List[float]] = {c: [] for c in configs}
        per_cfg_backtracks: Dict[str, List[float]] = {c: [] for c in configs}
        per_cfg_pruned: Dict[str, List[float]] = {c: [] for c in configs}
        per_cfg_time_ms: Dict[str, List[float]] = {c: [] for c in configs}
        per_cfg_solved: Dict[str, List[float]] = {c: [] for c in configs}

        for seed in SEEDS:
            rng = np.random.default_rng(seed + g_count * 100)
            puzzles = [
                p[0]
                for p in fast_generate_unique_puzzles(
                    rng, num_puzzles=puzzles_per_seed, num_givens=g_count
                )
            ]

            for cfg_name, params in configs.items():
                solver = SudokuCSP(
                    use_fc=params["use_fc"],
                    use_ac3=params["use_ac3"],
                    max_assignments=params["max_assignments"],
                )
                assigns, bts, pruned, times_ms, solved_flags = [], [], [], [], []
                for puz in puzzles:
                    t0 = time.perf_counter()
                    sol, st = solver.solve(puz)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    assigns.append(float(st.assignments))
                    bts.append(float(st.backtracks))
                    pruned.append(float(st.values_pruned))
                    times_ms.append(float(elapsed_ms))
                    solved_flags.append(100.0 if sol is not None else 0.0)

                per_cfg_assign[cfg_name].append(float(np.mean(assigns)))
                per_cfg_backtracks[cfg_name].append(float(np.mean(bts)))
                per_cfg_pruned[cfg_name].append(float(np.mean(pruned)))
                per_cfg_time_ms[cfg_name].append(float(np.mean(times_ms)))
                per_cfg_solved[cfg_name].append(float(np.mean(solved_flags)))

        for cfg_name in configs:
            results[cfg_name][str(g_count)] = {
                "assignments": _mean_std(per_cfg_assign[cfg_name]),
                "backtracks": _mean_std(per_cfg_backtracks[cfg_name]),
                "values_pruned": _mean_std(per_cfg_pruned[cfg_name]),
                "time_ms": _mean_std(per_cfg_time_ms[cfg_name]),
                "solve_rate": _mean_std(per_cfg_solved[cfg_name]),
            }

    return results


def run_experiment_3_ablation(
    dataset,
    nn_clf,
    test_probs: np.ndarray,
    base_puzzles_by_seed: Dict[int, List[Any]],
    puzzles_per_seed: int = 20,
) -> Dict[str, Any]:
    """
    Experiment 3: NN-guided correction vs No correction vs Random correction
    across misread levels k in {1, 2, 3} over 5 seeds.
    """
    modes = ["nn_guided", "no_correction", "random_correction"]
    misread_levels = [1, 2, 3]
    results: Dict[str, Dict[str, Any]] = {m: {} for m in modes}

    pipeline = SudokuReaderSolverPipeline(
        classifier=nn_clf,
        search_algorithm="astar",
        max_corrections=3,
        max_uncertain_cells=8,
        top_alts_per_cell=2,
    )

    for k in misread_levels:
        per_mode_exact: Dict[str, List[float]] = {m: [] for m in modes}
        per_mode_solved: Dict[str, List[float]] = {m: [] for m in modes}
        per_mode_nodes: Dict[str, List[float]] = {m: [] for m in modes}

        for seed in SEEDS:
            mosaics = generate_sudoku_mosaic_dataset(
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                num_puzzles=puzzles_per_seed,
                num_givens=30,
                num_injected_misreads=k,
                seed=seed + k * 500,
                classifier_probs_on_test=test_probs,
                predict_proba_fn=nn_clf.predict_proba,
                precomputed_puzzles=base_puzzles_by_seed[seed],
            )
            readings = [read_mosaic_grid(m.cell_images, nn_clf) for m in mosaics]

            for mode in modes:
                exact_list, solved_list, nodes_list = [], [], []
                for idx, m in enumerate(mosaics):
                    init_grid, c_probs = readings[idx]
                    out = pipeline.solve_from_reading(
                        initial_grid=init_grid,
                        cell_probs=c_probs,
                        mode=mode,
                        true_puzzle_grid=m.puzzle_grid,
                        true_solution_grid=m.solution_grid,
                        random_seed=seed + idx,
                    )
                    exact_list.append(100.0 if out.exact_recovery else 0.0)
                    solved_list.append(100.0 if out.solved else 0.0)
                    nodes_list.append(float(out.nodes_expanded))

                per_mode_exact[mode].append(float(np.mean(exact_list)))
                per_mode_solved[mode].append(float(np.mean(solved_list)))
                per_mode_nodes[mode].append(float(np.mean(nodes_list)))

        for mode in modes:
            results[mode][str(k)] = {
                "exact_recovery": _mean_std(per_mode_exact[mode]),
                "solvable_rate": _mean_std(per_mode_solved[mode]),
                "nodes_expanded": _mean_std(per_mode_nodes[mode]),
            }

    return results


def plot_experiments(
    exp1: Dict[str, Any],
    exp2: Dict[str, Any],
    exp3: Dict[str, Any],
    results_dir: Path,
) -> None:
    """Create publication-quality figures for Experiments 1, 2, and 3."""
    # --- Figure Exp 1: UCS vs Greedy vs A* ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    levels = [0, 1, 2, 3, 4]
    colors = {"ucs": "#1f77b4", "greedy": "#ff7f0e", "astar": "#2ca02c"}
    labels = {"ucs": "UCS", "greedy": "Greedy Best-First", "astar": "A* (Conflict Heuristic)"}
    markers = {"ucs": "o", "greedy": "s", "astar": "^"}

    for alg in ["ucs", "greedy", "astar"]:
        means_n = [exp1["by_algorithm"][alg][str(k)]["nodes_expanded"]["mean"] for k in levels]
        stds_n = [exp1["by_algorithm"][alg][str(k)]["nodes_expanded"]["std"] for k in levels]
        axes[0].errorbar(
            levels,
            means_n,
            yerr=stds_n,
            label=labels[alg],
            color=colors[alg],
            marker=markers[alg],
            linewidth=2,
            capsize=4,
        )

        means_r = [exp1["by_algorithm"][alg][str(k)]["exact_recovery_budget4"]["mean"] for k in levels]
        stds_r = [exp1["by_algorithm"][alg][str(k)]["exact_recovery_budget4"]["std"] for k in levels]
        axes[1].errorbar(
            levels,
            means_r,
            yerr=stds_r,
            label=labels[alg],
            color=colors[alg],
            marker=markers[alg],
            linewidth=2,
            capsize=4,
        )

    axes[0].set_title("Exp 1A: Search Nodes Expanded vs. Injected Misreads")
    axes[0].set_xlabel("Number of Injected Misreads (k)")
    axes[0].set_ylabel("Nodes Expanded (Mean ± SD, 5 seeds)")
    axes[0].set_xticks(levels)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Exp 1B: Exact-Solution Recovery Rate (100 Puzzles/Level)")
    axes[1].set_xlabel("Number of Injected Misreads (k)")
    axes[1].set_ylabel("Exact Solution Recovery Rate (%)")
    axes[1].set_xticks(levels)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "fig_exp1_search.png", dpi=200)
    plt.close(fig)

    # --- Figure Exp 2: BT vs BT+FC vs BT+FC+AC-3 ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g_levels = [25, 30, 35]
    x = np.arange(len(g_levels))
    width = 0.25
    cfg_colors = {"BT": "#d62728", "BT+FC": "#1f77b4", "BT+FC+AC3": "#2ca02c"}

    for idx, cfg in enumerate(["BT", "BT+FC", "BT+FC+AC3"]):
        means_a = [exp2[cfg][str(g)]["assignments"]["mean"] for g in g_levels]
        stds_a = [exp2[cfg][str(g)]["assignments"]["std"] for g in g_levels]
        axes[0].bar(
            x + (idx - 1) * width,
            means_a,
            width,
            yerr=stds_a,
            label=cfg,
            color=cfg_colors[cfg],
            capsize=4,
            alpha=0.9,
        )

        means_b = [max(0.5, exp2[cfg][str(g)]["backtracks"]["mean"]) for g in g_levels]
        stds_b = [exp2[cfg][str(g)]["backtracks"]["std"] for g in g_levels]
        axes[1].bar(
            x + (idx - 1) * width,
            means_b,
            width,
            yerr=stds_b,
            label=cfg,
            color=cfg_colors[cfg],
            capsize=4,
            alpha=0.9,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Exp 2A: CSP Variable Assignments (Log Scale)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{g} Givens" for g in g_levels])
    axes[0].set_ylabel("Assignments (Log Scale, Mean ± SD)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].set_yscale("log")
    axes[1].set_title("Exp 2B: CSP Backtracks (Log Scale)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{g} Givens" for g in g_levels])
    axes[1].set_ylabel("Backtracks (Log Scale, Mean ± SD)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "fig_exp2_csp.png", dpi=200)
    plt.close(fig)

    # --- Figure Exp 3: NN-Guided vs No Correction vs Random Correction ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    k_levels = [1, 2, 3]
    x3 = np.arange(len(k_levels))
    mode_labels = {
        "nn_guided": "NN-Guided Correction (A* + CSP)",
        "random_correction": "Random Correction Baseline",
        "no_correction": "No Correction (Raw NN Reading)",
    }
    mode_colors = {
        "nn_guided": "#2ca02c",
        "random_correction": "#ff7f0e",
        "no_correction": "#d62728",
    }

    for idx, mode in enumerate(["nn_guided", "random_correction", "no_correction"]):
        means_e = [exp3[mode][str(k)]["exact_recovery"]["mean"] for k in k_levels]
        stds_e = [exp3[mode][str(k)]["exact_recovery"]["std"] for k in k_levels]
        ax.bar(
            x3 + (idx - 1) * width,
            means_e,
            width,
            yerr=stds_e,
            label=mode_labels[mode],
            color=mode_colors[mode],
            capsize=4,
            alpha=0.9,
        )

    ax.set_title("Exp 3: Exact-Solution Recovery Rate by Correction Strategy (5 Seeds)")
    ax.set_xticks(x3)
    ax.set_xticklabels([f"{k} Injected Misread(s)" for k in k_levels])
    ax.set_ylabel("Exact Solution Recovery Rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "fig_exp3_ablation.png", dpi=200)
    plt.close(fig)


def run_all_experiments(
    results_dir: Path = Path("results"),
    puzzles_per_seed: int = 20,
) -> Dict[str, Any]:
    """Run all three project experiments across 5 seeds and save tables and plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_processed_mnist(seed=42)
    nn_clf, lr_clf, clf_metrics = load_or_train_models(dataset=dataset, seed=42)
    test_probs = nn_clf.predict_proba(dataset.X_test)

    print("[eval] Plotting NN vs Logistic Regression calibration & confusion matrices...")
    plot_classifier_comparison(dataset, nn_clf, lr_clf, clf_metrics, results_dir)

    print("[eval] Pre-generating 100 base 30-given Sudoku puzzles across 5 seeds...")
    base_puzzles_by_seed: Dict[int, List[Any]] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        base_puzzles_by_seed[seed] = fast_generate_unique_puzzles(
            rng, num_puzzles=puzzles_per_seed, num_givens=30
        )

    print(
        f"[eval] Running Experiment 1 (UCS vs Greedy vs A*, 5 seeds x {puzzles_per_seed} puzzles = "
        f"{5 * puzzles_per_seed} puzzles/level)..."
    )
    exp1 = run_experiment_1_search(
        dataset, nn_clf, test_probs, base_puzzles_by_seed, puzzles_per_seed=puzzles_per_seed
    )

    print(
        f"[eval] Running Experiment 2 (BT vs BT+FC vs BT+FC+AC3 on 25, 30, 35 givens, 5 seeds)..."
    )
    exp2 = run_experiment_2_csp(puzzles_per_seed=puzzles_per_seed)

    print(
        f"[eval] Running Experiment 3 (NN-Guided vs No Correction vs Random Correction, 5 seeds)..."
    )
    exp3 = run_experiment_3_ablation(
        dataset, nn_clf, test_probs, base_puzzles_by_seed, puzzles_per_seed=puzzles_per_seed
    )

    print("[eval] Generating publication plots for Experiments 1, 2, and 3...")
    plot_experiments(exp1, exp2, exp3, results_dir)

    payload = {
        "seeds": SEEDS,
        "puzzles_per_seed": puzzles_per_seed,
        "total_puzzles_per_condition": len(SEEDS) * puzzles_per_seed,
        "experiment_1_search": exp1,
        "experiment_2_csp": exp2,
        "experiment_3_ablation": exp3,
    }
    (results_dir / "experiment_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[eval] All experiments finished and saved to {results_dir.resolve()}.")
    return payload


if __name__ == "__main__":
    run_all_experiments()
