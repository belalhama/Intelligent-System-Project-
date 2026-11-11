"""
Integrated Demo Script (run_demo.py) for P7: Handwritten Sudoku Reader and Solver.

Deliverables #7 and #12:
- Shows the 9x9 handwritten MNIST mosaic input and ground-truth givens.
- Shows blank-cell detection (brightness rule) and calibrated NN initial reading + probabilities.
- Shows AC-3 conflict detection and domain shrinking on the initial reading.
- Prints the step-by-step Search trace (UCS, Greedy, A*) and the Correction Log
  (cell, read_digit, new_digit, cost).
- Compares CSP pruning with and without Forward Checking (FC) and Arc Consistency (AC-3).
- Compares the integrated system with NN-guided correction ON versus OFF (and Random).
- Demonstrates an unreadable puzzle case ("No solution: unreadable puzzle" + least confident cells).
- Saves a 4-panel cooperation plot to results/demo_cooperation.png.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / "results" / ".mplconfig"))
(ROOT_DIR / "results" / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from csp.sudoku_csp import Grid, SudokuCSP
from data.mnist_loader import BRIGHTNESS_THRESHOLD, load_processed_mnist
from data.sudoku_generator import (
    generate_sudoku_mosaic_dataset,
    render_mosaic_image,
)
from integration.pipeline import SudokuReaderSolverPipeline, read_mosaic_grid
from nn.digit_classifier import load_or_train_models
from search.correction_search import run_correction_search, run_small_known_example


def _format_grid(grid: Grid, highlight_cells=None) -> str:
    """Format a 9x9 Sudoku grid as a clean ASCII board."""
    highlight_cells = set(highlight_cells or [])
    lines: List[str] = []
    sep = "  +---------+---------+---------+"
    for r in range(9):
        if r % 3 == 0:
            lines.append(sep)
        row_str = f"{r} |"
        for c in range(9):
            val = int(grid[r][c])
            ch = "." if val == 0 else str(val)
            if (r, c) in highlight_cells:
                row_str += f"[{ch}]"
            else:
                row_str += f" {ch} "
            if (c + 1) % 3 == 0:
                row_str += "|"
        lines.append(row_str)
    lines.append(sep)
    return "\n".join(lines)


def save_cooperation_plot(
    mosaic_instance,
    initial_grid: Grid,
    cell_probs,
    astar_res,
    ucs_res,
    csp_comparison,
    out_path: Path,
) -> None:
    """Generate a 4-panel visual summary of how NN, Search, and CSP cooperate."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1 (Top-Left): 9x9 Handwritten Mosaic Input with Misreads & Corrections Highlighted
    ax0 = axes[0, 0]
    mosaic_2d = render_mosaic_image(mosaic_instance.cell_images, border_px=1)
    ax0.imshow(mosaic_2d, cmap="magma")
    ax0.set_title(
        "1. NN Input: 9x9 Handwritten MNIST Mosaic\n(Red = Misread by NN, Green Label = Search Correction)",
        fontsize=11,
        fontweight="bold",
    )
    ax0.axis("off")

    corr_map = {c.cell: c for c in astar_res.corrections}
    for (r, c), corr in corr_map.items():
        y0 = 1 + r * 29
        x0 = 1 + c * 29
        rect = patches.Rectangle(
            (x0 - 0.5, y0 - 0.5),
            28,
            28,
            linewidth=2.5,
            edgecolor="#ff3333",
            facecolor="none",
        )
        ax0.add_patch(rect)
        ax0.text(
            x0 + 1,
            y0 + 7,
            f"{corr.read_digit}->{corr.new_digit}",
            color="#00ff66",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.75),
        )

    # Panel 2 (Top-Right): Calibrated NN Probabilities & Step Costs for Top Uncertain Cells
    ax1 = axes[0, 1]
    top_uncertain = astar_res.least_confident_cells[:5]
    cell_labels = [f"Cell {cell}\n(Read={d})" for cell, d, _ in top_uncertain]
    top1_p, top2_p, top3_p = [], [], []
    top2_d, top3_d = [], []
    for cell, _, _ in top_uncertain:
        p = cell_probs[cell]
        order = np.argsort(-p)[:3]
        top1_p.append(float(p[order[0]]))
        top2_p.append(float(p[order[1]]))
        top3_p.append(float(p[order[2]]))
        top2_d.append(int(order[1] + 1))
        top3_d.append(int(order[2] + 1))

    y_pos = np.arange(len(top_uncertain))
    bar_h = 0.25
    ax1.barh(y_pos - bar_h, top1_p, height=bar_h, label="Top-1 (Initial Read)", color="#1f77b4")
    bars2 = ax1.barh(
        y_pos, top2_p, height=bar_h, label="Top-2 Alternative", color="#2ca02c"
    )
    bars3 = ax1.barh(
        y_pos + bar_h, top3_p, height=bar_h, label="Top-3 Alternative", color="#ff7f0e"
    )
    for idx, b in enumerate(bars2):
        ax1.text(
            b.get_width() + 0.015,
            b.get_y() + b.get_height() / 2,
            f"d={top2_d[idx]}",
            va="center",
            fontsize=8.5,
        )
    for idx, b in enumerate(bars3):
        ax1.text(
            b.get_width() + 0.015,
            b.get_y() + b.get_height() / 2,
            f"d={top3_d[idx]}",
            va="center",
            fontsize=8.5,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cell_labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("Calibrated Probability P(digit | image)")
    ax1.set_title(
        "2. Calibrated NN Uncertainty Ranking\n(Provides Non-Negative Search Step Costs)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.legend(loc="lower right")

    # Panel 3 (Bottom-Left): Search Trace & CSP Pruning Comparison
    ax2 = axes[1, 0]
    categories = [
        "Search Nodes\n(UCS)",
        "Search Nodes\n(A*)",
        "CSP Assigns\n(Pure BT)",
        "CSP Assigns\n(BT + FC)",
        "CSP Assigns\n(BT+FC+AC3)",
    ]
    values = [
        max(1, ucs_res.nodes_expanded),
        max(1, astar_res.nodes_expanded),
        max(1, csp_comparison["BT"].assignments),
        max(1, csp_comparison["BT+FC"].assignments),
        max(1, csp_comparison["BT+FC+AC3"].assignments),
    ]
    bar_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
    bars = ax2.bar(categories, values, color=bar_colors, width=0.55)
    ax2.set_yscale("log")
    for b, v in zip(bars, values):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v * 1.12,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_ylabel("Count (Log Scale)")
    ax2.set_title(
        "3. Search (UCS vs A*) & CSP Pruning (BT vs FC vs AC-3)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylim(1, max(values) * 4.5)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 4 (Bottom-Right): Solved 9x9 Sudoku Grid
    ax3 = axes[1, 1]
    ax3.set_xlim(0, 9)
    ax3.set_ylim(0, 9)
    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax3.set_title(
        "4. Final Verified Sudoku Solution\n(Blue = Original Given, Green = Corrected Given, Gray = CSP Filled)",
        fontsize=11,
        fontweight="bold",
    )

    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.6
        ax3.axhline(i, color="black", linewidth=lw)
        ax3.axvline(i, color="black", linewidth=lw)

    sol = astar_res.solution_grid
    for r in range(9):
        for c in range(9):
            val = sol[r][c]
            y_plot = 8 - r + 0.5
            x_plot = c + 0.5
            if (r, c) in corr_map:
                bg = patches.Rectangle((c, 8 - r), 1, 1, facecolor="#d4f8d4", zorder=0)
                ax3.add_patch(bg)
                color = "#007700"
                weight = "bold"
            elif initial_grid[r][c] != 0:
                bg = patches.Rectangle((c, 8 - r), 1, 1, facecolor="#e8f1fc", zorder=0)
                ax3.add_patch(bg)
                color = "#003388"
                weight = "bold"
            else:
                color = "#444444"
                weight = "normal"
            ax3.text(
                x_plot,
                y_plot,
                str(val),
                ha="center",
                va="center",
                fontsize=13,
                color=color,
                fontweight=weight,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("  P7: HANDWRITTEN SUDOKU READER AND SOLVER - INTEGRATED SYSTEM DEMO")
    print("=" * 78)

    # 1. Small Known-Answer Search Verification (Deliverable #5)
    print("\n[Part A] Small Deterministic Example with Known Answer (PDF Page 14):")
    small_res = run_small_known_example()
    for alg_name, s_res in small_res.items():
        corr_str = ", ".join(
            f"cell {c.cell}: {c.read_digit}->{c.new_digit} (cost={c.cost:.4f})"
            for c in s_res.corrections
        )
        print(
            f"  {alg_name.upper():<6} | solved={s_res.solved} | "
            f"expanded={s_res.nodes_expanded} | generated={s_res.nodes_generated} | "
            f"correction: [{corr_str}]"
        )

    # 2. Load MNIST Dataset and Trained Calibrated NN
    print("\n[Part B] Loading Processed MNIST (1..9) and Calibrated Neural Network...")
    dataset = load_processed_mnist(seed=42)
    nn_clf, lr_clf, metrics = load_or_train_models(dataset=dataset, seed=42)
    nn_m = metrics["nn"]
    lr_m = metrics["baseline_logreg"]
    print(
        f"  Calibrated NN (T={nn_m.temperature:.3f}): Test Accuracy = {nn_m.accuracy*100:.2f}%, "
        f"ECE = {nn_m.ece_calibrated:.4f} (Uncalibrated ECE = {nn_m.ece_uncalibrated:.4f})"
    )
    print(
        f"  Logistic Regression Baseline:  Test Accuracy = {lr_m.accuracy*100:.2f}%, "
        f"ECE = {lr_m.ece_uncalibrated:.4f}"
    )

    # 3. Generate a 30-given Handwritten Mosaic Puzzle with 2 Injected Hard Misreads
    test_probs = nn_clf.predict_proba(dataset.X_test)
    mosaics = generate_sudoku_mosaic_dataset(
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        num_puzzles=1,
        num_givens=30,
        num_injected_misreads=2,
        seed=42,
        classifier_probs_on_test=test_probs,
        predict_proba_fn=nn_clf.predict_proba,
    )
    demo_mosaic = mosaics[0]

    print(
        f"\n[Part C] Optical Grid Reading (Brightness threshold = {BRIGHTNESS_THRESHOLD}):"
    )
    initial_grid, cell_probs = read_mosaic_grid(demo_mosaic.cell_images, nn_clf)
    misread_coords = [ic.cell for ic in demo_mosaic.injected_cells]
    print("  Injected Hard Digits (Ground Truth vs NN Initial Top-1 Reading):")
    for ic in demo_mosaic.injected_cells:
        r, c = ic.cell
        p = cell_probs[(r, c)]
        read_d = initial_grid[r][c]
        print(
            f"    Cell {(r, c)}: True Digit = {ic.true_digit}, "
            f"NN Read = {read_d} (P={p[read_d-1]*100:.1f}%), "
            f"Second Choice = {ic.true_digit} (P={p[ic.true_digit-1]*100:.1f}%)"
        )

    print("\n  Initial 9x9 Grid Read by NN (bracketed [d] = misread cells):")
    print(_format_grid(initial_grid, highlight_cells=misread_coords))

    # 4. Run Pipeline with A*, UCS, and Greedy
    pipeline_astar = SudokuReaderSolverPipeline(
        classifier=nn_clf, search_algorithm="astar", max_corrections=3, require_unique=True
    )
    out_astar = pipeline_astar.solve_from_reading(
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        mode="nn_guided",
        true_puzzle_grid=demo_mosaic.puzzle_grid,
        true_solution_grid=demo_mosaic.solution_grid,
    )
    ucs_res = run_correction_search(
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        algorithm="ucs",
        max_corrections=3,
        require_unique=True,
    )
    greedy_res = run_correction_search(
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        algorithm="greedy",
        max_corrections=3,
        require_unique=True,
    )

    print("\n[Part D] Initial AC-3 Conflict Detection & Search Trace:")
    print(f"  Initial AC-3 Consistent: {out_astar.initial_ac3_consistent}")
    print(f"  Initial Duplicate-Given Conflicts Found: {out_astar.initial_conflicts}")
    print("\n  A* Search Step-by-Step Trace:")
    for step in out_astar.trace:
        corr_desc = (
            "None (Initial Reading)"
            if not step.corrections
            else ", ".join(f"{c}:{old}->{new}" for c, old, new in step.corrections)
        )
        print(
            f"    Step {step.step:2d} | g={step.g_cost:.4f}, h={step.h_cost:.4f}, "
            f"f={step.f_cost:.4f} | conflicts={step.num_conflicts} | "
            f"CSP_solved={step.csp_solved} | Corrections: {corr_desc}"
        )

    print("\n  Search Algorithm Comparison on Demo Puzzle:")
    for name, res_obj in [("UCS", ucs_res), ("Greedy", greedy_res), ("A*", out_astar)]:
        print(
            f"    {name:<8} | Solved={res_obj.solved} | "
            f"Expanded={res_obj.nodes_expanded:2d} | Generated={res_obj.nodes_generated:3d} | "
            f"Total Cost={res_obj.total_cost:.4f}"
        )

    print("\n  Correction Log (Deliverable: cell, read_digit, new_digit, cost):")
    for c_entry in out_astar.corrections:
        print(
            f"    Cell {c_entry.cell}: read_digit={c_entry.read_digit} "
            f"(P={c_entry.read_prob:.3f}) -> new_digit={c_entry.new_digit} "
            f"(P={c_entry.new_prob:.3f}) | step_cost={c_entry.cost:.4f}"
        )

    # 5. CSP Pruning Comparison (BT vs BT+FC vs BT+FC+AC-3) on the Corrected Puzzle
    print("\n[Part E] CSP Pruning Comparison (BT vs BT+FC vs BT+FC+AC-3):")
    csp_comparison = {}
    for cfg_label, use_fc, use_ac3 in [
        ("BT", False, False),
        ("BT+FC", True, False),
        ("BT+FC+AC3", True, True),
    ]:
        solver = SudokuCSP(use_fc=use_fc, use_ac3=use_ac3)
        _, st = solver.solve(out_astar.corrected_grid)
        csp_comparison[cfg_label] = st
        print(
            f"    {cfg_label:<10} | Solved={st.solved} | Assignments={st.assignments:5d} | "
            f"Backtracks={st.backtracks:5d} | Pruned Values={st.values_pruned:4d} | "
            f"Domains (Init->AC3)={st.initial_domain_sizes}->{st.post_ac3_domain_sizes}"
        )

    # 6. System with NN Correction ON vs OFF
    print("\n[Part F] System Comparison: NN-Guided Correction ON vs OFF:")
    out_off = pipeline_astar.solve_from_reading(
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        mode="no_correction",
        true_puzzle_grid=demo_mosaic.puzzle_grid,
        true_solution_grid=demo_mosaic.solution_grid,
    )
    print(
        f"    NN Correction OFF (No Correction): Solved={out_off.solved} | "
        f"Exact Recovery={out_off.exact_recovery} | Message='{out_off.message}'"
    )
    print(
        f"    NN Correction ON  (A* + CSP):      Solved={out_astar.solved} | "
        f"Exact Recovery={out_astar.exact_recovery} | Message='{out_astar.message}'"
    )

    print("\n  Final Solved 9x9 Sudoku Grid (bracketed [d] = cells corrected before solving):")
    print(_format_grid(out_astar.solution_grid, highlight_cells=misread_coords))

    # 7. Impossible / Unreadable Case Demonstration
    print("\n[Part G] Impossible / Unreadable Puzzle Case (5 conflicts > max_corrections=3):")
    unreadable_grid = [row[:] for row in initial_grid]
    # Corrupt 5 distinct cells to create 5 separate row/box conflicts
    unreadable_grid[0][0] = unreadable_grid[0][1] = 9
    unreadable_grid[2][2] = unreadable_grid[2][3] = 8
    unreadable_grid[4][4] = unreadable_grid[4][5] = 7
    unreadable_grid[6][6] = unreadable_grid[6][7] = 6
    unreadable_grid[8][7] = unreadable_grid[8][8] = 5
    out_impossible = pipeline_astar.solve_from_reading(
        initial_grid=unreadable_grid,
        cell_probs=cell_probs,
        mode="nn_guided",
    )
    print(f"    Solved: {out_impossible.solved}")
    print(f"    Status: {out_impossible.message}")
    print(
        "    Least Confident Cells Reported: "
        + ", ".join(
            f"{cell} (read={d}, conf={p:.2f})"
            for cell, d, p in out_impossible.least_confident_cells[:5]
        )
    )

    # 8. Save 4-panel cooperation plot
    plot_path = Path("results/demo_cooperation.png")
    save_cooperation_plot(
        mosaic_instance=demo_mosaic,
        initial_grid=initial_grid,
        cell_probs=cell_probs,
        astar_res=out_astar,
        ucs_res=ucs_res,
        csp_comparison=csp_comparison,
        out_path=plot_path,
    )
    print(f"\n[Demo Complete] Saved 4-panel cooperation plot to: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
