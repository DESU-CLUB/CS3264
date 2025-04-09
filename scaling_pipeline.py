import os
import sys
import logging
from chromadb import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from basic_eval_pipeline import run_model_evaluations, compare_results, analyze_csv_features, parse_synthetic_df
from generate_data import generate_data

"""
scaling_pipeline.py
-----------------
Loop over different n_samples values, call run_evaluation_pipeline each time,
collect accuracy for every model, and plot accuracy vs. synthetic‑set size.

Place this file next to basic_eval_pipeline.py and run:
    python scaling_pipeline.py
"""

import os, logging, matplotlib.pyplot as plt
from basic_eval_pipeline import run_evaluation_pipeline   # <-- your function

# ---------- CONFIG ----------
DATASET_PATH = "./evals/dataset/andrew_diabetes.csv"
OUTPUT_ROOT  = "./results/size_experiments"
MODELS       = ["knn", "mlp", "randomforest", "svm"]    # pick any subset of your models
SIZES        = [100, 200, 400, 800, 1200, 1600, 2000]   # synthetic rows to generate
TUNE_HYPERPARAMS = True                                 # Enable Optuna hyperparameter tuning
N_TRIALS = 20                                           # Number of Optuna trials per model
# -----------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("size‑driver")

# --- Step 3‑4: run experiments & gather accuracies ---------------------------
size_acc_syn_tuned = {m: [] for m in MODELS}      # (n, synthetic‑acc with tuning)
size_acc_syn_untuned = {m: [] for m in MODELS}    # (n, synthetic‑acc without tuning) - for comparison
baseline_acc_tuned = {}                           # original‑data accuracy with tuning
baseline_acc_untuned = {}                         # original‑data accuracy without tuning

# First run without tuning for baseline comparison
log.info("=== Running without hyperparameter tuning for baseline comparison ===")
for n in SIZES:
    log.info(f"=== n_samples = {n} (untuned) ===")
    run_dir = os.path.join(OUTPUT_ROOT, f"n_{n}_untuned")
    res = run_evaluation_pipeline(
        dataset_path = DATASET_PATH,
        output_dir   = run_dir,
        test_size    = 0.20,
        n_samples    = n,
        models       = MODELS,
        random_state = 42,
        tune_hyperparams = False
    )
    if res is None:
        log.warning(f" run_evaluation_pipeline returned None for n={n} (untuned)")
        continue

    # --- grab original & synthetic accuracies without tuning ----------------
    orig  = res["original_results"]
    synth = res["synthetic_results"]

    for m in MODELS:
        # store untuned baseline once
        if m not in baseline_acc_untuned and "accuracy" in orig.get(m, {}):
            baseline_acc_untuned[m] = orig[m]["accuracy"]

        # store untuned synthetic accuracy for this n
        acc_syn = synth.get(m, {}).get("accuracy")
        if acc_syn is not None:
            size_acc_syn_untuned[m].append((n, acc_syn))
            log.info(f"  {m:8s}  untuned synthetic‑acc = {acc_syn:.4f}   "
                     f"(untuned baseline {baseline_acc_untuned.get(m, 'NA'):.4f})")

# Now run with Optuna tuning
log.info("=== Running with hyperparameter tuning using Optuna ===")
for n in SIZES:
    log.info(f"=== n_samples = {n} (tuned) ===")
    run_dir = os.path.join(OUTPUT_ROOT, f"n_{n}_tuned")
    res = run_evaluation_pipeline(
        dataset_path = DATASET_PATH,
        output_dir   = run_dir,
        test_size    = 0.20,
        n_samples    = n,
        models       = MODELS,
        random_state = 42,
        tune_hyperparams = TUNE_HYPERPARAMS,
        n_trials = N_TRIALS
    )
    if res is None:
        log.warning(f" run_evaluation_pipeline returned None for n={n} (tuned)")
        continue

    # --- grab original & synthetic accuracies with tuning ------------------
    orig  = res["original_results"]
    synth = res["synthetic_results"]

    for m in MODELS:
        # store tuned baseline once
        if m not in baseline_acc_tuned and "accuracy" in orig.get(m, {}):
            baseline_acc_tuned[m] = orig[m]["accuracy"]

        # store tuned synthetic accuracy for this n
        acc_syn = synth.get(m, {}).get("accuracy")
        if acc_syn is not None:
            size_acc_syn_tuned[m].append((n, acc_syn))
            log.info(f"  {m:8s}  tuned synthetic‑acc = {acc_syn:.4f}   "
                     f"(tuned baseline {baseline_acc_tuned.get(m, 'NA'):.4f})")

# Create a plots directory for individual model graphs
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "model_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- INDIVIDUAL MODEL PLOTS WITH LOG SCALE ---------------------------------
log.info("Generating individual model plots with logarithmic scaling")

for m in MODELS:
    if not (m in baseline_acc_tuned and m in baseline_acc_untuned and 
            size_acc_syn_tuned[m] and size_acc_syn_untuned[m]):
        log.warning(f"Skipping plot for {m} due to insufficient data")
        continue
        
    plt.figure(figsize=(10, 6))
    
    # Plot tuned and untuned synthetic data results
    pairs_tuned = sorted(size_acc_syn_tuned[m], key=lambda x: x[0])
    ns_tuned, accs_tuned = zip(*pairs_tuned)
    
    pairs_untuned = sorted(size_acc_syn_untuned[m], key=lambda x: x[0])
    ns_untuned, accs_untuned = zip(*pairs_untuned)
    
    # Plot the synthetic data results with log scale
    plt.semilogx(ns_tuned, accs_tuned, 'o-', label=f'Synthetic Data (Tuned)', linewidth=2)
    plt.semilogx(ns_untuned, accs_untuned, 'x--', label=f'Synthetic Data (Untuned)', alpha=0.7)
    
    # Add baseline references
    base_tuned = baseline_acc_tuned[m]
    base_untuned = baseline_acc_untuned[m]
    
    plt.axhline(y=base_tuned, color='green', linestyle='-', 
                label=f'Original Data (Tuned): {base_tuned:.4f}')
    plt.axhline(y=base_untuned, color='red', linestyle='--', 
                label=f'Original Data (Untuned): {base_untuned:.4f}')
    
    # Calculate the improvement from tuning
    improvements = []
    untuned_dict = {n: acc for n, acc in size_acc_syn_untuned.get(m, [])}
    tuned_dict = {n: acc for n, acc in size_acc_syn_tuned.get(m, [])}
    common_sizes = sorted(set(untuned_dict.keys()) & set(tuned_dict.keys()))
    
    if common_sizes:
        for n in common_sizes:
            improvement = tuned_dict[n] - untuned_dict[n]
            improvement_pct = (improvement / untuned_dict[n]) * 100
            plt.annotate(f"+{improvement_pct:.1f}%", 
                         xy=(n, tuned_dict[n]), 
                         xytext=(0, 10),
                         textcoords="offset points",
                         ha='center', 
                         fontsize=8,
                         color='green' if improvement > 0 else 'red')
    
    # Add parity line showing where synthetic performance equals original
    plt.axhline(y=base_tuned, color='green', linestyle=':', alpha=0.5)
    
    # Format the plot
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(f'{m.upper()} Performance: Original vs Synthetic Data with Tuning')
    plt.xlabel('Synthetic Dataset Size (log scale)')
    plt.ylabel('Accuracy on Test Set')
    plt.legend(loc='best')
    
    # Set better x-ticks for log scale
    plt.xscale('log', base=2)
    plt.xticks(SIZES, [str(s) for s in SIZES])
    
    # Set y-axis to start from a reasonable minimum
    all_accs = list(accs_tuned) + list(accs_untuned) + [base_tuned, base_untuned]
    min_acc = max(0, min(all_accs) - 0.05)
    max_acc = min(1.0, max(all_accs) + 0.05)
    plt.ylim(min_acc, max_acc)
    
    # Save the plot
    model_plot_path = os.path.join(PLOTS_DIR, f"{m}_performance.png")
    plt.savefig(model_plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {m} performance plot → {model_plot_path}")

# --- TUNING IMPACT BY DATASET SIZE (SEPARATE PLOTS) -----------------------
for m in MODELS:
    improvements = []
    untuned_dict = {n: acc for n, acc in size_acc_syn_untuned.get(m, [])}
    tuned_dict = {n: acc for n, acc in size_acc_syn_tuned.get(m, [])}
    common_sizes = sorted(set(untuned_dict.keys()) & set(tuned_dict.keys()))
    
    if not common_sizes:
        continue
        
    improvements = [(n, tuned_dict[n] - untuned_dict[n]) for n in common_sizes]
    ns, diffs = zip(*improvements)
    
    plt.figure(figsize=(8, 5))
    plt.semilogx(ns, diffs, 'o-', linewidth=2)
    
    # Add percentage labels
    for n, diff in zip(ns, diffs):
        if untuned_dict[n] != 0:  # Protect against division by zero
            pct = (diff / untuned_dict[n]) * 100
            plt.annotate(f"{pct:+.1f}%", 
                         xy=(n, diff), 
                         xytext=(0, 5 if diff > 0 else -15),
                         textcoords="offset points",
                         ha='center',
                         color='green' if diff > 0 else 'red')
        else:
            # Handle the division by zero case
            plt.annotate("∞%", 
                         xy=(n, diff), 
                         xytext=(0, 5 if diff > 0 else -15),
                         textcoords="offset points",
                         ha='center',
                         color='green' if diff > 0 else 'red')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(f'Tuning Impact on {m.upper()} by Dataset Size')
    plt.xlabel('Synthetic Dataset Size (log scale)')
    plt.ylabel('Accuracy Improvement from Tuning')
    
    # Set better x-ticks for log scale
    plt.xscale('log', base=2)
    plt.xticks(ns, [str(n) for n in ns])
    
    # Save the plot
    impact_plot_path = os.path.join(PLOTS_DIR, f"{m}_tuning_impact.png")
    plt.savefig(impact_plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {m} tuning impact plot → {impact_plot_path}")

# --- COMBINED PLOT WITH ALL MODELS (TUNED ONLY) ---------------------------
plt.figure(figsize=(12, 8))

# Use a distinct color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']

for i, m in enumerate(MODELS):
    if not size_acc_syn_tuned[m] or m not in baseline_acc_tuned:
        continue
        
    pairs = sorted(size_acc_syn_tuned[m], key=lambda x: x[0])
    ns, accs = zip(*pairs)
    
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    plt.semilogx(ns, accs, marker=marker, linestyle='-', color=color, 
             label=f"{m}", linewidth=2, markersize=8)
    
    # Add baseline reference as horizontal line with same color
    base = baseline_acc_tuned[m]
    plt.axhline(y=base, color=color, linestyle=':', alpha=0.5)
    
    # Add label for baseline
    plt.text(ns[-1] * 1.05, base, f"{m} baseline ({base:.3f})", 
             color=color, va='center', fontsize=8)

plt.xlabel("Synthetic Dataset Size (log scale)")
plt.ylabel("Accuracy (Tuned Models)")
plt.title("All Models: Accuracy vs Synthetic Dataset Size")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(loc='lower right')

# Set better x-ticks for log scale
plt.xscale('log', base=2)
plt.xticks(SIZES, [str(s) for s in SIZES])

combined_plot_path = os.path.join(OUTPUT_ROOT, "all_models_log_scale.png")
plt.savefig(combined_plot_path, dpi=120, bbox_inches='tight')
plt.close()
log.info(f"Combined plot saved → {combined_plot_path}")

# --- SCALING EFFICIENCY METRIC -------------------------------------------
# Calculate how many synthetic samples are needed to match original performance
plt.figure(figsize=(10, 6))
efficiency_data = []

for m in MODELS:
    if not size_acc_syn_tuned[m] or m not in baseline_acc_tuned:
        continue
        
    base_acc = baseline_acc_tuned[m]
    pairs = sorted(size_acc_syn_tuned[m], key=lambda x: x[0])
    
    # Find the first size where synthetic performance matches or exceeds baseline
    match_size = None
    for n, acc in pairs:
        if acc >= base_acc:
            match_size = n
            break
    
    if match_size:
        efficiency_data.append((m, match_size, base_acc))
        plt.bar(m, match_size, label=f"{m} ({base_acc:.3f})")
        plt.text(m, match_size + 50, f"{match_size}", ha='center')
    else:
        largest_size = pairs[-1][0] if pairs else 0
        plt.bar(m, largest_size, color='lightgray', 
                label=f"{m} (never matches {base_acc:.3f})")
        plt.text(m, largest_size + 50, "never\nmatches", ha='center')

plt.yscale('log')
plt.ylabel("Synthetic Samples Needed to Match Original Performance")
plt.title("Synthetic Data Efficiency by Model")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

efficiency_plot_path = os.path.join(OUTPUT_ROOT, "synthetic_efficiency.png")
plt.savefig(efficiency_plot_path, dpi=120, bbox_inches='tight')
plt.close()
log.info(f"Efficiency plot saved → {efficiency_plot_path}")

log.info("Scaling pipeline with individual model plots completed successfully")
