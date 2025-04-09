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
size_vs_metric.py
-----------------
Loop over different n_samples values, call run_evaluation_pipeline each time,
collect accuracy for every model, and plot accuracy vs. synthetic‑set size.

Place this file next to basic_eval_pipeline.py and run:
    python size_vs_metric.py
"""

import os, logging, matplotlib.pyplot as plt
from basic_eval_pipeline import run_evaluation_pipeline   # <-- your function

# ---------- CONFIG ----------
DATASET_PATH = "./evals/dataset/andrew_diabetes.csv"
OUTPUT_ROOT  = "./results/size_experiments"
MODELS       = ["knn", "mlp"]          # pick any subset of your models
SIZES        = [100, 200, 400, 800, 1200, 1600, 2000]   # synthetic rows to generate
# -----------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("size‑driver")

# --- Step 3‑4: run experiments & gather accuracies ---------------------------
size_acc_syn   = {m: [] for m in MODELS}   # (n, synthetic‑acc)
baseline_acc   = {}                        # original‑data accuracy (one per model)

for n in SIZES:
    log.info(f"=== n_samples = {n} ===")
    run_dir = os.path.join(OUTPUT_ROOT, f"n_{n}")
    res = run_evaluation_pipeline(
        dataset_path = DATASET_PATH,
        output_dir   = run_dir,
        test_size    = 0.20,
        n_samples    = n,
        models       = MODELS,
        random_state = 42,
    )
    if res is None:
        log.warning(f" run_evaluation_pipeline returned None for n={n}")
        continue

    # --- grab original & synthetic accuracies --------------------------------
    orig  = res["original_results"]
    synth = res["synthetic_results"]

    for m in MODELS:
        # store baseline once
        if m not in baseline_acc and "accuracy" in orig.get(m, {}):
            baseline_acc[m] = orig[m]["accuracy"]

        # store synthetic accuracy for this n
        acc_syn = synth.get(m, {}).get("accuracy")
        if acc_syn is not None:
            size_acc_syn[m].append((n, acc_syn))
            log.info(f"  {m:8s}  synthetic‑acc = {acc_syn:.4f}   "
                     f"(baseline {baseline_acc.get(m, 'NA'):.4f})")

# --- PLOT --------------------------------------------------------------------
plt.figure(figsize=(10,6))

for m in MODELS:
    if not size_acc_syn[m] or m not in baseline_acc:
        continue

    # synthetic curve
    pairs = sorted(size_acc_syn[m], key=lambda x: x[0])
    ns, accs = zip(*pairs)
    plt.scatter(ns, accs, s=80, label=f"{m} (synthetic)")
    plt.plot(ns, accs, linestyle="--")

    # baseline (original‑data) horizontal line
    base = baseline_acc[m]
    plt.hlines(base, ns[0], ns[-1],
               colors="gray", linestyles="-",
               label=f"{m} (original)  acc={base:.3f}")

plt.xlabel("Synthetic‑set size (n_samples)")
plt.ylabel("Accuracy on held‑out test set")
plt.title("Original‑ vs Synthetic‑trained model accuracy")
plt.grid(True)
plt.legend()
plot_path = os.path.join(OUTPUT_ROOT, "size_vs_accuracy_with_baseline.png")
plt.savefig(plot_path, dpi=120)
plt.show()
log.info(f"Plot saved → {plot_path}")
