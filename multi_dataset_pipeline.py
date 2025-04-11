"""
multi_dataset_driver.py

Loops over multiple datasets, calls run_evaluation_pipeline on each,
and collects/prints the performance results for original vs synthetic data.
"""

import csv
import os
import logging
import pandas as pd
from basic_eval_pipeline import run_evaluation_pipeline  # <-- your pipeline code
from typing import Dict, Any
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DATASETS = {
    #"./evals/dataset/andrew_diabetes.csv" : ";",
    "./evals/dataset/diabetes_prediction_dataset.csv":";",
    "./evals/dataset/pima-diabetes.csv":";",
}

MODELS = ["knn", "mlp", "naivebayes", "randomforest", "sgd", "svm"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    # Prepare a results dictionary to store performance across datasets
    all_results = {}  # structure: { dataset_name: pipeline_output }

    for dataset_path,seperator in DATASETS.items():
        # Create an output directory per dataset, named after the dataset (minus file extension, for instance)
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        out_dir   = f"./multi_dataset_results/{base_name}_eval"
        logger.info(f"=== Running pipeline for dataset: {dataset_path} ===")
        # if seperator == ",":
        #     train_df = pd.read_csv(dataset_path, sep=",")
        #     train_df.to_csv(dataset_path,
        #   sep=";",
        #   index=False,
        #   quoting=csv.QUOTE_NONE,
        #   escapechar='\\')
            
        # You can set n_samples to, e.g., None (which defaults to len(train_df)),
        # or choose a specific number depending on your experiment.
        # Or loop over multiple n_samples for each dataset if desired.
        results = run_evaluation_pipeline(
            dataset_path=dataset_path,
            output_dir=out_dir,
            test_size=TEST_SIZE,
            n_samples=None,   # or any specific number of synthetic samples you want
            models=MODELS,
            random_state=RANDOM_STATE
        )

        if results:
            all_results[base_name] = results
            logger.info(f"Finished pipeline for dataset: {base_name}")
        else:
            logger.error(f"Pipeline returned None for dataset: {base_name}")

    # Now, all_results holds the pipeline outputs for each dataset.
    # For demonstration, we'll just log the accuracies for each model's original vs synthetic performance.
    # You can easily adapt this to create bar charts, CSV comparisons, etc.

    for dataset_name, pipeline_output in all_results.items():
        logger.info(f"\n=== Performance Summary for dataset: {dataset_name} ===")
        
        original   = pipeline_output["original_results"]
        synthetic  = pipeline_output["synthetic_results"]
        comparison = pipeline_output["comparison"]  # If you want the direct difference

        # Print out or log original vs synthetic accuracy per model
        for model_name in MODELS:
            orig_acc  = original.get(model_name, {}).get("accuracy", None)
            synth_acc = synthetic.get(model_name, {}).get("accuracy", None)
            if orig_acc is not None and synth_acc is not None:
                logger.info(
                    f"{model_name} -> Original: {orig_acc:.4f}, Synthetic: {synth_acc:.4f}, "
                    f"Difference: {synth_acc - orig_acc:.4f}"
                )
            else:
                logger.warning(f"No results for model {model_name} in dataset {dataset_name}")

if __name__ == "__main__":
    main()
