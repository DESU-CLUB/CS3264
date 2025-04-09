import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend to allow saving figures in parallel
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json


from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import LinkabilityEvaluator
from anonymeter.evaluators import InferenceEvaluator

# create a file handler
file_handler = logging.FileHandler("anonymeter.log")

# set the logging level for the file handler
file_handler.setLevel(logging.DEBUG)

# add the file handler to the logger
logger = logging.getLogger("anonymeter")
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# Fetch data
def fetch_data(dataset_path):
    # We need 3 datasets: Original, Synthetic, Control.
    train_path = os.path.join(dataset_path, "data", "train_data.csv")
    test_path = os.path.join(dataset_path, "data", "test_data.csv")
    synthetic_path = os.path.join(dataset_path, "data", "synthetic_data.csv")

    ori = pd.read_csv(train_path, sep=";")
    syn = pd.read_csv(synthetic_path, sep=";")
    control = pd.read_csv(test_path, sep=";")

    logger.info("\nori length = " + str(len(ori)))
    logger.info("\nsyn length = " + str(len(syn)))
    logger.info("\ncontrol length = " + str(len(control)))

    return ori, syn, control


# Evaluate the Singling Out risk
def singling_out_risk(ori, syn, control, output_dir="./results"):
    n = min(500, len(ori))  # ensure sample size is valid
    pass


# Evaluate the Linkability risk
def linkability_risk(ori, syn, control, aux_cols, run_num, output_dir="./results"):
    n = min(2000, len(ori), len(syn), len(control))  # ensure sample size is valid

    logger.info("Linkability risk sample size =" + str(n))
    logger.info("aux cols =" + str(aux_cols))

    evaluator = LinkabilityEvaluator(ori=ori, 
                                     syn=syn, 
                                     control=control,
                                     n_attacks=n,
                                     aux_cols=aux_cols,
                                     n_neighbors=10)

    evaluator.evaluate(n_jobs=-2)  # n_jobs follow joblib convention. -1 = all cores, -2 = all execept one
    evaluator.risk()
    res = evaluator.results()

    logger.info("Linkability risk evaluator done")

    # Collect the results in a dictionary
    results = {
        "main_attack_rate": {"main_attack_rate_value": res.attack_rate.value, "main_attack_rate_error": res.attack_rate.error},
        "baseline_attack_rate": {"baseline_attack_rate_value": res.baseline_rate.value, "baseline_attack_rate_error": res.baseline_rate.error},
        "control_attack_rate": {"control_attack_rate_value": res.control_rate.value, "control_attack_rate_error": res.control_rate.error},
        "n_neighbors_7_risk": {"n_neighbors_7_risk_value": evaluator.risk(n_neighbors=7).value,
                                "n_neighbors_7_risk_ci_lower": evaluator.risk(n_neighbors=7).ci[0],
                                "n_neighbors_7_risk_ci_upper": evaluator.risk(n_neighbors=7).ci[1]}
    }

    logger.info("Success rate of main attack: " + str(res.attack_rate))
    logger.info("Success rate of baseline attack: " + str(res.baseline_rate))
    logger.info("Success rate of control attack: " + str(res.control_rate))

    # This choooses the top n neighbours as successfully linking the 2 datasets: can choose a lower n.
    logger.info("risk = " + str(evaluator.risk(n_neighbors=7)))

    # Ensure the directory exists before saving the file
    output_dir_full = os.path.join(output_dir, "privacy_results")
    os.makedirs(output_dir_full, exist_ok=True)

    # Save the results to a JSON file in the privacy_results folder
    output_path = os.path.join(output_dir, "privacy_results", f"linkability_risk_results_{run_num}" + ".json")
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    logger.info(f"Linkability risk results saved to {output_path}")
    return results


# Evaluate the Inference risk
def inference_risk(ori, syn, control, run_num, output_dir="./results"):
    n = min(1000, len(ori), len(syn), len(control))  # ensure sample size is valid

    logger.info("Inference risk sample size =" + str(n))

    columns = ori.columns
    results = []

    for secret in columns:
    
        aux_cols = [col for col in columns if col != secret]
    
        # This evaluator is non deterministic! Have to run multiple times. Weird bug when running in for loop? 
        evaluator = InferenceEvaluator(ori=ori, 
                                       syn=syn, 
                                       control=control,
                                       aux_cols=aux_cols,
                                       secret=secret,
                                       n_attacks=n)
        evaluator.evaluate(n_jobs=-2)
        results.append((secret, evaluator.results()))
    
    logger.info("Inference risk evaluator done")

    # Visualise results and save them
    fig, ax = plt.subplots()

    risks = [res[1].risk().value for res in results]
    columns = [res[0] for res in results]

    ax.bar(x=columns, height=risks, alpha=0.5, ecolor='black', capsize=10)

    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Measured inference risk")
    _ = ax.set_xlabel("Secret column")

    inference_risk_plot_path = os.path.join(output_dir, "plots", "inference_risk"+ str(run_num) + ".png")
    plt.tight_layout()           # auto-adjusts spacing to prevent clipping
    plt.savefig(inference_risk_plot_path)
    logger.info(f"Inference risk results saved to {inference_risk_plot_path}")
    plt.close()


# Evaluate for one set of generated data vs original data
'''Checking how many of these guesses are correct, the success rates of the different attacks are measured and used to derive an estimate of the privacy risk. 
In particular, the "control attack" is used to separate what the attacker learns from the utility of the synthetic data, and what is instead indication of privacy leaks. 
The "baseline attack" instead functions as a sanity check. The "main attack" attack should outperform random guessing in order for the results to be trusted.'''
def anonymeter_eval(dataset_path="./results"):
    # Get datasets
    ori, syn, control = fetch_data(dataset_path)
    
    logger.info(ori.head()) # log the aux_cols

    aux_cols_all = ori.columns.tolist()

    logger.info("Features: " + str(aux_cols_all))

    # Evaluate singling out risk
    # singling_out_risk(ori, syn, control)


    # Evaluate linkability risk. Need more research on what columns to choose for effective evaluation.
    # Scenario 1: low linkability
    aux_cols_one = [
    [aux_cols_all[0]],      # Psuedo Dataset A: Age only
    [aux_cols_all[1]]       # Psuedo Dataset B: Gender only
    ]
    # Scenario 2: mid linkability
    aux_cols_two = [
    [aux_cols_all[0], aux_cols_all[1], aux_cols_all[2], aux_cols_all[14]],      # Psuedo Dataset A: Age, Gender, Polyuria, Alopecia
    [aux_cols_all[0], aux_cols_all[1], aux_cols_all[3], aux_cols_all[15]]       # Psuedo Dataset B: Age, Gender, Polydipsia, Obesity
    ]
    # Scenario 3: high linkability
    aux_cols_three = [
    [aux_cols_all[0], aux_cols_all[1], aux_cols_all[2], aux_cols_all[6], aux_cols_all[14]],      # Psuedo Dataset A: Age, Gender, Polyuria, Polyphagia Alopecia
    [aux_cols_all[0], aux_cols_all[1], aux_cols_all[3], aux_cols_all[8], aux_cols_all[15]]       # Psuedo Dataset B: Age, Gender, Polydipsia, Visual blurring, Obesity
    ]
    # Evaluate different scenarios for linkability
    aux_col_labels = ["low", "mid", "high"]
    aux_col_settings = [aux_cols_one, aux_cols_two, aux_cols_three]
    for j in range(3):
        linkability_risk(ori, syn, control, aux_col_settings[j], aux_col_labels[j])

    # Evaluate inference risk
    # for k in range(3):
        # inference_risk(ori, syn, control, k)
    

if __name__ == "__main__":
    logger.info("*** Starting Anonymeter evaluation ***")
    anonymeter_eval()
    logger.info("\n*** Anonymeter Evaluation completed ***")
    logger.info("-----------------------------------------------------------------------------"
    "----------------------------------------------------------------------------------------")