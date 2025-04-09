import os
import numpy as np
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
def singling_out_risk(ori, syn, control, run_num, output_dir="./results"):
    # n = max(500, len(ori), len(syn), len(control))  # ensure sample size is valid
    n = 500

    logger.info("Singling Out risk sample size =" + str(n))
    
    evaluator = SinglingOutEvaluator(ori=ori, 
                                 syn=syn, 
                                 control=control,
                                 n_attacks=n)

    try:
        # Use univariate for faster runtime when evaluating large datasets, multivariate to evaluate more hidden relationships
        evaluator.evaluate(mode='univariate')
        risk = evaluator.risk(confidence_level=0.95)
        logger.info(f"univariate risk = {risk}")
        res = evaluator.results()
        logger.info("Successs rate of main attack:", res.attack_rate)
        logger.info("Successs rate of baseline attack:", res.baseline_rate)
        logger.info("Successs rate of control attack:", res.control_rate)

        # Collect the results in a dictionary
        results = {
            "main_attack_rate": {"main_attack_rate_value": res.attack_rate.value, "main_attack_rate_error": res.attack_rate.error},
            "baseline_attack_rate": {"baseline_attack_rate_value": res.baseline_rate.value, "baseline_attack_rate_error": res.baseline_rate.error},
            "control_attack_rate": {"control_attack_rate_value": res.control_rate.value, "control_attack_rate_error": res.control_rate.error},
            "univariate_risk": {"univariate_risk_risk_value": risk.value,
                                "univariate_risk_ci_lower": risk.ci[0],
                                "univariate_risk_ci_upper": risk.ci[1]}
        }

        # Ensure the directory exists before saving the file
        output_dir_full = os.path.join(output_dir, "privacy_results")
        os.makedirs(output_dir_full, exist_ok=True)

        # Save the results to a JSON file in the privacy_results folder
        output_path = os.path.join(output_dir_full, f"singling_out_risk_results_{run_num}" + ".json")
        with open(output_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        logger.info(f"Singling Out risk results saved to {output_path}")
    
        # Now create a graphic using the json file
        # Load the data from the JSON string
        with open(output_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract values for plotting
        attack_rates = {
            "Main Attack Rate": (data["main_attack_rate"]["main_attack_rate_value"], data["main_attack_rate"]["main_attack_rate_error"]),
            "Baseline Attack Rate": (data["baseline_attack_rate"]["baseline_attack_rate_value"], data["baseline_attack_rate"]["baseline_attack_rate_error"]),
            "Control Attack Rate": (data["control_attack_rate"]["control_attack_rate_value"], data["control_attack_rate"]["control_attack_rate_error"]),
        }

        # Extract univariate risk values and confidence intervals# Extract univariate risk values and confidence intervals
        univariate_risk = {
            "Risk Value": data["univariate_risk"]["univariate_risk_risk_value"],
            "CI Lower": data["univariate_risk"]["univariate_risk_ci_lower"],
            "CI Upper": data["univariate_risk"]["univariate_risk_ci_upper"]
        }

        # Plotting the bar plot for attack rates
        labels = list(attack_rates.keys())
        values = [attack_rates[label][0] for label in labels]
        errors = [attack_rates[label][1] for label in labels]
        x_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the bar plot for attack rates with error bars
        ax.bar(x_pos, values, yerr=errors, capsize=5, color='skyblue', label='Attack Rates')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Attack Rate')
        ax.set_title('Attack Rates with Errors')

        # Plotting the univariate risk with confidence intervals
        risk_value = univariate_risk["Risk Value"]
        ci_lower = univariate_risk["CI Lower"]
        ci_upper = univariate_risk["CI Upper"]
        
        ax.errorbar(x=1, y=risk_value, yerr=[[risk_value - ci_lower], [ci_upper - risk_value]], fmt='o', color='red', label='Univariate Risk')

        # Adding labels and legends
        ax.set_xlabel('Attack Types / Risk')
        ax.legend()

        # Save the plot as an image file
        singling_out_risk_plot_path = os.path.join(output_dir, "plots", "singling_out_risk_"+ str(run_num) + ".png")
        plt.tight_layout()
        plt.savefig(singling_out_risk_plot_path)
    
        # Inform user that the plot was saved
        logger.info(f"Plot saved to {singling_out_risk_plot_path}")


    except RuntimeError as ex: 
        logger.info(f"Singling out evaluation failed with {ex}. Please re-run this cell."
                "For more stable results increase `n_attacks`. Note that this will "
                "make the evaluation slower.")


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
    output_path = os.path.join(output_dir_full, f"linkability_risk_results_{run_num}" + ".json")
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    logger.info(f"Linkability risk results saved to {output_path}")
    
    # Now create a graphic using the json file
    # Load the data from the JSON string
    with open(output_path, 'r') as json_file:
        data = json.load(json_file)
    # Extracting values
    attack_rates = {
        "Main Attack Rate": (data["main_attack_rate"]["main_attack_rate_value"], data["main_attack_rate"]["main_attack_rate_error"]),
        "Baseline Attack Rate": (data["baseline_attack_rate"]["baseline_attack_rate_value"], data["baseline_attack_rate"]["baseline_attack_rate_error"]),
        "Control Attack Rate": (data["control_attack_rate"]["control_attack_rate_value"], data["control_attack_rate"]["control_attack_rate_error"]),
    }

    n_neighbors_7_risk = {
        "Risk Value": data["n_neighbors_7_risk"]["n_neighbors_7_risk_value"],
        "CI Lower": data["n_neighbors_7_risk"]["n_neighbors_7_risk_ci_lower"],
        "CI Upper": data["n_neighbors_7_risk"]["n_neighbors_7_risk_ci_upper"]
    }

    # Plotting attack rates with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart for attack rates
    categories = list(attack_rates.keys())
    values = [value[0] for value in attack_rates.values()]
    errors = [value[1] for value in attack_rates.values()]

    ax.bar(categories, values, yerr=errors, capsize=5, color='skyblue', label='Attack Rate', alpha=0.7)

    # Add the n_neighbors_7_risk (risk value with confidence intervals)
    ax.errorbar(
        "Risk Value", n_neighbors_7_risk["Risk Value"],
        yerr=[[n_neighbors_7_risk["Risk Value"] - n_neighbors_7_risk["CI Lower"]],
              [n_neighbors_7_risk["CI Upper"] - n_neighbors_7_risk["Risk Value"]]],
        fmt='o', color='red', label="Linkability Risk (n_neighbors=7)", markersize=8, capsize=5
    )

    # Add labels, title, and legend
    ax.set_xlabel('Attack Types')
    ax.set_ylabel('Rates / Risk')
    ax.set_title('Linkability Risk and Attack Rates')
    ax.legend()

    # Save the figure as a .png file
    linkability_risk_plot_path = os.path.join(output_dir, "plots", "linkability_risk_"+ str(run_num) + ".png")
    plt.tight_layout()
    plt.savefig(linkability_risk_plot_path)

    # Inform user that the plot was saved
    logger.info(f"Plot saved to {linkability_risk_plot_path}")


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
    for i in range(3):
        singling_out_risk(ori, syn, control, i)


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
    for k in range(3):
        inference_risk(ori, syn, control, k)
    

if __name__ == "__main__":
    logger.info("*** Starting Anonymeter evaluation ***")
    anonymeter_eval()
    logger.info("\n*** Anonymeter Evaluation completed ***")
    logger.info("-----------------------------------------------------------------------------"
    "----------------------------------------------------------------------------------------")
    