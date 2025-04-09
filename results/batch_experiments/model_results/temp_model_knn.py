
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
import json
import csv

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
project_root = "/home/somneel/CS3264"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evals", "models"))
print("Python path after additions:", sys.path)

try:
    from evals.models.knn import *
    print("Successfully imported knn model")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load training and test data
df_train = pd.read_csv("./results/batch_experiments/model_results/synthetic_batch_4_train_data.csv", sep=';')
df_test = pd.read_csv("./results/batch_experiments/model_results/synthetic_batch_4_test_data.csv", sep=';')

print("Train data shape:", df_train.shape)
print("Test data shape:", df_test.shape)

# Determine target column (assumed to be the last column)
label_column = df_train.columns[-1]
X_train = df_train.drop(label_column, axis=1)
y_train = df_train[label_column]
X_test = df_test.drop(label_column, axis=1)
y_test = df_test[label_column]

print("X_train columns:", X_train.columns.tolist())
print("X_train shape:", X_train.shape)
print("Test labels count:", len(y_test))

# Train the model and get predictions
try:
    print("Calling run_model with training and test data...")
    # This assumes that run_model accepts (X_train, y_train, X_test, y_test)
    y_pred = run_model(X_train, y_train, X_test, y_test)
    print("run_model completed successfully!")
except Exception as e:
    import traceback
    print(f"Error in run_model: {type(e).__name__}: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}

with open("./results/batch_experiments/model_results/synthetic_batch_4_knn_metrics.json", 'w') as f_out:
    json.dump(metrics, f_out, indent=2)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('knn Confusion Matrix (synthetic_batch_4 data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("./results/batch_experiments/model_results/synthetic_batch_4_knn_confusion.png")
plt.close()

print(f"Accuracy: {metrics['accuracy']:.4f}")
cr = metrics['classification_report']
weighted_avg = cr['weighted avg']
print(f"Precision: {weighted_avg['precision']:.4f}")
print(f"Recall: {weighted_avg['recall']:.4f}")
print(f"F1 Score: {weighted_avg['f1-score']:.4f}")

with open("./results/batch_experiments/synthetic_batch_4_metrics.csv", 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        'knn', 
        f"{metrics['accuracy']:.4f}", 
        f"{weighted_avg['precision']:.4f}", 
        f"{weighted_avg['recall']:.4f}", 
        f"{weighted_avg['f1-score']:.4f}"
    ])
