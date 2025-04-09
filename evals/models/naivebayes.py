import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_model(X_train, Y_train, X_test, Y_test):
    """
    Standard interface for the pipeline.
    This function trains a Gaussian Naive Bayes classifier on the provided data (X, y)
    and returns predictions on the same data.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        
    Returns:
        np.array: Predictions on X.
    """
    # Assume X is already processed; if additional processing is needed, add it here
    nb_model = GaussianNB(var_smoothing=1e-9)
    nb_model.fit(X_train, Y_train)
    return nb_model.predict(X_test)

if __name__ == "__main__":
    # Standalone execution for testing the Naive Bayes model.
    # Load and process the data
    df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
    df = process_data(df)
    
    print("Data snapshot:")
    print(df.head())
    print(df.info())
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # For standalone testing, perform a train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Naive Bayes model on the training data
    nb_model = GaussianNB(var_smoothing=1e-9)
    nb_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = nb_model.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Get class probabilities for a deeper look
    y_prob = nb_model.predict_proba(X_test)
    print("\nSample of class probabilities (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"Example {i+1}: Class 0 prob: {y_prob[i][0]:.4f}, Class 1 prob: {y_prob[i][1]:.4f}, True class: {y_test.iloc[i]}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Naive Bayes Confusion Matrix')
    plt.savefig('naivebayes_confusion_matrix.png')
    plt.close()
    
    # Optional: Display feature means by class (Naive Bayes internal parameters)
    print("\nFeature means by class:")
    feature_means = {}
    for i, class_label in enumerate([0, 1]):
        feature_means[f"Class {class_label}"] = pd.Series(nb_model.theta_[i], index=X.columns)
    
    feature_means_df = pd.DataFrame(feature_means)
    feature_means_df['Difference'] = abs(feature_means_df['Class 0'] - feature_means_df['Class 1'])
    feature_means_df = feature_means_df.sort_values('Difference', ascending=False)
    print(feature_means_df)
