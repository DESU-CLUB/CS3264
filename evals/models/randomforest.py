import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_model(X, y):
    """
    Standard interface for the pipeline.
    Trains a RandomForestClassifier on the provided data and returns predictions on X.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Labels.
    
    Returns:
        np.array: Predictions for X.
    """
    # Initialize RandomForestClassifier with original parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,       # Unlimited depth
        max_features='sqrt',  # Use square root of the number of features
        random_state=42
    )
    # Train the model on the provided data
    rf_model.fit(X, y)
    # Return predictions on the same data
    return rf_model.predict(X)

if __name__ == "__main__":
    # Standalone execution for testing the Random Forest model.
    
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
    
    # Train the model on the training data
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('randomforest_confusion_matrix.png')
    plt.close()
