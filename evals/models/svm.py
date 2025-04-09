import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_model(X_train, Y_train, X_test, Y_test):
    """
    Standard interface for the pipeline.
    Trains an SVM model using the provided data (X, y) and returns predictions on the same data.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Labels.
    
    Returns:
        np.array: Predictions on X.
    """
    # Train the SVM model with your specified parameters
    svm_model = SVC(
        kernel='rbf',              # Radial basis function kernel
        gamma='scale',             # 'scale' is typically better than a fixed 0.0
        tol=0.001,                 # Convergence tolerance, equivalent to Weka's eps = 0.001
        C=10.0,                    # Inverse of loss parameter (1/0.1 = 10.0)
        class_weight='balanced',   # Handle class imbalance
        probability=True,          # Enable probability estimates
        random_state=42
    )
    svm_model.fit(X_train, Y_train)
    return svm_model.predict(X_test)

if __name__ == "__main__":
    # Standalone execution for testing/evaluation of the SVM model.
    
    # Load and process the data
    df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
    df = process_data(df)
    
    print("Data snapshot:")
    print(df.head())
    df.info()
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # For standalone testing, perform a train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the SVM model on the training data
    svm_model = SVC(
        kernel='rbf',
        gamma='scale',
        tol=0.001,
        C=10.0,
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    print("Training SVM model...")
    svm_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM Confusion Matrix')
    plt.savefig('svm_confusion_matrix.png')
    plt.close()
