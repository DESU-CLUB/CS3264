import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np

def run_model(X_train, y_train, X_test, y_test):
    """
    Standard interface for the pipeline.
    This function scales the features, trains an MLP model using X_train and y_train,
    and returns predictions on X_test.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels (not used in training).
        
    Returns:
        np.array: Predicted labels for X_test.
    """
    # Scale the features for both training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine the hidden layer size: ('a' in Weka means (attributes + classes) / 2)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    hidden_layer_size = int((n_features + n_classes) / 2)
    
    # Initialize the MLPClassifier with parameters analogous to Weka defaults
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),  # single hidden layer with 'a' neurons
        activation='logistic',                     # sigmoid activation
        solver='sgd',                              # stochastic gradient descent
        learning_rate_init=0.3,                    # learning rate as specified
        momentum=0.2,                              # momentum as specified
        max_iter=500,                              # number of iterations (epochs)
        random_state=42,
        early_stopping=False,
        learning_rate='constant',
        verbose=False
    )
    
    # Train the model using the training data
    mlp_model.fit(X_train_scaled, y_train)
    
    # Return predictions on the test data
    return mlp_model.predict(X_test_scaled)


if __name__ == "__main__":
    # Standalone execution for testing/evaluation purposes.
    
    # Load and process the data
    df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
    df = process_data(df)
    X = df.drop('class', axis=1)
    y = df['class']
    
    print("Data snapshot:")
    print(df.head())
    
    # For standalone testing, perform a train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the training and testing data separately
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine hidden layer size based on training data
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    hidden_layer_size = int((n_features + n_classes) / 2)
    
    # Train the MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=0.3,
        momentum=0.2,
        max_iter=500,
        random_state=42,
        early_stopping=False,
        learning_rate='constant',
        verbose=True
    )
    
    print("Training MLP model...")
    mlp_model.fit(X_train_scaled, y_train)
    y_pred = mlp_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_model.loss_curve_)
    plt.title('MLP Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('mlp_learning_curve.png')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('MLP Confusion Matrix')
    plt.savefig('mlp_confusion_matrix.png')
    plt.close()
    
    # Get prediction probabilities
    y_prob = mlp_model.predict_proba(X_test_scaled)
    print(f"\nSample of class probabilities (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"Example {i+1}: Class 0 prob: {y_prob[i][0]:.4f}, Class 1 prob: {y_prob[i][1]:.4f}, True class: {y_test.iloc[i]}")
    
    # Analyze feature importance via permutation importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        mlp_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature importance (via permutation importance):")
    print(importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('MLP Feature Importance')
    plt.tight_layout()
    plt.savefig('mlp_feature_importance.png')
    plt.close()
