import pandas as pd
import numpy as np
from data_util import process_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_model(X, y):
    """
    Train a 1-NN classifier using Euclidean distance on the provided data,
    then return predictions on X.
    
    Args:
        X: Feature DataFrame (typically the test data)
        y: True labels (for training)
        
    Returns:
        numpy array of predictions
    """
    # Initialize and train the classifier on the provided data.
    knn_model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_model.fit(X, y)
    # Return predictions on the same data
    return knn_model.predict(X)

# For backward compatibility, also define a global 'model'
model = None

if __name__ == "__main__":
    # Standalone execution for testing the KNN model.
    # Load and process the data
    df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
    df = process_data(df)
    print("Data snapshot:")
    print(df.head())
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier and assign to global 'model'
    model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Check class distributions
    print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Plot confusion matrix (optional)
    cm = confusion_matrix(y_test, y_pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('KNN Confusion Matrix')
    plt.savefig('KNN_confusion_matrix.png')
    plt.close()
