import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and process the data
df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
df = process_data(df)
X = df.drop('class', axis=1)
y = df['class']

# Important: MLP requires feature scaling for optimal performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Determine hidden layer size ('a' in Weka means (attributes + classes)/2)
n_features = X.shape[1]
n_classes = len(np.unique(y))
hidden_layer_size = int((n_features + n_classes) / 2)

print(f"Number of features: {n_features}")
print(f"Number of classes: {n_classes}")
print(f"Hidden layer size ('a' in Weka): {hidden_layer_size}")

# Train the MLP model with Weka equivalent parameters
# hidden layers: 'a' -> (attributes + classes)/2
# learning rate = 0.3
# momentum = 0.2
# training time = 500 epochs
mlp_model = MLPClassifier(
    hidden_layer_sizes=(hidden_layer_size,),  # Single layer with 'a' neurons
    activation='logistic',                    # Weka default uses sigmoid/logistic
    solver='sgd',                             # Stochastic gradient descent (like Weka)
    learning_rate_init=0.3,                   # Learning rate as specified
    momentum=0.2,                             # Momentum as specified
    max_iter=500,                             # Training time/epochs as specified
    random_state=42,
    early_stopping=False,                     # Don't stop early (match Weka behavior)
    learning_rate='constant',                 # Constant learning rate (like Weka)
    verbose=True                              # Show progress during training
)

# Train the model
print("Training MLP model...")
mlp_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred, zero_division=1))

# Check class distribution in predictions
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
y_prob = mlp_model.predict_proba(X_test)
print(f"\nSample of class probabilities (first 5):")
for i in range(min(5, len(y_test))):
    print(f"Example {i+1}: Class 0 prob: {y_prob[i][0]:.4f}, Class 1 prob: {y_prob[i][1]:.4f}, True class: {y_test.iloc[i]}")

# Optional: Analyze feature importance (via permutation importance)
# MLP doesn't have built-in feature importance, so we'll use permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(
    mlp_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
})

importance = importance.sort_values('Importance', ascending=False)
print("\nFeature importance (via permutation importance):")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('MLP Feature Importance')
plt.tight_layout()
plt.savefig('mlp_feature_importance.png')
plt.close() 