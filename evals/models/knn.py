import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load and process the data
df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
df = process_data(df)
X = df.drop('class', axis=1)
y = df['class']

# Important: KNN may benefit from feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the KNN model with Weka equivalent parameters:
# - K = 1 (n_neighbors=1)
# - Search algorithm: LinearNNSearch with Euclidean distance (using algorithm='brute', metric='euclidean')
knn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='euclidean')
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=1))

# Check class distribution in predictions and test set
print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# Get prediction probabilities for a deeper look (if available)
if hasattr(knn_model, "predict_proba"):
    y_prob = knn_model.predict_proba(X_test)
    print(f"\nSample of class probabilities (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"Example {i+1}: Class 0 prob: {y_prob[i][0]:.4f}, "
              f"Class 1 prob: {y_prob[i][1]:.4f}, True class: {y_test.iloc[i]}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN (K=1) Confusion Matrix')
plt.savefig('knn_confusion_matrix.png')
plt.close()

# Compute permutation importance to assess feature importance
result = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

print("\nFeature importance (via permutation importance):")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('KNN Feature Importance')
plt.tight_layout()
plt.savefig('knn_feature_importance.png')
plt.close()
