import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Need to run this in the evals\models folder with the virtual env

# Load and process the data
df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
df = process_data(df)

print(df.head())

X = df.drop('class', axis=1)
y = df['class']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with K=1 and Euclidean distance (default)
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# # Evaluate performance
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # 10-fold cross-validation (using accuracy as scoring metric)
# cv_scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred, zero_division=1))

# Check class distribution in predictions
print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# Optional: Plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('KNN_confusion_matrix.png')
plt.close()
