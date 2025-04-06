import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data

df = pd.read_csv('../dataset/andrew_diabetes.csv',sep=';')

df.head()

df.info()

#Train a SVM model

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
df = process_data(df)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the SVM model
svm_model = SVC(
    kernel='rbf',         # radial basis function kernel as specified
    gamma='scale',        # 'scale' works better than 0.0 for most cases
    tol=0.001,            # equivalent to Weka's eps = 0.001
    C=10.0,               # inverse of loss (1/0.1 = 10.0)
    class_weight='balanced', # Handle class imbalance
    probability=True,     # Enable probability estimates
    random_state=42
)
svm_model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = svm_model.predict(X_test)

#Evaluate the model
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
plt.savefig('svm_confusion_matrix.png')
plt.close()




