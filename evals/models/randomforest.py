import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data

df = pd.read_csv('../dataset/andrew_diabetes.csv',sep=';')

df.head()

df.info()

#Train a random forest model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Split the data into training and testing sets 
df = process_data(df)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,       # Set to None instead of 0 for unlimited depth
    max_features='sqrt',  # Standard setting for classification tasks
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))







