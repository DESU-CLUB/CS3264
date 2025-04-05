import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import process_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Load and process the data
df = pd.read_csv('../dataset/andrew_diabetes.csv', sep=';')
df = process_data(df)
X = df.drop('class', axis=1)
y = df['class']

# Important: SGD requires feature scaling for optimal performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SGD model with Weka equivalent parameters
# epochs = 500
# epsilon = 0.001
# lambda = 10^-4
# learningRate = 0.01
# lossFunction: Hinge loss (SVM)
sgd_model = SGDClassifier(
    loss='hinge',         # Hinge loss (SVM)
    alpha=0.0001,         # lambda = 10^-4 (regularization parameter)
    epsilon=0.001,        # epsilon as specified
    eta0=0.01,            # Learning rate as specified
    learning_rate='constant',  # Constant learning rate
    max_iter=500,         # Training epochs as specified
    tol=0.001,            # Convergence tolerance
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    verbose=1             # Show progress during training
)

# Train the model
print("Training SGD model...")
sgd_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = sgd_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred, zero_division=1))

# Check class distribution in predictions
print(f"\nClass distribution in predictions: {np.bincount(y_pred)}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# Plot decision function values
if hasattr(sgd_model, "decision_function"):
    decision_values = sgd_model.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(decision_values, bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('SGD Decision Function Values')
    plt.xlabel('Decision Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('sgd_decision_values.png')
    plt.close()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SGD Confusion Matrix')
plt.savefig('sgd_confusion_matrix.png')
plt.close()

# Get model coefficients (feature weights)
feature_weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': sgd_model.coef_[0]
})

feature_weights['AbsWeight'] = abs(feature_weights['Weight'])
feature_weights = feature_weights.sort_values('AbsWeight', ascending=False)

print("\nFeature weights (importance):")
print(feature_weights[['Feature', 'Weight']])

# Plot feature weights
plt.figure(figsize=(10, 6))
sns.barplot(x='Weight', y='Feature', data=feature_weights)
plt.title('SGD Feature Weights')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('sgd_feature_weights.png')
plt.close()

# Compute learning curve on training data
print("\nComputing learning curve...")
n_samples = len(X_train)
train_errors = []
validation_errors = []

# Need to precompute class weights since 'balanced' isn't supported with partial_fit
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
print(f"Computed class weights: {class_weight_dict}")

# Create model for learning curve
sgd_curve = SGDClassifier(
    loss='hinge', 
    alpha=0.0001,         # lambda = 10^-4
    epsilon=0.001, 
    eta0=0.01,            # learning rate
    learning_rate='constant',
    tol=None,             # Disable convergence early stopping
    random_state=42,
    class_weight=None     # We'll handle sample weights manually
)

# Track performance during training
for epoch in range(1, 501):
    # Create sample weights based on class weights
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
    # Partial fit with sample weights
    sgd_curve.partial_fit(X_train, y_train, classes=np.unique(y), sample_weight=sample_weights)
    
    # Only record every 10 epochs to keep the plot manageable
    if epoch % 10 == 0:
        # Training error
        y_pred_train = sgd_curve.predict(X_train)
        train_error = 1 - accuracy_score(y_train, y_pred_train)
        
        # Validation error
        y_pred_test = sgd_curve.predict(X_test)
        validation_error = 1 - accuracy_score(y_test, y_pred_test)
        
        train_errors.append((epoch, train_error))
        validation_errors.append((epoch, validation_error))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train error = {train_error:.4f}, Validation error = {validation_error:.4f}")

# Plot learning curve
train_epochs, train_err = zip(*train_errors)
val_epochs, val_err = zip(*validation_errors)

plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_err, label='Training Error')
plt.plot(val_epochs, val_err, label='Validation Error')
plt.title('SGD Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig('sgd_learning_curve.png')
plt.close() 