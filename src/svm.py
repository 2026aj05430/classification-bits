import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Loading the pre-split training and testing datasets
train_df = pd.read_csv('../data/train_data.csv', sep=';', encoding='utf-8')
test_df = pd.read_csv('../data/test_data.csv', sep=';', encoding='utf-8')

# Cleaning up any trailing/leading whitespaces from column headers
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Selecting the last column as the target
target_column = train_df.columns[-1]
print(f"Target column dynamically set to: '{target_column}'")

# Splitting features (X) and target (y) for both train and test sets
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Applying Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Support Vector Machine Classifier
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

# Making the predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled) # Probabilities for AUC score

# Calculating evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

# Printing evaluation metrics
print("\n==========================================")
print(" SUPPORT VECTOR MACHINE (SVM) METRICS     ")
print("==========================================")
print(f"1. Accuracy:  {accuracy:.4f}")
print(f"2. AUC Score: {auc_score:.4f}")
print(f"3. Precision: {precision:.4f}")
print(f"4. Recall:    {recall:.4f}")
print(f"5. F1 Score:  {f1:.4f}")
print(f"6. MCC Score: {mcc:.4f}")
print("==========================================")

# Saving the trained SVM model and scaler in .pkl format 
joblib.dump(model, '../model/svm_model.pkl')
joblib.dump(scaler, '../model/svm_scaler.pkl')
print("\nSVM model saved as 'svm_model.pkl' and scaler saved as 'svm_scaler.pkl'")