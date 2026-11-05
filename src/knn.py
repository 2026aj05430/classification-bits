import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Cleanin up any trailing/leading whitespaces from column headers
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

# Doing Feature Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the KNN Classifier (default n_neighbors=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled) # Probabilities for AUC score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

# Printing evaluation metrics
print("\n==========================================")
print("     K-NEAREST NEIGHBORS (KNN) METRICS     ")
print("==========================================")
print(f"1. Accuracy:  {accuracy:.4f}")
print(f"2. AUC Score: {auc_score:.4f}")
print(f"3. Precision: {precision:.4f}")
print(f"4. Recall:    {recall:.4f}")
print(f"5. F1 Score:  {f1:.4f}")
print(f"6. MCC Score: {mcc:.4f}")
print("==========================================")

# Saving the trained KNN model and scaler in .pkl format
joblib.dump(model, '../model/knn_model.pkl')
joblib.dump(scaler, '../model/knn_scaler.pkl')
print("\nKNN model saved as 'knn_model.pkl' and scaler saved as 'knn_scaler.pkl'")