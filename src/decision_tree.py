import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

#Loading the pre-split training and testing datasets
train_df = pd.read_csv('../data/train_data.csv', sep=';', encoding='utf-8')
test_df = pd.read_csv('../data/test_data.csv', sep=';', encoding='utf-8')

# Cleaning up any trailing/leading whitespaces from column headers just in case
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

#Selecting the last column as the target
target_column = train_df.columns[-1]
print(f"Target column dynamically set to: '{target_column}'")

# Splitting features (X) and target (y) for both train and test sets
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Training the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Making predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) # Probabilities for AUC score

#Calculating required evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

# Printing evaluation metrics
print("\n==========================================")
print("   DECISION TREE CLASSIFIER METRICS       ")
print("==========================================")
print(f"1. Accuracy:  {accuracy:.4f}")
print(f"2. AUC Score: {auc_score:.4f}")
print(f"3. Precision: {precision:.4f}")
print(f"4. Recall:    {recall:.4f}")
print(f"5. F1 Score:  {f1:.4f}")
print(f"6. MCC Score: {mcc:.4f}")
print("==========================================")

# 7. Saving the trained Decision Tree model in .pkl format
joblib.dump(model, '../model/decision_tree_model.pkl')
print("\nDecision Tree model successfully saved as 'decision_tree_model.pkl'")