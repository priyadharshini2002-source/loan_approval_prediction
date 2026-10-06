import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("loan_approval_dataset.csv")

print(data.head())
print(data.info())
print(data.describe().T)

data.columns = data.columns.str.strip()
print(data.columns.tolist())

plt.figure(figsize=(6,4))

sns.countplot(data=data, x="loan_status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(6,4))

sns.countplot(data=data, x="education")

plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(7,4))

sns.countplot(
    data=data,
    x="no_of_dependents"
)

plt.title("Number of Dependents")
plt.xlabel("Number of Dependents")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8,5))
sns.histplot(
    data["income_annum"],
    bins=30,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(
    data["loan_amount"],
    bins=30,
    kde=True
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=data,
    x="loan_term"
)

plt.title("Loan Term Distribution")
plt.xlabel("Loan Term")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    data["cibil_score"],
    bins=30,
    kde=True
)

plt.title("CIBIL Score Distribution")
plt.xlabel("CIBIL Score")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=data,
    x="loan_status",
    y="cibil_score"
)

plt.title("CIBIL Score vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("CIBIL Score")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=data,
    x="loan_status",
    y="income_annum"
)

plt.title("Annual Income vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")

plt.show()


plt.figure(figsize=(8,5))

sns.boxplot(
    data=data,
    x="loan_status",
    y="loan_term"
)

plt.title("Loan Term vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Term")

plt.show()


numeric_cols = data.select_dtypes(include="number").columns

plt.figure(figsize=(10, 7))

sns.heatmap(
    data[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

print(data.isnull().sum())

print(data.duplicated().sum())


# encoding

categorical_cols = data.select_dtypes(include="object").columns

data[categorical_cols] = data[categorical_cols].apply(
    lambda col: col.str.strip()
)

for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")
    print("-" * 50)

data["education"] = data["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

data["self_employed"] = data["self_employed"].map({
    "Yes": 1,
    "No": 0
})

data["loan_status"] = data["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

print(data.head())

X = data.drop(["loan_status", "loan_id"], axis=1)

y = data["loan_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# logistic regression


log = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log.fit(X_train_scaled, y_train)
log_pred = log.predict(X_test_scaled)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("-"*100)
#classification_report
print(classification_report(y_test, log_pred))
print("-"*100)
# confusion matrix
cm_log = confusion_matrix(y_test, log_pred)
plt.figure(figsize=(8,5))
print("Logistic Regression Confusion Matrix : ", )
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
print("-"*100)

# Decision Tree

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train,y_train)
DT_pred = DT.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, DT_pred))
print("-"*100)
#classification_report
print(classification_report(y_test, DT_pred))
print("-"*100)
# confusion matrix
cm_dt = confusion_matrix(y_test, DT_pred)
plt.figure(figsize=(8,5))
print("Decision Tree Confusion Matrix : ", )
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()
print("-"*100)

# SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
print(y_train.value_counts())
print(y_train_smote.value_counts())

# logistic regression + SMOTE

scaler_smote = StandardScaler()
X_train_smote_scaled = scaler_smote.fit_transform(X_train_smote)
X_test_smote_scaled = scaler_smote.transform(X_test)

log_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_smote.fit(
    X_train_smote_scaled,
    y_train_smote
)

log_smote_pred = log_smote.predict(
    X_test_smote_scaled
)

# Accuracy
print("Logistic Regression Smote Accuracy:", accuracy_score(y_test, log_smote_pred))
print("-"*100)
#classification_report
print(classification_report(y_test, log_smote_pred))
print("-"*100)
# confusion matrix
cm_logs = confusion_matrix(y_test, log_smote_pred)
plt.figure(figsize=(8,5))
print("Logistic Regression Smote Confusion Matrix : ", )
sns.heatmap(cm_logs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Smote Confusion Matrix')
plt.show()
print("-"*100)

DT_smote = DecisionTreeClassifier(
    random_state=42
)

DT_smote.fit(
    X_train_smote,
    y_train_smote
)

DT_smote_pred = DT_smote.predict(
    X_test
)

# Accuracy
print("Decision Tree smote Accuracy:", accuracy_score(y_test, DT_smote_pred))
print("-"*100)
#classification_report
print(classification_report(y_test, DT_smote_pred))
print("-"*100)
# confusion matrix
cm_dts = confusion_matrix(y_test, DT_smote_pred)
plt.figure(figsize=(8,5))
print("Decision Tree smote Confusion Matrix : ", )
sns.heatmap(cm_dts, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree smote Confusion Matrix')
plt.show()
print("-"*100)

from sklearn.metrics import (precision_score,recall_score,f1_score)

results = []

models_predictions = {
    "Logistic Regression": log_pred,
    "Decision Tree": DT_pred,
    "Logistic Regression + SMOTE": log_smote_pred,
    "Decision Tree + SMOTE": DT_smote_pred
}

for model_name, predictions in models_predictions.items():

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="macro"
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="macro"
        ),
        "F1-Score": f1_score(
            y_test,
            predictions,
            average="macro"
        )
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

results_df.round(4)

plt.figure(figsize=(10, 5))

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.ylim(0.7, 1.0)
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

import joblib

joblib.dump(DT, "loan_approval_model.pkl")

joblib.dump(X.columns.tolist(), "loan_features.pkl")

print("Model saved successfully.")
print("Features saved successfully.")