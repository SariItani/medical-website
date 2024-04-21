from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

# Read the data
train = pd.read_csv('../data/Training.csv')
test = pd.read_csv('../data/Testing.csv')

# Drop the unwanted column
train = train.drop(["Unnamed: 133"], axis=1)

# Encode the target labels
label_encoder = LabelEncoder()
P_train = label_encoder.fit_transform(train["prognosis"])
P_test = label_encoder.transform(test["prognosis"])

# Separate features and target variables
X_train = train.drop(["prognosis"], axis=1)
X_test = test.drop(["prognosis"], axis=1)

# Split the data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(X_train, P_train, test_size=0.45, random_state=42)

# Scale the features
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
xval_scaled = scaler.transform(xval)
X_test_scaled = scaler.transform(X_test)

# Initialize models
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression()
svm = SVC()
gb = GaussianNB()

# Fit models
models = {'Random Forest': rf, 'Decision Tree': dt, 'Logistic Regression': lr, 'Support Vector Machine': svm, 'Gaussian Naive Bayes': gb}
for name, model in models.items():
    model.fit(xtrain_scaled, ytrain)
    tr_pred = model.predict(xtrain_scaled)
    ts_pred = model.predict(xval_scaled)
    tt_pred = model.predict(X_test_scaled)
    
    dump(model, f'../models/{name}.joblib')
    
    print("================= FOR", name, "=================")
    print("Training accuracy:", accuracy_score(ytrain, tr_pred))
    print("Validation accuracy:", accuracy_score(yval, ts_pred))
    print("Testing accuracy:", accuracy_score(P_test, tt_pred))
    print("Training precision:", precision_score(ytrain, tr_pred, average='weighted'))
    print("Validation precision:", precision_score(yval, ts_pred, average='weighted'))
    print("Testing precision:", precision_score(P_test, tt_pred, average='weighted'))
    print("Training recall:", recall_score(ytrain, tr_pred, average='weighted'))
    print("Validation recall:", recall_score(yval, ts_pred, average='weighted'))
    print("Testing recall:", recall_score(P_test, tt_pred, average='weighted'))
    print("Training F1-score:", f1_score(ytrain, tr_pred, average='weighted'))
    print("Validation F1-score:", f1_score(yval, ts_pred, average='weighted'))
    print("Testing F1-score:", f1_score(P_test, tt_pred, average='weighted'))
    print()

dump(label_encoder, '../models/label_encoder.joblib')
dump(scaler, '../models/scaler.joblib')

best_model = rf

# Making predictions on the test set
test_predictions = best_model.predict(X_test_scaled)
test["predicted"] = label_encoder.inverse_transform(test_predictions)

# Display the test data with predictions and available symptoms
test_with_symptoms = pd.concat([test, X_test], axis=1)  # Concatenate test data with available symptoms

# Filter symptoms where value is equal to 1
symptoms = test_with_symptoms.columns[2:][test_with_symptoms.iloc[:, 2:].eq(1).any()]

print("Shape of test_with_symptoms.iloc[:, 2:]:", test_with_symptoms.iloc[:, 2:].shape)
print("Shape of boolean index:", test_with_symptoms.iloc[:, 2:].eq(1).any().shape)

print(test_with_symptoms[["prognosis", "predicted"] + list(symptoms)])



# data visualization
train["prognosis"].value_counts().plot(kind="bar")
plt.xlabel("Prognosis")
plt.ylabel("Count")
plt.title("Target Label Distribution")
plt.savefig('../results/Target Label Distribution.png')

conf_mat = confusion_matrix(test["prognosis"], test["predicted"])
sns.heatmap(conf_mat, annot=True, cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('../results/Confusion Matrix.png')

importances = best_model.feature_importances_
feature_names = X_train.columns
plt.figure()
plt.bar(feature_names, importances)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.savefig('../results/Feature Importance.png')

for name, model in models.items():
    train_sizes, train_scores, val_scores = learning_curve(model, xtrain_scaled, ytrain, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title(f"Learning Curve of {name}")
    plt.legend()
    plt.savefig(f'../results/Learning Curve_{name}.png')

metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": []
}
for name, model in models.items():
    model.fit(xtrain_scaled, ytrain)
    tr_pred = model.predict(xtrain_scaled)
    ts_pred = model.predict(xval_scaled)
    tt_pred = model.predict(X_test_scaled)
    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(P_test, tt_pred))
    metrics["Precision"].append(precision_score(P_test, tt_pred, average="weighted"))
    metrics["Recall"].append(recall_score(P_test, tt_pred, average="weighted"))
    metrics["F1-score"].append(f1_score(P_test, tt_pred, average="weighted"))
metrics_df = pd.DataFrame(metrics)
metrics_df.plot(x="Model", y=["Accuracy", "Precision", "Recall", "F1-score"], kind="bar")
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend(loc="lower right")
plt.xticks(rotation=45)
plt.savefig('../results/Model Comparison.png')
