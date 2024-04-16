import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../data/dataset.csv')

X = data['symptoms']
y = data['disease']

print(data.head())
print(data.describe())
print(data.dtypes)
print(data.isnull().sum())

disease_counts = y.value_counts()
plt.figure(figsize=(10, 6))
disease_counts.plot(kind='bar')
plt.title('Count of Unique Diseases')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.savefig('../results/Count of Unique Diseases.png')

symptom_counts = X.value_counts()
plt.figure(figsize=(12, 6))
symptom_counts[:20].plot(kind='bar')
plt.title('Count of Unique Symptoms (Top 20)')
plt.xlabel('Symptom')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('../results/Count of Unique Symptoms (Top 20).png')

encoder = OneHotEncoder()
X = encoder.fit_transform(X.to_frame()).toarray()
y = encoder.fit_transform(y.to_frame()).toarray()

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

print(pd.DataFrame(X).head())
print(pd.DataFrame(X).describe())
print(pd.DataFrame(X).dtypes)

print(pd.DataFrame(y).head())
print(pd.DataFrame(y).describe())
print(pd.DataFrame(y).dtypes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluate accuracy, precision, recall, and f1-score for each model
models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes']
y_preds = [y_pred_dt, y_pred_rf, y_pred_lr, y_pred_svm, y_pred_nb]

for model, y_pred in zip(models, y_preds):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"--- {model} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()