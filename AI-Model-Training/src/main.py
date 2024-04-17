import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

# Read the data
train = pd.read_csv('../data/Training.csv')
test = pd.read_csv('../data/Testing.csv')

# Drop the unwanted column
train = train.drop(["Unnamed: 133"], axis=1)

labels = ["blister","red_sore_around_nose","yellow_crust_ooze","prognosis"]

# Encode the target labels
label_encoder = LabelEncoder()
P_train = label_encoder.fit_transform(train["blister"])
P_test = label_encoder.transform(test["blister"])

# Separate features and target variables
X_train = train.drop(["blister"], axis=1)
X_test = test.drop(["blister"], axis=1)

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
dt = DecisionTreeClassifier()
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

# Predict using the best model (you can choose based on the evaluation metrics)
best_model = rf  # Change this to the best model based on your evaluation
test_predictions = best_model.predict(X_test_scaled)
test["predicted"] = label_encoder.inverse_transform(test_predictions)

# Display the test data with predictions
print(test[["prognosis", "predicted"]])

# dump('../models/svm.pkl')
