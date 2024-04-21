import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the pre-trained models
models = {}
for model_name in ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Support Vector Machine', 'Gaussian Naive Bayes']:
    models[model_name] = load(f'../models/{model_name}.joblib')

# Load the feature names
feature_names = pd.read_csv('../data/Training.csv').drop(["Unnamed: 133", "prognosis"], axis=1).columns

# Function to make predictions for selected symptoms
label_encoder = load('../models/label_encoder.joblib')

# Function to make predictions for selected symptoms
def predict_symptoms(symptoms):
    # Convert symptoms into feature vector
    symptoms_vector = np.zeros(len(feature_names))
    for symptom in symptoms:
        index = np.where(feature_names == symptom)[0]
        if len(index) > 0:
            symptoms_vector[index[0]] = 1
    
    # Scale the features
    scaler = load('../models/scaler.joblib')
    symptoms_vector_scaled = scaler.fit_transform(symptoms_vector.reshape(1, -1))
    
    # Make predictions using each model
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(symptoms_vector_scaled)
        predictions[model_name] = label_encoder.inverse_transform(prediction)  # Inverse transform the predicted labels
        
    return predictions


train = pd.read_csv('../data/Training.csv')
all_symptoms = train.columns[:-1]  # Exclude the last column which is the target variable 'prognosis'
print(all_symptoms)

# Select 4 random symptoms from all symptoms
for _ in range(100):
    print()
    selected_symptoms = random.sample(list(all_symptoms), 4)
    print("Selected symptoms:", selected_symptoms)

    # Generate predictions
    results = predict_symptoms(selected_symptoms)
    for model_name, prediction in results.items():
        print(f"Prediction using {model_name}: {prediction}")
