from collections import Counter
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

models = {}
for model_name in ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Support Vector Machine', 'Gaussian Naive Bayes']:
    models[model_name] = load(f'../models/{model_name}.joblib')

feature_names = pd.read_csv('../data/Training.csv').drop(["Unnamed: 133", "prognosis"], axis=1).columns
all_symptoms = pd.read_csv('../data/Training.csv').columns[:-2].tolist()

label_encoder = load('../models/label_encoder.joblib')

def predict_symptoms(symptoms_vector):
    scaler = load('../models/scaler.joblib')
    symptoms_vector_scaled = scaler.transform(symptoms_vector.reshape(1, -1))
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(symptoms_vector_scaled)
        predictions[model_name] = label_encoder.inverse_transform(prediction)
    return predictions

def encode_symptoms(symptoms):
    encoded_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return encoded_vector

def predict_symptoms_with_probabilities_and_models(symptoms_vector):
    predictions = predict_symptoms(symptoms_vector)
    prediction_counts = Counter()
    for model_name, prediction in predictions.items():
        prediction_counts[prediction[0]] += 1
    total_predictions = sum(prediction_counts.values())
    probabilities = {prediction: count / total_predictions * 100 for prediction, count in prediction_counts.items()}
    return probabilities, predictions


def test():
    selected_symptoms = ['knee_pain', 'history_of_alcohol_consumption', 'dehydration', 'vomiting', 'movement_stiffness', 'diarrhoea']
    print("\nSelected symptoms:", selected_symptoms)
    encoded_vector = np.array(encode_symptoms(selected_symptoms))
    print("Encoded vector:", encoded_vector)
    probabilities, predictions = predict_symptoms_with_probabilities_and_models(encoded_vector)
    print("Predictions by each model:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction[0]}")
    print("Percentage chances of each unique prediction:")
    returnings = []
    for prediction, percentage in probabilities.items():
        returnings.append(f"Prediction: {prediction}, Percentage: {percentage:.2f}%")
        print(f"Prediction: {prediction}, Percentage: {percentage:.2f}%")
test()