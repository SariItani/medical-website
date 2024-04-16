import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/dataset.csv')
X = data['symptoms']
y = data['disease']

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