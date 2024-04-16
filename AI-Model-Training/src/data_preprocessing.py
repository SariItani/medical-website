import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../data/dataset.csv') 

print(data.head())

print(data.describe())
