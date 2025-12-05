import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp

# Load the dataset
df = pd.read_csv("D:/EDAI5/Model/heart_disease_health_indicators_BRFSS2015.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nTarget variable distribution:")
print(df['HeartDiseaseorAttack'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

