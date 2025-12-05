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
# Additional analysis functions
def detailed_eda(df):
    """Detailed Exploratory Data Analysis"""
    print("=== Detailed EDA ===")
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df['HeartDiseaseorAttack'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Heart Disease Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='HeartDiseaseorAttack', data=df)
    plt.title('Heart Disease Count')
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Distribution of key features
    key_features = ['HighBP', 'HighChol', 'BMI', 'Age', 'GenHlth']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(key_features):
        if i < len(axes):
            df.groupby('HeartDiseaseorAttack')[feature].mean().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Average {feature} by Heart Disease')
            axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()

# Run detailed EDA
df = pd.read_csv("D:/EDAI5/Model/heart_disease_health_indicators_BRFSS2015.csv")
detailed_eda(df)

# Feature importance using Random Forest (for comparison)
from sklearn.ensemble import RandomForestClassifier

X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()