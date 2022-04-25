import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import pipeline
from sklearn import impute, compose, metrics, preprocessing
import joblib

#   Import the data
df = pd.read_csv('C:/Users/andre/Documents/Strive_repository/Heart_disease_dataset_analysis/heart.csv')

print(df.head())
print(df.shape)
print(df.duplicated())
print(df.describe())


col_names = df.columns
for col in col_names:
    uv_values = df[col].nunique()
    print(f'{col}:{uv_values}')


df['age_class'] = pd.cut(df['age'], bins=[29,44,59,80], labels = [0, 1, 2])

print(df.head(3))

def data_enhancement(data):  
      
    gen_data = data
    
    for rest_ecg in data['restecg'].unique():
        seasonal_data =  gen_data[gen_data['restecg'] == rest_ecg]
        thalachh_std = seasonal_data['thalachh'].std()
        oldpeak_std = seasonal_data['oldpeak'].std()





