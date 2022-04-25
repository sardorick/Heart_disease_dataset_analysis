from multiprocessing import Pipe
import numpy as np
import pandas as pd
from scipy import rand
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


data = pd.read_csv(r'data\heart.csv')

# print(data.isnull().sum()) - no missing data

x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

scaler_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])

tree_pipe = ColumnTransformer(['num', scaler_pipeline, [0, -1]], remainder='passthrough')

classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Ada Boost": AdaBoostClassifier(random_state=0),
    "Extra Trees": ExtraTreesClassifier(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0),
    "Catboost": CatBoostClassifier(random_state=0),
    "Logistic Regression": LogisticRegression(random_state=0)
}

