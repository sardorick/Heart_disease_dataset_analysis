from multiprocessing import Pipe
import numpy as np
import pandas as pd
from scipy import rand
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

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
# (data.corr()['output'].sort_values().plot.barh())
(data.corr()['output'].abs().sort_values().plot.barh())
# plt.show()
# exng       -0.436757
# oldpeak    -0.430696
# caa        -0.391724
# thall      -0.344029
# sex        -0.280937
# age        -0.225439
# trtbps     -0.144931
# chol       -0.085239
# fbs        -0.028046
# restecg     0.137230
# slp         0.345877
# thalachh    0.421741
# cp          0.433798
# output      1.000000


# print(data.isnull().sum()) - no missing data
x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

vals = ['age', 'trtbps', 'chol', 'thalachh','oldpeak']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train)
print(y_train)
scaler_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])

tree_pipe = ColumnTransformer(['num', scaler_pipeline, vals], remainder='passthrough')

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

classifiers = {name: make_pipeline(tree_pipe, model) for name, model in classifiers.items()}


results = pd.DataFrame({'Model': [], "Accuracy Score": [], "Balanced Accuracy score": [], "Time": []})


# for model_name, model in classifiers.items():
#     start_time = time.time()

#     model.fit(x_train, y_train)

#     predics = model.predict(x_val)
#     total_time = time.time() - start_time


#     results = results.append({"Model": model_name,
#                             "Accuracy": accuracy_score(y_val, predics)*100,
#                             "Balanced Accuracy": balanced_accuracy_score(y_val, predics)*100,
#                             "Time": total_time}, ignore_index=True)

# results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)

# print(results_ord)

