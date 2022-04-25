from ast import Or
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


data = pd.read_csv('C:/Users/andre/Documents/Strive_repository/Heart_disease_dataset_analysis/heart.csv')
# (data.corr()['output'].sort_values().plot.barh())
# (data.corr()['output'].abs().sort_values().plot.barh())
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


#   Data enhancement

def data_enhancement(data):
    gen_data = data.copy()

    for cp in data['cp'].unique():
        cp_data = gen_data[gen_data['cp'] == cp]
        trtbps_std = cp_data['trtbps'].std()
        age_std = cp_data['age'].std()
        chol_std = cp_data['chol'].std()
        thalachh_std = cp_data['thalachh'].std()

        for i in gen_data[gen_data['cp'] == cp].index:
            if np.random.randint(2) == 1:
                gen_data['trtbps'].values[i] += trtbps_std/10
            else:
                gen_data['trtbps'].values[i] -= trtbps_std/10
            if np.random.randint(2) == 1:
                gen_data['age'].values[i] += age_std/10
            else:
                gen_data['age'].values[i] -= age_std/10
            if np.random.randint(2) == 1:
                gen_data['chol'].values[i] += chol_std/10
            else:
                gen_data['chol'].values[i] -= chol_std/10
            if np.random.randint(2) == 1:
                gen_data['thalachh'].values[i] += thalachh_std/10
            else:
                gen_data['thalachh'].values[i] -= thalachh_std/10
    return gen_data

gen = data_enhancement(data)
x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

num_vals = ['age', 'trtbps', 'chol', 'thalachh','oldpeak']
cat_vals = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)


#  Enhanced data to 20%
enhanced_sample = gen.sample(gen.shape[0] // 5)
x_train = pd.concat([x_train, enhanced_sample.drop(['output'], axis=1 ) ])
y_train = pd.concat([y_train, enhanced_sample['output'] ])
# print(x_train)
# print(y_train)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))])

tree_pipe = ColumnTransformer([('num', num_pipeline, num_vals), ('cat', cat_pipeline, cat_vals)], remainder='passthrough')

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


for model_name, model in classifiers.items():
    start_time = time.time()

    model.fit(x_train, y_train)

    predics = model.predict(x_val)
    total_time = time.time() - start_time


    results = results.append({"Model": model_name,
                            "Accuracy Score": accuracy_score(y_val, predics)*100,
                            "Balanced Accuracy score": balanced_accuracy_score(y_val, predics)*100,
                            "Time": total_time}, ignore_index=True)

results_ord = results.sort_values(by=['Accuracy Score'], ascending=False, ignore_index=True)

print(results_ord)




'''TEST'''


