import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

from learntools.core import binder
binder.bind(globals())
from learntools.ml_explainability.ex4 import *

data = pd.read_csv('../input/hospital-readmissions/train.csv')
y = data.readmitted
base_features = [c for c in data.columns if c != "readmitted"]
X = data[base_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
print("Setup Complete")


import shap

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(val_X.iloc[0:100])

shap.summary_plot(shap_values[1], val_X.iloc[0:100])
step_1.check()

step_2.solution()

shap.summary_plot(shap_values[1], val_X.iloc[0:100])
step_3.check()

shap.dependence_plot('num_medications', shap_values[1], val_X.iloc[0:100])
step_4.check()

shap.dependence_plot('time_in_hospital', shap_values[1], val_X.iloc[0:100])
step_5.check()
