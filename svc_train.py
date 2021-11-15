#!/usr/bin/env python
# coding: utf-8

# In[3]:


from src.dataset_importer import TrainImporter
from src.utils import import_train_configuration

config = import_train_configuration('settings/training_settings_1.ini')
X, y = TrainImporter(config).make_binary()

# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

stages = [
    ('preprocessor', MinMaxScaler()),
    ('classifier', BaggingClassifier(SVC()))
]

n_estimators = 50
stratified_kfold = StratifiedKFold(n_splits=5)
pipe = Pipeline(stages)
parameter_grid = {
    'classifier__base_estimator__kernel': ['linear', 'rbf'],
    'classifier__base_estimator__probability': [True],
    'classifier__max_samples': [1. / n_estimators],
}
model = GridSearchCV(estimator=pipe,
                     param_grid=parameter_grid,
                     cv=stratified_kfold,
                     scoring='average_precision',
                     n_jobs=1,
                     verbose=5,
                     error_score='raise')

model.fit(X, y)

# In[ ]:
