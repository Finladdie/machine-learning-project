import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

random_state = 42
kFoldSplits = 10

metadata = pd.read_csv("features/ravdess_speech_metadata.csv").to_numpy()
features = np.load("features/featureMatrix30Coeffs.npy")

"""
index:            0        1          2          3           4      5...
One row contents: emotion, intensity, statement, repetition, actor, [features]
"""
yXMat = np.concatenate(((metadata[:,1:6]), features), axis=1)

"""
label:   1        2     3      4    5      6        7        8
emotion: neutral, calm, happy, sad, angry, fearful, disgust, surprised
"""
# UNCOMMENT THIS TO REMOVE "neutral"-LABELLED DATA POINTS FROM DATASET
# yXMat = yXMat[yXMat[:,0] != 1]

nOfActors = 24
actors = np.arange(1, nOfActors+1)

log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

log_parameters = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs'],
    'clf__max_iter': [1000, 5000, 10000],
    'clf__class_weight': [None, 'balanced']
}

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

svc_parameters = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto'],
    'clf__class_weight': [None, 'balanced']
}
"""
Leave-One-Actor-Out (LOAO) Cross Validation

For each of the 24 actors we set them aside and perform a
train/val set on the rest. For efficiency, we precompute
boolean masks to access the 23-actor subgroup very efficiently. 
"""
log_results = []
svc_results = []

for actor in actors:

    training = yXMat[yXMat[:,4] != actor]
    validation = yXMat[yXMat[:,4] == actor]

    # shuffle dataset
    training = shuffle(training, random_state=random_state)

    X_train = training[:,5:].astype(float)
    y_train = training[:,0].astype(int)

    X_val = validation[:,5:].astype(float)
    y_val = validation[:,0].astype(int)

    # Create an array of the actor number for each feature vector in order, so
    # GridSearchCV can correctly isolate actors from one another.
    groups = training[:,4]
    cv = GroupKFold(n_splits=kFoldSplits)
    
    # Do a grid search of the best hyperparameter combination, then train the model with those params
    log_grid = GridSearchCV(
        estimator=log_pipeline,
        param_grid=log_parameters,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    svc_grid = GridSearchCV(
        estimator=svc_pipeline,
        param_grid=svc_parameters,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Predict the remaining test set using the trained model
    log_grid.fit(X_train, y_train, groups=groups)
    y_pred_log = log_grid.predict(X_val)

    svc_grid.fit(X_train, y_train, groups=groups)
    y_pred_svc = svc_grid.predict(X_val)

    log_acc = accuracy_score(y_val, y_pred_log, normalize=True)
    log_cm = confusion_matrix(y_val, y_pred_log)

    log_results.append({
        "actor": actor,
        "accuracy": log_acc,
        "confusion_matrix": log_cm.tolist(),
        "best_params": log_grid.best_params_
    })

    svc_acc = accuracy_score(y_val, y_pred_svc, normalize=True)
    svc_cm = confusion_matrix(y_val, y_pred_svc)

    svc_results.append({
        "actor": actor,
        "accuracy": svc_acc,
        "confusion_matrix": svc_cm.tolist(),
        "best_params": svc_grid.best_params_
    })
    
    print(f"Outer fold {actor}\nlog-reg accuracy: {log_acc}\nSVC accuracy: {svc_acc}")

np.save("results/resultsSVC30Reduced", svc_results)
np.save("results/resultsLOG30Reduced", log_results)
