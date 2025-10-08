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

# Fixed seed for reproductibility
random_state = 42

# Number of folds for hyperparameter grid search
n_folds = 24

# Number of actors in the dataset. In CV, actor groups must not be split.
# nOfActors = 24
# actors = np.arange(1, nOfActors+1)

metadata = pd.read_csv("features/ravdess_speech_metadata.csv").to_numpy()
features = np.load("features/featureMatrix30Coeffs.npy")

valMetadata = pd.read_csv("features/ravdess_song_metadata.csv").to_numpy()
valFeatures = np.load("features/featuresSUNG.npy")

"""
index:            0        1          2          3           4      5...
One row contents: emotion, intensity, statement, repetition, actor, [features]
"""
XMat = np.concatenate(((metadata[:,1:6]), features), axis=1)
yMat = np.concatenate(((valMetadata[:,1:6]), valFeatures), axis=1)

"""
label:   1        2     3      4    5      6        7        8
emotion: neutral, calm, happy, sad, angry, fearful, disgust, surprised
"""
# UNCOMMENT THIS TO REMOVE "neutral"-LABELLED DATA POINTS FROM DATASET
XMat = XMat[XMat[:,0] != 1]
yMat = yMat[yMat[:,0] != 1]



log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])


log_parameters = {
    'clf__C': [0.01],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs'],
    'clf__max_iter': [1000],
    'clf__class_weight': ['balanced']
}


log_results = []


# shuffle dataset
training = shuffle(XMat, random_state=random_state)
validation = shuffle(yMat, random_state=random_state)

X_train = training[:,5:].astype(float)
y_train = training[:,0].astype(int)
X_val = validation[:,5:].astype(float)
y_val = validation[:,0].astype(int)


# Create an array of the actor number of each data point, so
# GridSearchCV can correctly isolate actors from one another during
# inner k-fold.
groups = training[:,4]
cv = GroupKFold(n_splits=n_folds)

# Do a grid search of the best hyperparameter combination, then train the model with those params
log_grid = GridSearchCV(
    estimator=log_pipeline,
    param_grid=log_parameters,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)

# Predict the remaining test set using the trained model
log_grid.fit(X_train, y_train, groups=groups)
y_pred_log = log_grid.predict(X_val)

log_acc = accuracy_score(y_val, y_pred_log, normalize=True)
log_cm = confusion_matrix(y_val, y_pred_log)

log_results.append({
    "training_score": log_grid.best_estimator_.score(X_train, y_train),
    "validation_score": log_acc,
    "confusion_matrix": log_cm.tolist(),
    "best_params": log_grid.best_params_,
    "cv_results": log_grid.cv_results_
})

np.save("results/resultsFINALreduced", log_results)
