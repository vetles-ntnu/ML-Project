import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix

# Import data
data_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.dirname(data_dir) + "/data/data.csv"
cancer_data = pd.read_csv(data_dir)
cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.contains('^Unnamed')] # remove Unnamed column (NaN)

# Set random seed for reproducibility
np.random.seed(10)

# Repeat pipeline n = 20 times to estimate mean & SD of test results
n = 20
tt_split = 0.1  # train/test split
kernels = ['rbf', 'linear', 'sigmoid', 'poly'] # kernel functions

for kernel in kernels:

    # Hyperparameter space for all kernels
    param_space = {
        'model__C': [0.1, 1, 10, 25, 50, 75, 100, 150, 1000, 10000, 100000],
        'model__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        'model__kernel': [kernel]
    }

    # Add hyperparameters
    if kernel == 'linear':
        del param_space['model__gamma']
    elif kernel == 'poly':
        param_space['model__degree'] = np.linspace(1,4,4,dtype=int)

    # Lists for estimating means and SDs classifications metrics
    accuracy_train_list = []
    accuracy_test_list = []
    recall_list = []
    precision_list = []
    f_fone_list = []

    # Best model parameter list
    best_params = []

    for i in range(n):
        # Split training/test data
        X = cancer_data.drop(['id', 'diagnosis'], axis=1)
        y = cancer_data['diagnosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_split, stratify=y)    # stratified split (classification)

        # Pipeline
        p_line = Pipeline(
            [
            ('scale', StandardScaler()),
            ('model',SVC())
            ]
        )

        # Grid search of parameter space (applied on the pipeline) using a 5-fold stratified CV split 
        model = GridSearchCV(p_line, param_space, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
        model.fit(X_train,y_train)

        # Add best model parameters and accuracy score
        best_params.append(model.best_params_)
        accuracy_train_list.append(model.best_score_)

        # Predict training data
        y_predict = model.predict(X_test)
        
        # Add test classification metrics
        accuracy_test_list.append(accuracy_score(y_test, y_predict))
        recall_list.append(recall_score(y_test, y_predict, pos_label='M'))
        precision_list.append(precision_score(y_test, y_predict, pos_label='M'))
        f_fone_list.append(f1_score(y_test, y_predict, pos_label='M'))
    
    # Print estimates of classification metrics
    print("Optimal model parameters (%s): " % kernel)
    print(*best_params, sep='\n')
    print("The model obtained a best training accuracy of " + str(round(np.mean(accuracy_train_list),4)) + " +/- " + str(round(np.std(accuracy_train_list),4)))
    print("The model obtained a best test accuracy of " + str(round(np.mean(accuracy_test_list),4)) + " +/- " + str(round(np.std(accuracy_test_list),4)))
    print("The model obtained a best test recall of " + str(round(np.mean(recall_list),4)) + " +/- " + str(round(np.std(recall_list),4)))
    print("The model obtained a best test precision of " + str(round(np.mean(precision_list),4)) + " +/- " + str(round(np.std(precision_list),4)))
    print("The model obtained a best test F1 score of " + str(round(np.mean(f_fone_list),4)) + " +/- " + str(round(np.std(f_fone_list),4)))

