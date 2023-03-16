import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np


def get_hyperparameter_search_results(model, df_train, df_labels, search_type,
                                      param_grid, param_distributions, n_iter, cv):
    """
    Get hyperparameter search results
    @params
        model:        ML Model
        df_train      Input features (pre-processed)
        df_labels:    Labels
        search_type:  "grid", "randomized"
        param_grid:   List of dictionaries with parameters to be searched over
    @returns
        search_results: with following fields
            ["search", "bestparams", "bestestimator", "cvresults"]
    """
    search_results = {}

    ## train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    if search_type == "grid":
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv,
                              scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    elif search_type == "randomized":
        search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                                    n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error', random_state=42,
                                    return_train_score=True, n_jobs=-1)
    else:
        print("Invalid search type")
        return

    search_results["search"] = search
    search.fit(df_train, df_labels)

    if hasattr(search, 'bestparams'):
        search_results["bestparams"] = search.bestparams
        search_results["bestestimator"] = search.bestestimator
        search_results["cvresults"] = search.cvresults
    else:
        print("No hyperparameters found that improve the model's performance")

    return search_results

rf_param_grid = [
    {'n_estimators': [20, 70, 120, 170, 220, 320, 370, 420, 470, 570],
     'min_samples_split': [50, 100, 150, 200, 300, 500, 700, 1000, 2000, 3000], 
     'max_depth': [4, 8, 12, 16, 20, 24, 32, 36, 40,None],
     'warm_start': [True],
     'min_samples_leaf':[1, 5, 20, 30, 70, 100, 150, 200, 300, 500]} 
  ]

subset_1 = [
    {'n_estimators': [20, 40, 60, 80], 
     'min_samples_split': [200, 500, 700, 900], 
     'max_depth': [4,8,12,16], 
     'warm_start': [True], 
     'min_samples_leaf':[20, 50, 70, 90]}
   
  ]

subset_2 = [
    {'n_estimators': [80, 100, 120, 140], 
     'min_samples_split': [900, 1000, 1200, 1500], 
     'max_depth': [16,20,24,32], 
     'warm_start': [True], 
     'min_samples_leaf':[90, 100, 120, 150]}
  ]

subset_3 = [
    {'n_estimators': [140, 160, 180, 200], 
     'min_samples_split': [1500, 2000, 3000, 4000], 
     'max_depth': [32, 36, 40,None], 
     'warm_start': [True], 
     'min_samples_leaf':[150, 200, 300, 400],
     "random_state": [42]}
  ]


svm_param_grid = [
    {"C": [0.0001, 0.001, 0.01, .1, 1, 10, 100, 1000, 10000], 
     "kernel":["linear", "poly", "rbf", "sigmoid"],
     "degree":[1,2,3,4],
     "random_state":[42]} 
]
