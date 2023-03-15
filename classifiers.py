import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rand_forest_params = {"n_estimators":[100, 200, 300], "max_depth" : [3,4,5], "max_features": ["sqrt", None]}

rand_forest_grid = GridSearchCV(estimator = RandomForestClassifier(), 
                                param_grid = rand_forest_params, 
                                verbose = 3)


SVM_grid = GridSearchCV(estimator = SVM)
