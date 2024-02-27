global Models

Models = {
    "SVC": {
        "Role": "Classification"
    },
    "SVR": {
        "Role": "Regression"
    }
}

global Tasks 

Tasks = {""} | {i["Role"] for i in Models.values()}

global KWargs

KWargs = {'decision_function_shape': 'ovr', 
          'break_ties': False, 
          'kernel': 'rbf', 
          'degree': 3, 
          'gamma': 'scale', 
          'coef0': 0.0, 
          'tol': 0.001, 
          'C': 1.0, 
          'nu': 0.0, 
          'epsilon': 0.0, 
          'shrinking': True, 
          'probability': False, 
          'cache_size': 200, 
          'class_weight': None, 
          'verbose': False, 
          'max_iter': -1, 
          'random_state': None}
# def getModelsNames() -> list[str]:
#     return ["SVM"]