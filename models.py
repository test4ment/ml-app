import numpy as np
from sklearn.svm import SVC, SVR

global Models

Models = {
    "SVC": {
        "Role": "Classification",
        "kwargs": "SVC",
        "ModelClass": SVC,
        "TrainedModel": None
    },
    "SVR": {
        "Role": "Regression",
        "kwargs": "SVC",
        "ModelClass": SVR,
        "TrainedModel": None
    }
}

global Tasks

Tasks = {""} | {i["Role"] for i in Models.values()}

global KWargs

KWargs = {
    "SVC": { # hint
        'decision_function_shape': {"type": "list", "items": ["ovr", "ovo"], "parseCallable": str},
        'break_ties': {"type": "list", "items": [False, True], "parseCallable": lambda arg: to_bool(arg)},
        'kernel': {"type": "list", "items": ["rbf", "linear", "poly", "sigmoid"], "parseCallable": str},
        'degree': {"type": "int", "range": (0, np.inf, 1), "default": 3, "kwargs": {"state": "normal"}, "parseCallable": int},
        'gamma': {"type": "list", "items": ["scale", "auto"], "kwargs": {"state": "normal"}, "parseCallable": lambda i: float(i) if is_float(i) else str(i)},
        'coef0': {"type": "float", "range": (-np.inf, np.inf, 0.1), "default": 0.0, "kwargs": {"state": "normal"}, "parseCallable": float},
        'tol': {"type": "float", "range": (0, 1, 1e-5), "default": 1e-3, "kwargs": {"state": "normal"}, "parseCallable": float},
        'C': {"type": "float", "range": (0, np.inf, 0.1), "default": 1.0, "kwargs": {"state": "normal"}, "parseCallable": float},
        'shrinking': {"type": "list", "items": [True, False], "parseCallable": lambda arg: to_bool(arg)},
        'cache_size': {"type": "int", "range": (0, np.inf, 25), "default": 200, "parseCallable": int},
        'max_iter': {"type": "int", "range": (-1, np.inf, 100), "default": -1, "kwargs": {"state": "normal"}, "parseCallable": int},
    }
}

global KWHints

KWHints = {
    'decision_function_shape': "Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) \n\
as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape \n\
(n_samples, n_classes * (n_classes - 1) / 2). However, note that internally, one-vs-one (‘ovo’) is always \n\
used as a multi-class strategy to train models; an ovr matrix is only constructed from the ovo matrix. \n\
The parameter is ignored for binary classification.",
    'break_ties': "If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties \n\
according to the confidence values of decision_function; otherwise the first class among the tied classes is returned.\n\
Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.",
    'degree': "Degree of the polynomial kernel function (‘poly’). \n\
Must be non-negative. Ignored by all other kernels.",
    'kernel': "Specifies the kernel type to be used in the algorithm.",
    'gamma': "Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.",
    'coef0': "Independent term in kernel function. \n\
It is only significant in ‘poly’ and ‘sigmoid’.",
    'tol': "Tolerance for stopping criterion.",
    'C': "Regularization parameter. The strength of the regularization is inversely proportional to C.\n\
Must be strictly positive. The penalty is a squared l2 penalty.",
    'shrinking': "Whether to use the shrinking heuristic. ",
    'cache_size': "Specify the size of the kernel cache (in MB).",
    'max_iter': "Hard limit on iterations within solver, or -1 for no limit.",
}


def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def to_bool(element: any) -> bool:
    try:
        return {
            "true": True,
            "false": False,
            "True": True,
            "False": False
        }[element]
    except:
        return bool(element)