import numpy as np

global Models

Models = {
    "SVC": {
        "Role": "Classification",
        "kwargs": "SVC"
    },
    "SVR": {
        "Role": "Regression",
        "kwargs": "SVC"
    }
}

global Tasks

Tasks = {""} | {i["Role"] for i in Models.values()}

global KWargs

KWargs = {
    "SVC": { # hint
        'decision_function_shape': {"type": "list", "items": ["ovo", "ovr"]},
        'break_ties': {"type": "list", "items": [True, False]},
        'kernel': {"type": "list", "items": ["linear", "poly", "rbf", "sigmoid"]},
        'degree': {"type": "int", "range": (0, np.inf, 1), "default": 3},
        'gamma': {"type": "list", "items": ["scale", "auto"], "kwargs": {"state": "normal"}},
        'coef0': {"type": "float", "range": (-np.inf, np.inf, 0.1), "default": 0.0},
        'tol': {"type": "float", "range": (0, 1, 1e-5), "default": 1e-3},
        'C': {"type": "float", "range": (0, np.inf, 0.1), "default": 1.0},
        'shrinking': {"type": "list", "items": [True, False]},
        'cache_size': {"type": "int", "range": (0, np.inf, 25), "default": 200},
        'max_iter': {"type": "int", "range": (-1, np.inf, 1), "default": -1},
    }
}

global KWHints

KWHints = {
    'decision_function_shape': "Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, note that internally, one-vs-one (‘ovo’) is always used as a multi-class strategy to train models; an ovr matrix is only constructed from the ovo matrix. The parameter is ignored for binary classification.",
    'break_ties': "If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.",
    'degree': "Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.",
    'kernel': "Specifies the kernel type to be used in the algorithm.",
    'gamma': "Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.",
    'coef0': "Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.",
    'tol': "Tolerance for stopping criterion.",
    'C': "Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.",
    'shrinking': "Whether to use the shrinking heuristic. ",
    'cache_size': "Specify the size of the kernel cache (in MB).",
    'max_iter': "Hard limit on iterations within solver, or -1 for no limit.",
}
# def getModelsNames() -> list[str]:
#     return ["SVM"]
