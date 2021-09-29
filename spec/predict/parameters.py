# TODO: put all settings of all notebooks here
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn import neural_network
from sklearn.model_selection import KFold
import eli5

from . import data_transform

##################################################
# Feature Selection

# cross-validation
RS = 42
n_folds = 5
k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=RS)

# trans_y
TRANS_Y = "log"
# en elastic net hyperparameters
EN_ALPHA = 1e-3
EN_TOL = 1e-4
EN_MAX_ITER = 10000
# dt decision tree hyperparameters
DT_MAX_DEPTH = 15
# fs random forest hyperparameters
RF_N_ESTIMATORS = 30
RF_MAX_FEATURES = 0.5
# fs mlp hyperparameters
MLP_HIDDEN_LAYER_SIZES = (20,)
MLP_MAX_ITER = 10000
MLP_ACTIVATION = "tanh"
MLP_TOL = 1e-3
MLP_ADAM_SGD_EARLY_STOPPING = True
# eli5 fs permutation importance hyperparameters
PI_N_REPS = 5
PI_SCORING = "neg_mean_absolute_error"
# sklearn permutation importance hyperparameters
PI_N_JOBS = -1
#####
# correlation fs
CLUSTER_TH = 0.35
#####
# rfecv parameters
RFECV_SCORING = "neg_mean_absolute_error"
RFECV_N_JOBS = -1
# gridcv parameters
GRIDCV_SCORING = "neg_mean_absolute_error"
GRIDCV_N_JOBS = -1


# fs elastic net
fs_EN = data_transform.CustomTransformedTargetRegressor(
    regressor=linear_model.MultiTaskElasticNet(
        alpha=EN_ALPHA, random_state=RS, tol=EN_TOL, max_iter=EN_MAX_ITER
    ),
    trans_y=TRANS_Y,
)

# fs decision tree
fs_DT = data_transform.CustomTransformedTargetRegressor(
    regressor=tree.DecisionTreeRegressor(max_depth=DT_MAX_DEPTH, random_state=RS,),
    trans_y=TRANS_Y,
)

# fs random forest
fs_RF = data_transform.CustomTransformedTargetRegressor(
    regressor=ensemble.RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_features=RF_MAX_FEATURES, random_state=RS,
    ),
    trans_y=TRANS_Y,
)

# fs mlp adam
fs_MLP_adam = data_transform.CustomTransformedTargetRegressor(
    regressor=neural_network.MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        solver="adam",
        activation=MLP_ACTIVATION,
        tol=MLP_TOL,
        early_stopping=MLP_ADAM_SGD_EARLY_STOPPING,
        random_state=RS,
    ),
    trans_y=TRANS_Y,
)

# fs mlp lbfgs
fs_MLP_lbfgs = data_transform.CustomTransformedTargetRegressor(
    regressor=neural_network.MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        solver="lbfgs",
        activation=MLP_ACTIVATION,
        tol=MLP_TOL,
        random_state=RS,
    ),
    trans_y=TRANS_Y,
)

#####
# eli5 PermutationImportance with fs_DT
fs_pi_dt = eli5.sklearn.PermutationImportance(
    estimator=fs_DT, scoring=PI_SCORING, n_iter=PI_N_REPS, random_state=RS, cv=k_fold
)

# eli5 PermutationImportance with fs_RF
fs_pi_rf = eli5.sklearn.PermutationImportance(
    estimator=fs_RF, scoring=PI_SCORING, n_iter=PI_N_REPS, random_state=RS, cv=k_fold
)

# eli5 PermutationImportance with fs_MLP_adam
fs_pi_mlp_adam = eli5.sklearn.PermutationImportance(
    estimator=fs_MLP_adam,
    scoring=PI_SCORING,
    n_iter=PI_N_REPS,
    random_state=RS,
    cv=k_fold,
)

# eli5 PermutationImportance with fs_MLP_lbfgs
fs_pi_mlp_lbfgs = eli5.sklearn.PermutationImportance(
    estimator=fs_MLP_lbfgs,
    scoring=PI_SCORING,
    n_iter=PI_N_REPS,
    random_state=RS,
    cv=k_fold,
)

# RFECV ests
RFECV_ESTS = [fs_EN, fs_DT, fs_RF, fs_pi_dt, fs_pi_rf, fs_pi_mlp_adam, fs_pi_mlp_lbfgs]

##################################################
# Grid Search CV
PARALLEL = False

param_grid = {}
# ESTS = [fs_EN, fs_DT, fs_RF, fs_pi_dt, fs_pi_rf, fs_pi_mlp_adam, fs_pi_mlp_lbfgs]
# EN param_grids
param_grid[0] = {
    "eval__regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "eval__regressor__l1_ratio": [0.25, 0.5, 0.75, 1],
    "eval__regressor__max_iter": [5000],
}

# DT param_grids
param_grid[1] = {
    "eval__regressor__criterion": ["mse", "friedman_mse", "mae"],
    "eval__regressor__max_depth": [5, 10, 15, 20, 25, None],
}

# RF param_grids
param_grid[2] = {
    # "eval__regressor__criterion":       ["mse", "mae"],
    "eval__regressor__n_estimators": [10, 30, 50, 100, 200],
    "eval__regressor__max_features": [0.5, 0.7, 0.9, None],
    "eval__regressor__max_samples": [0.8, 0.9, None],
}

# PI DT param_grids
param_grid[3] = param_grid[1]

# PI RF param_grids
param_grid[4] = param_grid[2]

# PI MLP param_grids
param_grid[5] = {
    "eval__regressor__hidden_layer_sizes": [
        (10,),
        (20,),
        (30,),
        (50,),
        # (100,),
        # (10,10), (30,30), (50,50), (100,100),
        # (1,1,1,), (10,10,10,), (20, 20, 20), (50, 50, 50)
    ],
    "eval__regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2],  # regularization
    "eval__regressor__tol": [5e-4, 1e-3],
    "eval__regressor__activation": ["tanh", "relu"],  # 'identity', 'logistic'],
    "eval__regressor__early_stopping": [True],  # sgd or adam
    "eval__regressor__solver": ["adam"],
    # 'eval__regressor__learning_rate': ['constant', 'invscaling', 'adaptive'], #sgd
    # 'eval__regressor__learning_rate_init': [0.001, 0.01], # sgd or adam
    # "eval__regressor__epsilon": [1e-8, 1e-7] # adam
}

# PI MLP param_grids
param_grid[6] = {
    "eval__regressor__hidden_layer_sizes": [
        (10,),
        (20,),
        (30,),
        (50,),
        # (100,),
        # (10,10), (30,30), (50,50), (100,100),
        # (1,1,1,), (10,10,10,), (20, 20, 20), (50, 50, 50)
    ],
    "eval__regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "eval__regressor__tol": [5e-4, 1e-3],
    "eval__regressor__activation": ["tanh", "relu",],  # 'identity', 'logistic'],
    "eval__regressor__solver": ["lbfgs"],
}
