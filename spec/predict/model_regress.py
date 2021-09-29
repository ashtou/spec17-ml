import numpy as np
import pandas as pd
from . import data_transform
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, cross_validate
import time
from sklearn import metrics
import pprint
from termcolor import colored


def single_mean_absolute_percentage_error(y_true, y_pred):
    """MAPE for single output"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_percentage_error(y_true_arr, y_pred_arr):
    """MAPE for multi-output"""
    # print(y_true_arr.shape, y_pred_arr.shape)
    mapes = np.empty([y_true_arr.shape[1]])
    y_pred_arr = pd.DataFrame(y_pred_arr, columns=y_true_arr.columns)
    for ind, target in enumerate(y_true_arr):
        y_true = y_true_arr[target].to_numpy()
        y_pred = y_pred_arr[target].to_numpy()
        mapes[ind] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mapes


class RegrModel:
    """
    This class provides methods and attributes for regression.
    It supports multi-output, target transform, and feature scale/transform.
    
    It can accept a model in a simple form or as:

    ('Name', Pipeline(steps=
        [(`'transform_X'`, transformer),
         ('scale', scaler), 
         (`'regress'`, data_transform.transform_regressor(
             Estimator(), 
             trans_y='log'))
        ])
    ), 

    `_model`: can be an sklearn model or a `Pipeline` as above
    """

    __slots__ = [
        "_name",
        "_model",
        "_trans_y",
        "_n_features",
        "_train_time",
        "_pred_times",
        "_mean_pred_time",
        "_std_pred_time",
        "_r2s",
        "_avg_r2",
        "_median_r2",
        "_std_r2",
        "_maes",
        "_avg_mae",
        "_median_mae",
        "_std_mae",
        "_mapes",
        "_avg_mape",
        "_median_mape",
        "_std_mape",
    ]

    def __init__(self, name="regr_model", model=None):
        self._name = name
        self._model = model
        self._n_features = 0
        if isinstance(self._model, Pipeline):  # isinstance is True for subclasses to
            self._trans_y = model.steps[-1][1].trans_y
        else:
            self._trans_y = model.trans_y
        self._r2s, self._avg_r2, self._median_r2, self._std_r2 = (None, 0, 0, 0)
        self._maes, self._avg_mae, self._median_mae, self._std_mae = (None, 0, 0, 0)
        self._mapes, self._avg_mape, self._median_mape, self._std_mape = (None, 0, 0, 0)
        self._train_time, self._mean_pred_time, self._std_pred_time = 0, 0, 0
        self._pred_times = []

    # @property turns method into a “getter” for an attribute with the same name
    # it'd also make it easier to change what we return without changing the interface
    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def trans_y(self):
        return self._trans_y

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, val):
        self._n_features = val

    @property
    def train_time(self):
        return self._train_time

    @train_time.setter
    def train_time(self, val):
        self._train_time = val

    @property
    def pred_times(self):
        return self._pred_times

    # assign an empty list in `predict()`
    @pred_times.setter
    def pred_times(self, val_list):
        self._pred_times = val_list

    @property
    def mean_pred_time(self):
        return self._mean_pred_time

    @mean_pred_time.setter
    def mean_pred_time(self, val):
        self._mean_pred_time = val

    @property
    def std_pred_time(self):
        return self._std_pred_time

    @std_pred_time.setter
    def std_pred_time(self, val):
        self._std_pred_time = val

    # r2
    @property
    def r2s(self):
        return self._r2s

    @r2s.setter
    def r2s(self, val):
        self._r2s = val

    @property
    def avg_r2(self):
        return self._avg_r2

    @avg_r2.setter
    def avg_r2(self, val):
        self._avg_r2 = val

    @property
    def median_r2(self):
        return self._median_r2

    @median_r2.setter
    def median_r2(self, val):
        self._median_r2 = val

    @property
    def std_r2(self):
        return self._std_r2

    @std_r2.setter
    def std_r2(self, val):
        self._std_r2 = val

    # mae
    @property
    def maes(self):
        return self._maes

    @maes.setter
    def maes(self, val):
        self._maes = val

    @property
    def avg_mae(self):
        return self._avg_mae

    @avg_mae.setter
    def avg_mae(self, val):
        self._avg_mae = val

    @property
    def median_mae(self):
        return self._median_mae

    @median_mae.setter
    def median_mae(self, val):
        self._median_mae = val

    @property
    def std_mae(self):
        return self._std_mae

    @std_mae.setter
    def std_mae(self, val):
        self._std_mae = val

    # mape
    @property
    def mapes(self):
        return self._mapes

    @mapes.setter
    def mapes(self, val):
        self._mapes = val

    @property
    def avg_mape(self):
        return self._avg_mape

    @avg_mape.setter
    def avg_mape(self, val):
        self._avg_mape = val

    @property
    def median_mape(self):
        return self._median_mape

    @median_mape.setter
    def median_mape(self, val):
        self._median_mape = val

    @property
    def std_mape(self):
        return self._std_mape

    @std_mape.setter
    def std_mape(self, val):
        self._std_mape = val

    # regression
    def fit(self, X_df, y_df, verbose=False):
        # we do a separate fit to get _n_features so that the timing remains accurate
        if isinstance(self.model, Pipeline):
            self.model.steps.insert(-1, ("_Debug", data_transform.Debug()))
            t0 = time.time()
            fitted = self.model.fit(X_df, y_df)
            self.train_time = round(time.time() - t0, 3)

            self.n_features = self.model.steps[-2][1].X_head.shape[1]
            self.model.steps.pop(-2)
        else:
            t0 = time.time()
            fitted = self.model.fit(X_df, y_df)
            self.train_time = round(time.time() - t0, 3)

            self.n_features = X_df.shape[1]

        if verbose:
            print(
                colored("NOTE:", "red"),
                "due to a current bug in VSCode,"
                " `func` and `inverse_func` might not get printed in the notebook",
            )
            print(colored("Fitted:\n", "magenta"), str(fitted))
            print(colored("Model Parameters:\n", "magenta"))
            pprint.pprint(self.model.get_params(), width=1)

        return fitted

    def predict(self, test_X_df, test_y_df, pred_iter=10, verbose=False):
        self.pred_times = []
        for _ in range(0, pred_iter):
            t1 = time.time()
            ypred = self.model.predict(test_X_df)
            self.pred_times.append(round(time.time() - t1, 6))
        self.mean_pred_time = round(np.mean(self.pred_times), 4)
        self.std_pred_time = round(np.std(self.pred_times), 4)
        # print(self.name, "pred and, train times: ",
        #     self.pred_time, self.train_time)
        self.r2s = metrics.r2_score(test_y_df, ypred, multioutput="raw_values")
        # take the average manually,
        # since multioutput can take other values than 'raw_values'
        self.avg_r2 = np.mean(self.r2s)
        self.median_r2 = np.median(self.r2s)
        self.std_r2 = np.std(self.r2s)
        self.maes = metrics.mean_absolute_error(
            test_y_df, ypred, multioutput="raw_values"
        )
        self.avg_mae = np.mean(self.maes)
        self.median_mae = np.median(self.maes)
        self.std_mae = np.std(self.maes)
        self.mapes = mean_absolute_percentage_error(test_y_df, ypred)
        self.avg_mape = np.mean(self.mapes)
        self.median_mape = np.median(self.mapes)
        self.std_mape = np.std(self.mapes)

        if verbose:
            print(self.name, ", model.score:", self.model.score(test_X_df, test_y_df))
            print(self.trans_y, "y-transform, raw scores: ", self.r2s)
            print("avg. r2 score ('uniform_average'): ", self.avg_r2)
            print("pred time iters:", self.pred_times)

        return ypred

    def get_metrics(self, verbose=False):
        res = self.__repr__()

        if verbose:
            res = (
                res
                + "\n   raw r2s: "
                + "\033[0m"
                + str(self.r2s)
                + "\n   raw maes: "
                + str(self.maes)
                + "\n   raw mapes: "
                + str(self.mapes)
            )

        return res + "\n"

    # print
    def __repr__(self):
        """Print object.
        """
        return (
            colored(self.name + "\n", "cyan")
            + (
                str(self.model.steps[-1][1])
                if isinstance(self.model, Pipeline)
                else str((self.model))
            )
            + colored("\navg r2: ", "red")
            + colored("%.4f" % self.avg_r2, "green")
            + colored(", avg mae: ", "red")
            + colored("%.4f" % self.avg_mae, "green")
            + colored(", avg mape: ", "red")
            + colored("%.2f" % self.avg_mape, "green")
            + ", median mape: %.2f" % self.median_mape
            + ", train time: %.4f" % self.train_time
            + ", mean_pred time: %.4f" % self.mean_pred_time
            + colored("\nn_features: ", "red")
            + str(self.n_features)
            + (
                "\nPipeline Steps: " + str(self.model.named_steps.keys())
                if isinstance(self.model, Pipeline)
                else ""
            )
            + "\n----------\n"
        )


def check_scores(predictor, train_X, train_y, test_X, test_y, cv=None, verbose=False):
    """`predictor`: can be a regressor or a `Pipeline` with a regressor at the last step

    `cv`: if `None`, cross-validation scores will not be computed
    """
    if cv is not None:
        cv_scoring = ["r2", "neg_mean_absolute_error"]
        cv_dict = cross_validate(
            predictor, train_X, train_y, scoring=cv_scoring, cv=cv, n_jobs=-1,
        )

    # fit on train
    # Pipeline
    if isinstance(predictor, Pipeline):
        predictor.steps.insert(-1, ("_Debug", data_transform.Debug()))
        # fit
        t_tr0 = time.time()
        fitted = predictor.fit(train_X, train_y)
        t_tr1 = time.time()
        n_features = predictor.steps[-2][1].X_head.shape[1]
        predictor.steps.pop(-2)
    else:
        # fit
        t_tr0 = time.time()
        fitted = predictor.fit(train_X, train_y)
        t_tr1 = time.time()
        # meta-estimator feature selector such as RFE or RFECV
        if hasattr(predictor, "n_features_"):
            n_features = predictor.n_features_
        # regressor
        else:
            n_features = train_X.shape[1]

    print("Fitting time:", round(t_tr1 - t_tr0, 4))
    print("Num of features:", n_features)
    if cv is not None:
        print(
            "Cross-validation MAE: {:.6f}".format(
                -np.mean(cv_dict["test_neg_mean_absolute_error"])
            )
        )
        print("Cross-validation R2: {:.6f}".format(np.mean(cv_dict["test_r2"])))
        if verbose:
            print("Cross-validation detailed results:")
            for res in cv_dict:
                print("\t" + res + ": ", cv_dict[res])

    # predict on test
    t_ts0 = time.time()
    y_pred = predictor.predict(test_X)
    t_ts1 = time.time()
    print("Prediction time:", round(t_ts1 - t_ts0, 4))

    print(
        "MAE on Test:: {:.6f}".format(
            metrics.mean_absolute_error(test_y, y_pred, multioutput="uniform_average")
        )
    )
    print(
        "R2 on Test: {:.6f}".format(
            metrics.r2_score(test_y, y_pred, multioutput="uniform_average")
        )
    )
    print("\n")
    return fitted

