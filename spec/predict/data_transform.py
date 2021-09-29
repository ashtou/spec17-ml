import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from IPython.display import display  # Jupyter display

# by default drop_first to avoid the 'dummy variable trap'
DROP = True


class Debug(BaseEstimator, TransformerMixin):
    """This class is designed to be used as an intermediate step in `Pipeline`s.
    """

    def __init__(self, rows=5):
        """`rows`: number of rows of the transformed X to store for debugging purposes
        """
        self.rows = rows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X_head = X[: self.rows, :]
        return X


class ColumnReorder(FunctionTransformer):
    """
    this custom transformer class is specifically designed to be used after a
    `ColumnTransformer` in a `Pipeline`,
    and reorder the columns transformed by the `ColumnTransformer` back to the
    original ordering of the `X` columns
    """

    def __init__(self, initial_features, trans_features):
        """
        `sklearn.base.BaseEstimator`: "all estimators should specify all 
        the parameters that can be set at the class level in their __init__ 
        as explicit keyword arguments (no *args or **kwargs)".
        Therefore, we need the internal versions of the parameters too
        """
        super().__init__(
            func=self._col_reorder_func,
            validate=True,
            kw_args={"init_feats": initial_features, "trans_feats": trans_features,},
        )
        # `get_params` looks at the internal versions
        self.initial_features = initial_features
        self.trans_features = trans_features

    # private static method
    @staticmethod
    def _col_reorder_func(X, init_feats, trans_feats):
        res_feats = trans_feats.copy()
        for feat in init_feats:
            if feat not in res_feats:
                res_feats.append(feat)
        # now `res_features` contains feature names in the transformed version
        order_ind = [res_feats.index(x0) for x0 in init_feats]
        X[:] = X[:, order_ind]
        return X


class CustomPipeline(Pipeline):
    """A Pipeline that exposes `coef_` or `feature_importances_`

    Note: `Pipeline` has a property called `_final_estimator` 
    """

    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    def __init__(self, regressor, trans_y):
        trans_funcs = {
            "log": {"func": np.log, "inverse_func": np.exp},
            "sqrt": {"func": np.sqrt, "inverse_func": lambda a: np.power(a, 2)},
            "none": {"func": lambda a: a, "inverse_func": lambda a: a},
        }
        func = trans_funcs[trans_y]["func"]
        inverse_func = trans_funcs[trans_y]["inverse_func"]
        # if you don't use super(), you'll have to pass all arguments
        super().__init__(regressor=regressor, func=func, inverse_func=inverse_func)
        self.trans_y = trans_y

    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_

    @property
    def coef_(self):
        return self.regressor_.coef_

    # in case it has `alpha_` as in MultiTaskLassoCV
    @property
    def alpha_(self):
        return self.regressor_.alpha_


def add_unique_os_columns(df, NESTED_CATEGORICAL):
    # Unique OS-Names excluding NaNs
    # we use a dictionary, because we'd like to track the presence (1) or absence (0) of unique values (need to remove one: dummy variable trap)
    # Tip: add prefix to column in pandas: df['col'] = 'prefix' + df['col'].astype(str)
    unique_os_dic = {
        os: 1 for os in ("OS_" + df["os_name"].dropna().astype(str)).unique()
    }
    os_base = "OS_CentOS"
    for uos in unique_os_dic:
        # Tip: create a new column in DataFrame based on a condition on another columns, axis=1 uses for different rows
        if NESTED_CATEGORICAL == False:  # using weighted dummy variables
            # Tip: Use df.apply(func, axis=1) to send every single row to a function. Axis along which the function is applied: 0 or 'index': apply function to each column. 1 or 'columns': apply function to each row
            df[uos] = df.apply(
                lambda row: int(str(row["os_vid"]).replace(".", ""))
                if row["os_name"] == uos.replace("OS_", "")
                else 0,
                axis=1,
            )
        else:
            df[uos] = df.apply(
                lambda row: 1 if row["os_name"] == uos.replace("OS_", "") else 0, axis=1
            )
            df["IA_" + uos] = df.apply(
                lambda row: row[uos] * float(row["os_vid"]), axis="columns"
            )
    if DROP == True:
        ##### NOTE: if you want to drop the first dummy
        df.drop(os_base, "columns")
        unique_os_dic[os_base] = 0

        # no need to remove one interaction in either case,
        # because if the main effect is absent, we still need
        # interaction terms for all unique values
        print("Remove one OS dummy var: ", os_base)
    print("unique_OS_dic: ", unique_os_dic)
    return unique_os_dic


def add_unique_compiler_columns(df, NESTED_CATEGORICAL):
    comp_name_vid_ser = (
        df["compiler"]
        .str.split("Build")
        .str[0]
        .str.split("Compiler")
        .str[0]
        .str.replace(",", "")
        .str.replace(".", "")
        .str.replace(":", "")
        .str.replace(r"C/C\+\+/Fortran", "")
        .str.replace(":", "")
        .str.replace(r"C/C\+\+", "")
        .str.replace("Version", "")
        .str.strip()
    )
    # ['1901144 of Intel', ...]
    df["comp_name"] = comp_name_vid_ser.str.split("of", 1).str[1].str.strip()
    df["comp_vid"] = comp_name_vid_ser.str.split("of", 1).str[0].str.strip().str[:4]
    unique_comp_name_vid_list = comp_name_vid_ser.dropna().unique().tolist()

    # Tip: unique values in a list: convert it to 'set'
    unique_compiler_dic = {
        comp: 1
        for comp in list(
            set(
                [
                    "COMP_" + i.split("of", 1)[1].strip()
                    for i in unique_comp_name_vid_list
                ]
            )
        )
    }
    comp_base = "COMP_AOCC"
    # Tip: manual long to wide
    for ucomp in unique_compiler_dic:
        if NESTED_CATEGORICAL == False:  # using weighted dummy variables
            df[ucomp] = df.apply(
                lambda row: int(row["comp_vid"])
                if row["comp_name"] == ucomp.replace("COMP_", "")
                else 0,
                axis=1,
            )
        else:
            df[ucomp] = df.apply(
                lambda row: 1 if row["comp_name"] == ucomp.replace("COMP_", "") else 0,
                axis=1,
            )
            df["IA_" + ucomp] = df.apply(
                lambda row: row[ucomp] * float(row["comp_vid"]), axis="columns"
            )

    if DROP == True:
        ##### NOTE: if you want to drop the first dummy
        df.drop(comp_base, "columns")
        unique_compiler_dic[comp_base] = 0

        # no need to remove one interaction in either case,
        # because if the main effect is absent, we still need
        # interaction terms for all unique values
        print("Remove one Compiler dummy var: ", comp_base)
    print("unique_Compiler_dic: ", unique_compiler_dic)
    return unique_compiler_dic


def make_Xy_df(
    all_data_df,
    NESTED_CATEGORICAL,
    numerical_predictors,
    categorical_predictors,
    unique_oses,
    unique_compilers,
    benchmarks,
    test_size=0.2,
    shuffle=True,
    random_state=None,
):
    """
    Get a df, convert all features to numerics, return X_df, y_df, ty_df, Xy_df
    """
    #####
    # split into train and test
    train_df, test_df = train_test_split(
        all_data_df.copy(),
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    # transform predictors
    def transform_predictors(inp_df):
        num_predictors = (
            numerical_predictors.copy()
        )  # to be able to extend in different calls without touching the original
        unique_os_interacts = ["IA_" + o for o in unique_oses]
        unique_comp_interacts = ["IA_" + c for c in unique_compilers]
        if NESTED_CATEGORICAL == True:
            # Tip: extend a list with multiple lists
            num_predictors += (
                [o for o in unique_oses if unique_oses[o] == 1]
                + unique_os_interacts
                + [c for c in unique_compilers if unique_compilers[c] == 1]
                + unique_comp_interacts
            )
        else:
            num_predictors += unique_oses + unique_compilers

        num_df = inp_df[
            num_predictors
        ]  # in this technique, the VIDs are already added to the dummy variables
        cat_df = inp_df[categorical_predictors]

        ###################################
        # Change categorical to dummy,
        # concat them to numerical and build the final df of all features
        ###################################
        if not cat_df.empty:
            if DROP == True:
                # Tip: DataFrames, avoid the dummy variable trap by
                # dropping the first dummy variable
                dummy_df = pd.get_dummies(cat_df, drop_first=True)
            else:
                dummy_df = pd.get_dummies(cat_df, drop_first=False)

            # Tip: DataFrame: concat two DataFrames
            X_df = pd.concat([num_df, dummy_df], axis="columns")
        else:
            X_df = num_df

        y_df = inp_df[benchmarks]
        ty_df = inp_df[["t_" + b for b in benchmarks]]
        # Tip: DataFrame: add columns (Series) to a DataFrame
        Xy_df = pd.concat([X_df, y_df], axis=1)
        # display(Xy_df.head(2))
        return X_df, y_df, ty_df, Xy_df

    assert (
        train_df.shape[1] == all_data_df.shape[1]
    ), "columns of train_df and all_data_df differ"
    # need stars * to unpack tuples
    return (
        *transform_predictors(all_data_df),
        *transform_predictors(train_df),
        *transform_predictors(test_df),
    )

