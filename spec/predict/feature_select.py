import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
import scipy.stats, scipy.cluster
from collections import defaultdict
import matplotlib.pyplot as plt


class RemoveCorrFeatures(SelectorMixin, BaseEstimator):
    """This custom transformer class is designed to be used in a `Pipeline` 
    for feature selection by removing highly correlated features
    """

    def __init__(self, threshold=0):
        """`sklearn.base.BaseEstimator`: "all estimators should specify all 
        the parameters that can be set at the class level in their __init__ 
        as explicit keyword arguments (no *args or **kwargs)".
        Therefore, we need the internal versions of the parameters too.
        """
        self.threshold = threshold
        self.spearmanr_corr = None
        self.spearmanr_corr_linkage = None

    def fit(self, X, y=None):
        """Since `check_is_fitted` looks for attributes ending with "_",
        we define ` self.support_mask_` and `self.selected_indexes_` here.
        """
        self.support_ = [False for i in range(X.shape[1])]

        self.spearmanr_corr = scipy.stats.spearmanr(X).correlation
        # NOTE: we use absolute correlation values to cluster
        self.spearmanr_corr_linkage = scipy.cluster.hierarchy.ward(
            np.abs(self.spearmanr_corr)
        )

        cluster_ids = scipy.cluster.hierarchy.fcluster(
            self.spearmanr_corr_linkage, self.threshold, criterion="distance"
        )

        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        # select the first element in each cluster
        self.selected_indexes_ = [v[0] for v in cluster_id_to_feature_ids.values()]
        for ind, _ in enumerate(self.support_):
            if ind in self.selected_indexes_:
                self.support_[ind] = True
        return self

    def transform(self, X):
        """If you use fit_transform, then it will return the following from the base.py
        return self.fit(X, **fit_params).transform(X)
        In that case, X is not an ndarray, so we need to convert dfs to ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return X[:, self.support_]

    def _get_support_mask(self):
        """This is required so that we can call `get_support()` 
        to get a mask, or integer index, of the features selected
        
        `check_is_fitted` works like this:
        Checks if the estimator is fitted by verifying the **presence** of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        """
        check_is_fitted(self)

        return self.support_

    def dendrogram(self, X_df, title):
        """This method is only for the visualisation purpose.
        X_df: has to be a dataframe, since we need the column names for the plots.
        """
        self.fit(X_df)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        dendro = scipy.cluster.hierarchy.dendrogram(
            self.spearmanr_corr_linkage, labels=list(X_df), ax=ax1, leaf_rotation=90,
        )
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        ax2.imshow(self.spearmanr_corr[dendro["leaves"], :][:, dendro["leaves"]])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax2.set_yticklabels(dendro["ivl"])
        fig.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.show()
        plt.close()

        return


class MaskFeatureSelect(SelectorMixin, BaseEstimator):
    """This custom transformer class is designed to be used in a `Pipeline` 
    for feature selection using a support_mask, e.g. provided by `RFE` or `RFECV`
    """

    def __init__(self, mask):
        self.mask = mask

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return X[:, self.support_]

    def fit(self, X, y=None):
        """Since `check_is_fitted` looks for attributes ending with "_",
        we define ` self.support_mask_` and `self.selected_indexes_` here.
        """
        self.support_ = [elm for elm in self.mask]

        return self

    def _get_support_mask(self):
        """This is required so that we can call `get_support()` 
        to get a mask, or integer index, of the features selected
        
        `check_is_fitted` works like this:
        Checks if the estimator is fitted by verifying the **presence** of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        """
        check_is_fitted(self)

        return self.support_


def get_avg_corr_with_outputs(X_df, y_df, trans):  # df includes all outs in outputs
    # Tip: DataFrame: filter columns
    X_cols = X_df.columns
    y_cols = y_df.columns
    mean_cor_dic = {key: 0 for key in X_cols}
    tmp_df = pd.concat([X_df, y_df], axis=1)
    for out in y_df.columns:
        # trnasform the output column
        if trans == "log":
            tmp_df[out] = np.log(tmp_df[out])
        elif trans == "sqrt":
            tmp_df[out] = np.sqrt(tmp_df[out])
        elif trans == "none":
            tmp_df[out] = tmp_df[out]

        cor = tmp_df.corr()

        cor_ser = abs(cor[out]).round(2)  # a series with names of df columns as indexes
        # Tip: Series, filter with multiple conditions:
        # ser[~ser.index.isin(my_list)][ser > 1.5]
        relevant_feat_ser = cor_ser[~cor_ser.index.isin(y_cols)]
        # Tip: Series, iterate over items, get index and value
        for ind, val in relevant_feat_ser.iteritems():
            mean_cor_dic[ind] += val
    for key in mean_cor_dic:
        mean_cor_dic[key] = round(mean_cor_dic[key] / len(y_cols), 2)
    sorted_mean_cor = sorted(mean_cor_dic.items(), key=lambda kv: kv[1], reverse=True)
    return mean_cor_dic, sorted_mean_cor  # return all the features


def get_multicollinear_features(X_df, y_df, sig_level):
    multicoll_list = []
    tmp_df = pd.concat([X_df, y_df], axis=1)
    cor = tmp_df.corr()
    for inp in X_df.columns:
        cor_ser = abs(cor[inp]).round(3)
        tmp_outs = list(y_df)
        # replace inps one by one and compare them with all other inputs (not outputs)
        tmp_outs.append(inp)
        multicoll_ser = cor_ser[~cor_ser.index.isin(tmp_outs)][
            cor_ser >= sig_level
        ]  # any input that has a corr > sig_level with this inp
        for ind, val in multicoll_ser.iteritems():
            multicoll_list.append((val, inp, ind))
        # Tip: Sort list of tuples in-place
        multicoll_list.sort(key=lambda tup: tup[0], reverse=True)
    return multicoll_list


def del_multicoll_from_mean_corrs(multicoll, mean_corr):
    for item in multicoll:
        if item[1] in mean_corr and item[2] in mean_corr:
            # NOTE: just giving a bit of priority to CPU compared to threads_or_copies
            if (
                item[1] == "log_cpus"
                and item[2] == "log_threads_or_copies"
                and mean_corr[item[2]] - mean_corr[item[1]] < 0.05
            ):
                del mean_corr[item[2]]
            elif mean_corr[item[1]] > mean_corr[item[2]]:
                # Tip: Dict, remove an item
                del mean_corr[item[2]]
            elif mean_corr[item[1]] < mean_corr[item[2]]:
                del mean_corr[item[1]]
            else:  # equal: just delete the second!!
                del mean_corr[item[2]]
    sorted_mean_cor = sorted(mean_corr.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_mean_cor


def sort_feat(votes, all_features, cat):
    feat_sum = {f: s for (f, s) in zip(all_features[cat], votes[cat]["sum"])}
    sorted_feat = sorted(feat_sum.items(), key=lambda kv: kv[1], reverse=True)
    # print(sorted_feat)
    return [k for (k, v) in sorted_feat]


def kbest_sort_feat(votes, all_features, cat):
    feat_sum = {f: s for (f, s) in zip(all_features[cat], votes[cat])}
    sorted_feat = sorted(feat_sum.items(), key=lambda kv: kv[1], reverse=True)
    return [k for (k, v) in sorted_feat]

