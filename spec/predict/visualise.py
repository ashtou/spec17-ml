import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve

sns.set()  # set aesthetic parameters in one step.
sns.set_style("whitegrid")

XS = 6
SM = 8
MD = 10
LG = 12
XL = 14

# TODO: better to move it inside the get_color() function
colors = [
    "mango",  # "amber",
    "rose pink",
    "light greenish blue",
    "deep rose",  # "scarlet",
    "bluish green",
    "marine",  # "grey",
    "light urple",  # "wisteria", #"dark"
]


def get_estimator_label(est):
    label = (
        "Permutation_" + est.estimator.regressor.__class__.__name__
        if hasattr(est, "estimator")
        else est.regressor.__class__.__name__
    )
    if "MLPRegressor" in label:
        label += "_lbfgs" if "lbfgs" in str(est) else "_adam"

    return label


def get_est_short_label(est):
    sh_label = get_estimator_label(est)
    sh_label = (
        sh_label.replace("Permutation", "PI")
        .replace("MultiTaskElasticNet", "MT_EN")
        .replace("DecisionTreeRegressor", "DT")
        .replace("RandomForestRegressor", "RF")
        .replace("MLPRegressor", "MLP")
    )

    return sh_label


def get_linestyle(est):
    label = get_estimator_label(est)
    if "Permutation" in label:
        return "-"
    return "--"


def get_color(ind):
    return sns.xkcd_palette(colors)[ind]
    # colors = [
    #     "#fdb462",
    #     "#fb8072", #"#e31a1c",
    #     "#74c476",
    #     "#67001f",
    #     "#1b7837",
    #     "#6a3d9a",
    #     "#02d8e9", # aqua blue
    # ]
    # return colors[ind]


def get_bar_color(ind):
    bar_colors = [
        "mango",  # "amber",
        "rose pink",
        "turquoise",
        "deep rose",  # "scarlet",
        "greenish blue",
        "marine",  # "grey",
        "greyish purple",  # "wisteria", #"dark"
    ]
    return sns.xkcd_palette(bar_colors)[ind]


def get_marker(est):
    label = get_estimator_label(est)
    if "Tree" in label:
        return ".-"
    if "Forest" in label:
        return "-"
    if "Elastic" in label:
        return "v-"
    if "adam" in label:
        return "|-"
    if "lbfgs" in label:
        return "|-"
    return "-"


def get_short_features(features):
    sh_dict = {
        "nominal_mhz": "nom_mhz",
        # "max_mhz": "max_mhz",
        # cpus is log-transformed
        "cpus": "log_cpus",
        "threads_per_core": "threads/core",
        "cores_per_socket": "cores/socket",  # "cores/skt",
        # "sockets": "skts",
        # "numas": "numas",
        "l1d_cache_kb": "l1d_cache",
        "l1i_cache_kb": "l1i_cache",
        "l2_cache_kb": "l2_cache",
        "l3_cache_kb": "l3_cache",
        # mem_kb is log-transformed
        "mem_kb": "log_mem_kb",
        "mem_channels": "mem_chnl",
        "channel_kb": "chnl_kb",
        "mem_data_rate": "mem_rate",
        "COMP_Intel": "CMP_Intel",
        "IA_COMP_Intel": "IA_CMP_Intel",
        "OS_RHEL": "OS_RHEL",
        "OS_SLES": "OS_SLES",
        "IA_OS_SLES": "IA_OS_SLES",
        # threads_or_copies in log-transfomred
        "threads_or_copies": "log_th_copies",
    }
    sh_feats = []
    for feat in features:
        if feat in sh_dict:
            sh_feats.append(sh_dict[feat])
        else:
            sh_feats.append(feat)

    return sh_feats


# def plot_stacked_bar(
#     data,
#     series_labels,
#     category_labels=None,
#     show_values=False,
#     value_format="{}",
#     y_label=None,
#     colors=None,
#     grid=True,
#     reverse=False,
# ):
#     """Plots a stacked bar chart with the data and labels provided.

#     Keyword arguments:
#     data            -- 2-dimensional numpy array or nested list
#                        containing data for each series in rows
#     series_labels   -- list of series labels (these appear in
#                        the legend)
#     category_labels -- list of category labels (these appear
#                        on the x-axis)
#     show_values     -- If True then numeric value labels will
#                        be shown on each bar
#     value_format    -- Format string for numeric value labels
#                        (default is "{}")
#     y_label         -- Label for y-axis (str)
#     colors          -- List of color labels
#     grid            -- If True display grid
#     reverse         -- If True reverse the order that the
#                        series are displayed (left-to-right
#                        or right-to-left)
#     """

#     ny = len(data[0])
#     ind = list(range(ny))

#     axes = []
#     cumm_size = np.zeros(ny)

#     data = np.array(data)

#     if reverse:
#         data = np.flip(data, axis=1)
#         category_labels = reversed(category_labels)

#     for i, row_data in enumerate(data):
#         color = colors[i] if colors is not None else None
#         axes.append(
#             plt.bar(ind, row_data, bottom=cum_size, label=series_labels[i], color=color)
#         )
#         cumm_size += row_data

#     if category_labels:
#         plt.xticks(ind, category_labels)

#     if y_label:
#         plt.ylabel(y_label)

#     plt.legend()

#     if grid:
#         plt.grid()

#     if show_values:
#         for axis in axes:
#             for bar in axis:
#                 w, h = bar.get_width(), bar.get_height()
#                 plt.text(
#                     bar.get_x() + w / 2,
#                     bar.get_y() + h / 2,
#                     value_format.format(h),
#                     ha="center",
#                     va="center",
#                 )


def plot_line_rfecv(rfecv_ests, rfecv_pipe, cat, linewidth, vert_line, size):
    plt.rc("font", size=MD)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=LG)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=size)
    ax.set_xlabel("Selected Features by RFECV")
    ax.set_ylabel("Negated Avg. MAE")
    for ind, est in enumerate(rfecv_ests):
        num_feats = range(1, len(rfecv_pipe[cat][ind]["rfecv"].grid_scores_) + 1)
        ax.plot(
            num_feats,
            rfecv_pipe[cat][ind]["rfecv"].grid_scores_,
            get_marker(est),
            linestyle=get_linestyle(est),
            linewidth=linewidth,
            label=get_est_short_label(est),
            color=get_color(ind),
        )
    plt.xticks(num_feats)
    # ax.grid(linestyle=':', linewidth=0.3)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    plt.axvline(x=vert_line, color="gray", linestyle="--")
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.grid(axis="y", color="#dddddd")
    plt.tick_params(axis="y", length=0)
    plt.ylim(None, 0)
    plt.legend()
    plt.title(cat, fontdict={"size": LG})
    plt.savefig(
        "plots/feature_selection/rfe_" + cat + ".pdf", bbox_inches="tight", dpi=300
    )
    plt.show()
    plt.close()


def plot_barh_importance(y, width, color, yticklabels, ylim, title, cat, sh_label):
    """Feature importance barh plot for linear models"""
    sns.set()
    fig, ax = plt.subplots(figsize=(2, 3))

    ax.barh(y=y, width=width, height=0.7, color=color)
    ax.tick_params(axis="y", which="major", labelsize=LG)  # "both"

    ax.set_yticklabels(yticklabels)
    ax.set_yticks(y)
    ax.set_ylim(ylim)
    if "EN" in sh_label:
        ax.set_ylabel(cat, fontsize=XL)
    # ax.title.set_text(title)
    plt.savefig(
        "plots/feature_selection/imp_" + cat + "_" + sh_label + ".pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    # plt.show()
    plt.close()


def plot_box_importance(x, vert, labels, title, ind, cat, sh_label):
    """Feature importance box plot"""
    sns.set()
    fig, ax = plt.subplots(figsize=(2, 3))
    bp = ax.boxplot(x=x, vert=vert, labels=labels, patch_artist=True)
    ax.tick_params(axis="y", which="major", labelsize=LG)  # "both"

    for patch in bp["boxes"]:
        patch.set(facecolor=get_color(ind))
    # ax.title.set_text(title)
    plt.savefig(
        "plots/feature_selection/imp_" + cat + "_" + sh_label + ".pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    # plt.show()
    plt.close()


def plot_influence():
    # deprecated
    return


def plot_tradeoff(
    prediction_performances, prediction_times, xticks, ests, TOP_K, size, cat, save=True
):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=size)
    bar_width = 0.08

    perf_bars = [
        prediction_performances[i * TOP_K : (i + 1) * TOP_K] for i in range(len(ests))
    ]
    delay_bars = [
        prediction_times[i * TOP_K : (i + 1) * TOP_K] for i in range(len(ests))
    ]
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # Set position of bar on X axis
    pos = []

    # pos.append(np.arange(len(bars[0])))               # normal grouping (one from each category)
    pos.append([i * bar_width for i in range(TOP_K)])  # sequential grouping
    axes[0].bar(
        pos[0],
        perf_bars[0],
        color=get_bar_color(0),
        width=bar_width,
        edgecolor="white",
        label="var0",
    )
    for c in range(1, len(ests)):
        # pos.append([x + bar_width for x in pos[c-1]])         # normal grouping (one from each category)
        pos.append([(c + i * bar_width) for i in range(TOP_K)])  # sequential grouping
        # Make the plot
        axes[0].bar(
            pos[c],
            perf_bars[c],
            color=get_bar_color(c),
            width=bar_width,
            edgecolor="white",
            label="var" + str(c),
        )
    # axes[0].set(title=cat + ": Prediction Error vs. Time")
    # axes[0].text(
    #     0.5,
    #     0.85,
    #     cat + ": Prediction Error vs. Time",
    #     horizontalalignment="center",
    #     transform=axes[0].transAxes,
    #     bbox=props,
    #     size=MD,
    # )
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(False)
    axes[0].set_ylabel("MAE", size=MD)
    axes[0].grid(axis="y", color="#dddddd", ls="--")

    # axes[1]
    axes[1].bar(
        pos[0],
        delay_bars[0],
        color="none",
        edgecolor=get_bar_color(0),
        width=bar_width,
        label="var0",
    )
    for c in range(1, len(ests)):
        pos.append([(c + i * bar_width) for i in range(TOP_K)])  # sequential grouping
        axes[1].bar(
            pos[c],
            delay_bars[c],
            color="none",
            edgecolor=get_bar_color(c),
            width=bar_width,
            label="var" + str(c),
        )

    # axes[1].text(
    #     0.5,
    #     0.85,
    #     "CV Pred Time (s)",
    #     horizontalalignment="center",
    #     transform=axes[1].transAxes,
    #     bbox=props,
    #     size=MD,
    # )
    # axes[1].invert_yaxis()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(False)
    axes[1].grid(axis="y", color="#dddddd", ls="--")
    axes[1].set_ylabel("Time (s)", size=MD)
    axes[1].set_yticks([0, 0.02, 0.04])

    # Add xticks on the middle of the group bars
    fig.text(-0.02, 0.5, cat, va="center", rotation="vertical")
    # plt.xticks([r + 4*bar_width for r in range(len(bars[0]))], ['A', 'B', 'C', 'D', 'E', 'F', 'G']) # normal grouping
    plt.xticks(
        [r + 4 * bar_width for r in range(len(ests))],
        [get_est_short_label(ests[i]) for i in range(len(ests))],
        fontsize=12,
    )

    fig.tight_layout()
    # align ylabels
    fig.align_ylabels(axes)
    if save:
        plt.savefig(
            "plots/model_selection/tradeoff_" + cat + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()
    plt.close()


def plot_learning_curve(
    pipe,
    est,
    X,
    y,
    fig,
    axes,
    i,
    cat,
    ylim=None,
    cv=None,
    scoring=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    """
    Generate multiple plots: the test and training learning curve, the training
    samples vs fit times curve, 
    """
    # and the fit times vs score curve.

    axes[i].set_title(get_est_short_label(est) + " Learning Curve")
    # if ylim is not None:
    #     axes[i].set_ylim(*ylim)

    axes[i].set_xlabel("Train set size")
    axes[0].set_ylabel("Score (neg MAE)")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    # axes[0][i].grid()
    axes[i].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="xkcd:navy blue",
    )
    axes[i].plot(
        train_sizes,
        train_scores_mean,
        ".-",
        color="xkcd:navy blue",
        label="Train score",
    )
    axes[i].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=get_color(i + 3),
    )
    axes[i].plot(
        train_sizes, test_scores_mean, ".-", color=get_color(i + 3), label="CV score"
    )
    axes[i].legend(loc="lower right")
    axes[i].grid(False)
    axes[i].grid(axis="y", color="#dddddd", ls="--")
    # turn on tick labels when the sharey option is used
    # axes[0][i].yaxis.set_tick_params(labelbottom=True)
    axes[i].spines["left"].set_color("#aaaaaa")
    axes[i].spines["bottom"].set_color("#aaaaaa")
    axes[i].spines["right"].set_visible(False)
    axes[i].spines["top"].set_visible(False)
    if i == 1:
        axes[1].set_ylim(axes[0].get_ylim())
    fig.text(-0.02, 0.5, cat, va="center", rotation="vertical")

    # Plot n_samples vs fit_times
    axes[2].set_title("Training Time")
    axes[2].plot(
        train_sizes,
        fit_times_mean,
        ".-",
        color=get_color(i + 3),
        label=get_est_short_label(est),
    )
    axes[2].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
        color=get_color(i + 3),
    )
    axes[2].set_xlabel("Train set size")
    axes[2].set_ylabel("Time (s)")
    # axes[1].set_title("Scalability of the model")
    axes[2].grid(False)
    axes[2].grid(axis="y", color="#dddddd", ls="--")
    # axes[2].yaxis.set_tick_params(labelbottom=True)
    axes[2].spines["left"].set_color("#aaaaaa")
    axes[2].spines["bottom"].set_color("#aaaaaa")
    axes[2].spines["right"].set_visible(False)
    axes[2].spines["top"].set_visible(False)
    axes[2].legend(loc="upper left")

    # # Plot fit_time vs score
    # axes[3].plot(fit_times_mean, test_scores_mean, "o-")
    # axes[3].fill_between(
    #     fit_times_mean,
    #     test_scores_mean - test_scores_std,
    #     test_scores_mean + test_scores_std,
    #     alpha=0.1,
    # )
    # axes[3].set_xlabel("fit_times")
    # if i == 0:
    #     axes[3].set_ylabel("Score")
    # # axes[3].set_title("Performance of the model")
    # axes[3].grid(False)
    # axes[3].grid(axis="y", color="#dddddd", ls="--")

    return
