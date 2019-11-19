from __future__ import division

import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from functions import CorrelationThreshhold, split_dataset

np.random.seed(88)


def load_config(filename="config.yml"):
    """Loads a configuration file from a  specified location.
    
    Args:
        filename (str, optional): Config file path. Defaults to "config.yml".
    
    Returns:
        config (dict): Dictionary of config settings.
    """
    with open(filename, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if config["RUN_DATE"] == "auto":
        now = str(datetime.now().strftime("%Y%m%d"))
    else:
        now = config["RUN_DATE"]

    config["SELECTOR"] = "../output/{}_{}_selector.joblib".format(
        now, config["RUN_NAME"]
    )
    config["PERMUTATION_TEST"] = "../output/{}_{}_permtest.joblib".format(
        now, config["RUN_NAME"]
    )
    config["REPORT"] = "../output/{}_{}_classifier_report.csv".format(
        now, config["RUN_NAME"]
    )
    return config


def load_data(DATA, meta_cols=False, meta_return=False, config_file="config.yml"):
    """Loads data from .csv file, optionally removing columns 
    that contain metadata better used for grouping than for modeling.
    
    Args:
        DATA (str): Path to .csv file containing data.
        meta_cols (bool, optional): True if there are metadata columns to remove. Defaults to False.
        meta_return (bool, optional): True to return the metadata columns as a separate DataFrame. Defaults to False.
    
    Returns:
        X (DataFrame): DataFrame of features.
        y (DataFrame): DataFrame of labels.
        meta (DataFrame, optional): DataFrame of metadata.
    """
    config = load_config(config_file)
    X = pd.read_csv(DATA)
    X = X.dropna(axis=1)
    y = X.pop(config["TARGET"])

    # Drop columns containing metadata
    if meta_cols == False:
        if config["META_COLS"] != "N":
            if len(config["META_COLS"]) == 1:
                meta = X.pop(config["META_COLS"][0])
            else:
                meta = X[config["META_COLS"]].copy()
                X = X.drop(config["META_COLS"], axis=1).copy()
    if meta_return == True and config["META_COLS"] != "N":
        return X, y, meta
    else:
        return X, y


def get_steps(type, config_file="config.yml"):
    config = load_config(config_file)

    if type == "selector":
        steps = [
            ("correlation_filter", CorrelationThreshhold()),
            ("logscale", FunctionTransformer(np.log1p, validate=True)),
            ("normalize", StandardScaler()),
            ("mutual_info", SelectKBest(score_func=mutual_info_classif, k=10)),
        ]
    elif type == "classifier":
        steps = [
            ("logscale", FunctionTransformer(np.log1p, validate=True)),
            ("normalize", StandardScaler()),
        ]

    processed_steps = []

    for step in steps:
        key = step[0].upper()
        if config[key]["USE"] == "Y":
            processed_steps.append(step)
    
    return processed_steps


def build_selector(X, y, clf, k=30, config_file="config.yml"):
    """Creates a feature selection pipeline.
    
    Args:
        X (DataFrame): DataFrame of features.
        y (DataFrame): DataFrame of labels.
        clf (sklearn classifier): Model for classification.
        k (int, optional): Maximum size of featureset. Defaults to 30.
    
    Returns:
        (sklearn Pipeline): Feature selection pipeline.
    """
    config = load_config(config_file)
    # Instantiate cross-validation scheme
    stratkfold = RepeatedStratifiedKFold(
        n_splits=config["CV"]["FOLDS"],
        n_repeats=config["CV"]["REPEATS"],
        random_state=88,
    ).split(X, y)
    cv = list(stratkfold)

    # Instantiate sequential forward floating selection object
    sffs = SFS(
        clf,
        k_features=k,
        forward=True,
        floating=True,
        verbose=0,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    # Instantiate pipeline steps

    steps = get_steps("selector", config_file = config_file)
    steps += [("sffs", sffs)]
    # Instantiate feature selection pipeline
    selector = Pipeline(steps=steps)

    return selector


def build_classifier(config_file="config.yml"):
    """Creates a classification pipeline, using settings from the feature selection pipeline.
    
    Returns:
        (sklearn Pipeline): Classification pipeline.
    """
    config = load_config(config_file)
    selector = joblib.load(config["SELECTOR"])
    # Instantiate pipeline steps
    steps = get_steps("classifier", config_file=config_file)
    steps += [("clf", selector.steps[-1][1].estimator)]

    # Instantiate classification pipeline
    classifier = Pipeline(steps=steps)

    return classifier


def build_transformer():
    """Creates a data preprocessing pipeline.
    
    Returns:
        (sklearn Pipeline): Preprocessing pipeline.
    """
    # Instantiate pipeline steps
    steps = [
        ("logscale", FunctionTransformer(np.log1p, validate=True)),
        ("normalize", StandardScaler()),
    ]

    # Instantiate classification pipeline
    transformer = Pipeline(steps=steps)

    return transformer


def select_features(
    clf=RandomForestClassifier(n_estimators=10, random_state=88),
    config_file="config.yml",
):
    """Runs a feature selection pipeline, saving the results to a *.joblib file.
    
    Args:
        clf (sklearn classifier): Scikit-learn classifier. Can be any classifier that supplies both a predicted class and probabilities.
    """
    config = load_config(config_file)
    X, y = load_data(config["MAIN"], config_file=config_file)
    print("Searching for features...")
    selector = build_selector(X, y, clf, config_file=config_file)

    selector.fit(X, y)

    joblib.dump(selector, config["SELECTOR"])
    print("Selector object saved at {}".format(config["SELECTOR"]))
    pass


def evaluate_feature_search(test=False, config_file="config.yml"):
    """Plots the accuracy of each featureset identified by the selection pipeline. Saves to disk as a *.png file.
    
    Args:
        test (bool, optional): True if a separate test set is available. Defaults to False.
    """
    config = load_config(config_file)
    selector = joblib.load(config["SELECTOR"])
    X, y = load_data(config["MAIN"], config_file=config_file)
    if test == False:
        formatted_filename = plot_feature_search(selector, config_file=config_file)
    if test == True:
        X_test, y_test = load_data(config["TEST"], config_file=config_file)
        formatted_filename = plot_feature_search(
            selector, X, y, X_test, y_test, config_file=config_file
        )
    print("Plot saved at {}".format(formatted_filename))
    pass


def permutation_test(
    clf=RandomForestClassifier(n_estimators=10, random_state=88),
    config_file="config.yml",
    k="auto",
):
    """Executes feature selection pipeline with permuted labels, repeats ten times.
    
    Args:
        clf (sklearn classifier): Scikit-learn classifier. Can be any classifier that supplies both a predicted class and probabilities.
        k (int or str, optional): Maximum number of features for test. Bound by size of best non-permuted feature set to prevent overfitting. Defaults to config['K_FEATURES'].
    """
    config = load_config(config_file)
    if k == "auto":
        k = config["K_FEATURES"]
    X, y = load_data(config["MAIN"], config_file=config_file)
    permutation_dict = {}
    for i in range(10):
        print("Searching over permutation #{}".format(i + 1))
        y_perm = np.random.RandomState(seed=i).permutation(y)
        selector = build_selector(X, y_perm, clf, k=k, config_file=config_file)
        selector.fit(X, y_perm)
        permutation_dict[i + 1] = selector
    joblib.dump(permutation_dict, config["PERMUTATION_TEST"])
    print("Permutation test object saved at {}".format(config["PERMUTATION_TEST"]))
    pass


def permtest_evaluate(config_file="config.yml", k="auto", features=None):
    """Produces violin plots of real accuracy vs permuted accuracy.
    
    Args:
        k ([type], optional): [description]. Defaults to config['K_FEATURES'].
        features ([type], optional): [description]. Defaults to None.
    """
    config = load_config(config_file)
    if k == "auto":
        k = config["K_FEATURES"]
    true = joblib.load(config["SELECTOR"])
    permuted = joblib.load(config["PERMUTATION_TEST"])
    # get true cv scores
    if type(features) == list:
        X, y = load_data(config["MAIN"], config_file=config_file)
        classifier = build_classifier(config_file=config_file)
        stratkfold = RepeatedStratifiedKFold(
            n_splits=config["CV"]["FOLDS"],
            n_repeats=config["CV"]["REPEATS"],
            random_state=88,
        ).split(X, y)
        cv = list(stratkfold)
        true_scores = cross_val_score(classifier, X[features], y, cv=cv)
    else:
        true_scores = list(true.steps[-1][1].get_metric_dict()[k]["cv_scores"])
        true_scores = [
            true_scores[i : i + config["CV"]["FOLDS"]]
            for i in range(0, len(true_scores), config["CV"]["FOLDS"])
        ]
        true_scores = [np.mean(x) for x in true_scores]
    # get cv scores from permutation test
    permuted_scores = []
    for i in permuted:
        scores = list(permuted[i].steps[-1][1].get_metric_dict()[k]["cv_scores"])
        permuted_scores.append(np.mean(scores))
    plot_filename = plot_permutation_violins(
        true_scores, permuted_scores, config_file=config_file
    )
    print("Permutation test plot saved at {}".format(plot_filename))
    pass


def output_predictions(config_file="config.yml", k="auto", features=None):
    """Outputs a .csv file of predicted probabilities and labels, annotated with metadata if available.
    
    Args:
        k ([type], optional): [description]. Defaults to config['K_FEATURES'].
        features ([type], optional): [description]. Defaults to None.
    
    Returns:
        report (DataFrame): DataFrame of predictions and probabilities.
    """
    config = load_config(config_file)
    if k == "auto":
        k = config["K_FEATURES"]
    if type(features) != list:
        features = get_feature_names(k, config_file=config_file)
    if config["META_COLS"] != "N":
        X, y, meta = load_data(
            config["MAIN"], config_file=config_file, meta_return=True
        )
    else:
        X, y = load_data(config["MAIN"], config_file=config_file)

    classifier = build_classifier(config_file=config_file)
    classifier.fit(X[features], y)
    y_pred = classifier.predict(X[features])
    y_prob = classifier.predict_proba(X[features])[:, 1]
    group = ["MAIN" for x in y_pred]
    report = pd.DataFrame(
        {"group": group, "target": y, "prediction": y_pred, "probability": y_prob}
    )
    if config["META_COLS"] != "N":
        report = pd.concat([meta, report], axis=1)

    if config["TEST"]:
        if config["META_COLS"] != "N":
            Xt, yt, metat = load_data(
                config["TEST"], meta_return=True, config_file=config_file
            )
        else:
            Xt, yt = load_data(config["TEST"], config_file=config_file)
        yt_pred = classifier.predict(Xt[features])
        yt_prob = classifier.predict_proba(Xt[features])[:, 1]
        groupt = ["TEST" for x in yt_pred]
        report_test = pd.DataFrame(
            {
                "group": groupt,
                "target": yt,
                "prediction": yt_pred,
                "probability": yt_prob,
            }
        )
        if config["META_COLS"] != "N":
            report_test = pd.concat([metat, report_test], axis=1)
        report = pd.concat([report, report_test], axis=0)
    report.to_csv(config["REPORT"], index=False)
    print("Predictions saved at {}".format(config["REPORT"]))
    return report


######################
# PLOTTING FUNCTIONS #
######################


def plot_permutation_violins(
    pipeline_scores, permuted_scores, config_file="config.yml"
):
    """Create violin plots of true vs permuted pipeline runs.
    
    Args:
        pipeline_scores (list): List of accuracy scores, true labels.
        permuted_scores (list): List of accuracy scores, permuted labels.
    
    Returns:
        plot_filename (str): Output file location.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Accuracy vs Run | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_permutation_violins.png".format(
        now, config["RUN_NAME"]
    )

    pipeline = pd.DataFrame({"Run": "True", "Accuracy": pipeline_scores})
    permuted = pd.DataFrame({"Run": "Permuted", "Accuracy": permuted_scores})
    for_plotting = pd.concat([pipeline, permuted], axis=0)

    _, p = ttest_ind(pipeline_scores, permuted_scores)
    d, _ = cliffsDelta(pipeline_scores, permuted_scores)
    p = process_pval(p)
    d = str(round(d, 3))

    for_plotting["pval"] = p
    for_plotting["cliffsdelta"] = d
    for_plotting.to_csv(plot_filename.split(".")[0] + ".csv")
    current_palette = sns.color_palette()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        x="Run",
        y="Accuracy",
        data=for_plotting,
        bw="scott",
        cut=0,
        palette=[current_palette[1], current_palette[0]],
        ax=ax,
    )
    plt.axhline(0.5, color="r", linestyle="--")
    plt.axhline(np.mean(pipeline_scores), color="darkorange", linestyle="--")
    plt.axhline(np.mean(permuted_scores), color="darkblue", linestyle="--")
    plt.ylim([-0.1, 1.1])
    plt.title(plot_title)
    plt.text(
        0.5,
        0.1,
        "P Value: " + p + "\nCliff's Delta: " + d,
        horizontalalignment="center",
    )
    plt.tight_layout()
    plt.savefig(plot_filename)
    return plot_filename


def plot_feature_search(
    pipeline, X=None, y=None, X_test=None, y_test=None, config_file="config.yml"
):
    """[summary]
    
    Args:
        pipeline ([type]): [description]
        X ([type], optional): [description]. Defaults to None.
        y ([type], optional): [description]. Defaults to None.
        X_test ([type], optional): [description]. Defaults to None.
        y_test ([type], optional): [description]. Defaults to None.
    
    Returns:
        [type]: [description]
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)
    plot_title = "SFFS Accuracy vs Number of Features | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    search_object = pipeline.steps[-1][1]
    clf = search_object.estimator

    crossval_scores = pd.DataFrame(
        {
            key: val
            for key, val in zip(
                search_object.get_metric_dict().keys(),
                [
                    search_object.get_metric_dict()[x]["cv_scores"]
                    for x in search_object.get_metric_dict().keys()
                ],
            )
        }
    )
    crossval_scores = crossval_scores.melt()

    if type(X_test) == pd.core.frame.DataFrame:
        test_index = []
        test_accuracy = []
        evaluator = build_classifier()
        for x in search_object.get_metric_dict().keys():
            test_features = get_feature_names(x, config_file=config_file)
            print("Features for set {}:\n{}".format(x, test_features))
            evaluator.fit(X[test_features], y)
            predictions = evaluator.predict(X_test[test_features])
            accuracy = accuracy_score(y_test, predictions)
            test_index.append(x)
            test_accuracy.append(accuracy)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.lineplot(
        x="variable",
        y="value",
        data=crossval_scores,
        ci="sd",
        ax=ax,
        label="Cross-Validation",
    )
    if type(X_test) == pd.core.frame.DataFrame:
        plt.plot(test_index, test_accuracy, label="Test Accuracy")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title(plot_title)

    num_features = len(crossval_scores.variable.unique())
    plt.xticks(list(range(1, num_features + 1)), rotation=90)
    plt.ylim([-0.05, 1.15])
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plot_filename = "../output/{}_{}_sffs_plot.png".format(now, config["RUN_NAME"])
    plt.savefig(plot_filename)
    return plot_filename


def plot_coefficients(k='auto', config_file="config.yml"):
    """Plots coefficient bars. Logistic Regression only.
    
    Args:
        k (int, optional): Feature set size to plot. Defaults to config file setting.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Coefficients | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_coefficients.png".format(now, config["RUN_NAME"])

    features = get_feature_names(k, config_file=config_file)
    coefficients = get_coefficients(features, config_file=config_file)

    df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})

    largest_coefficient = max(np.abs(coefficients))
    ymax = largest_coefficient + 0.10 * largest_coefficient

    df = df.sort_values("Coefficient", ascending=False)
    df["Color"] = df.Coefficient.apply(lambda x: "#7cfc00" if x >= 0 else "#551a8b")

    df.to_csv(plot_filename.split(".")[0] + ".csv")

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x="Feature", y="Coefficient", data=df, palette=df["Color"])
    plt.ylim([-ymax, ymax])
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print("Coefficients plot saved at {}".format(plot_filename))
    pass


def plot_confusion_matrix(filename=None, config_file="config.yml"):
    """[summary]
    
    Args:
        filename ([type], optional): [description]. Defaults to None.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Confusion Matrix | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_confusion_matrix.png".format(
        now, config["RUN_NAME"]
    )

    if filename == None:
        filename = config["REPORT"]

    df = pd.read_csv(filename)
    if config['TEST'] == True:
        df = df.loc[df.group == "TEST"].copy()
    y = df["target"]
    y_pred = df["prediction"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), cmap="Greens", annot=True, fmt="d")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print("Confusion matrix plot saved at {}".format(plot_filename))
    pass


def plot_pca(config_file="config.yml"):
    """[summary]
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "PCA Scatterplot | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_PCA.png".format(now, config["RUN_NAME"])

    features = get_feature_names(k="auto", config_file=config_file)
    X, y = load_data(config["MAIN"], config_file=config_file)

    pca = PCA(n_components=2, random_state=88)
    components = pca.fit_transform(X[features])

    datamap = pd.concat([y, pd.DataFrame(components)], axis=1)
    datamap.columns = ["Group", "First Component", "Second Component"]
    datamap.to_csv(plot_filename[:-4] + ".csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sns.scatterplot(
        x="First Component",
        y="Second Component",
        hue="Group",
        data=datamap,
        palette=["#7cfc00", "#551a8b"],
        ax=ax,
    )
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    pass


def plot_box(filename=None, config_file="config.yml"):
    """[summary]
    
    Args:
        filename ([type], optional): [description]. Defaults to None.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Predicted Probability by Group | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_boxplots.png".format(now, config["RUN_NAME"])

    if filename == None:
        filename = config["REPORT"]

    df = pd.read_csv(filename)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        x="group",
        y="probability",
        hue="target",
        data=df,
        palette=["plum", "palegreen"],
        hue_order=config["CLASSES"],
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        x="group",
        y="probability",
        hue="target",
        data=df,
        color="black",
        dodge=True,
        hue_order=config["CLASSES"],
        ax=ax,
    )
    plt.axhline(0.5, linestyle="--", color="black")
    plt.axvline(0.5, linestyle="--", color="black")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Predicted Probability - {}".format(config["CLASSES"][1]))
    plt.xlabel("Group")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1.025, 0.5))
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    pass


#####################
# UTILITY FUNCTIONS #
#####################


def get_feature_names(k="auto", selector_object=None, config_file="config.yml"):
    """[summary]
    
    Args:
        k ([type], optional): [description]. Defaults to None.
        selector_object ([type], optional): [description]. Defaults to None.
    
    Returns:
        [type]: [description]
    """
    config = load_config(config_file)
    if k == "auto":
        k = config["K_FEATURES"]

    if selector_object == None:
        selector = joblib.load(config["SELECTOR"])
    else:
        selector = selector_object
    steps = {}
    for name, step in selector.steps:
        steps[name] = step

    X, _ = load_data(config["MAIN"], config_file=config_file)
    feature_names = pd.Series(X.columns)
    if "correlation_filter" in steps.keys():
        feature_names = pd.Series(steps["correlation_filter"].feature_names)
    if "mutual_info" in steps.keys():
        kbest_idx = steps["mutual_info"].get_support()
        feature_names = feature_names[kbest_idx].reset_index(drop=True)

    sffs_idx = list(steps["sffs"].get_metric_dict()[k]['feature_idx'])

    selected_features = feature_names[sffs_idx]
    return selected_features


def get_coefficients(feature_names, config_file="config.yml"):
    """[summary]
    
    Args:
        feature_names ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    config = load_config(config_file)
    X, y = load_data(config["MAIN"], config_file=config_file)
    classifier = build_classifier(config_file=config_file)
    print("Calculating coefficients...")
    classifier.fit(X[feature_names], y)
    coefficients = classifier.steps[-1][1].coef_[0]
    return coefficients


def process_pval(p):
    if p < 0.001:
        return "< 0.001"
    else:
        return str(round(p, 3))


###########################
# CLIFF'S DELTA FUNCTIONS #
###########################


def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and pipeline if there are more than 'dull' differences"""
    if not dull:
        dull = {
            "small": 0.147,
            "medium": 0.33,
            "large": 0.474,
        }  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull["small"]:
        return "negligible"
    if dull["small"] <= delta < dull["medium"]:
        return "small"
    if dull["medium"] <= delta < dull["large"]:
        return "medium"
    if delta >= dull["large"]:
        return "large"


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two
