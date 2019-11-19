import click
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from model import (
    build_selector,
    evaluate_feature_search,
    get_feature_names,
    load_config,
    load_data,
    output_predictions,
    permtest_evaluate,
    permutation_test,
    plot_box,
    plot_coefficients,
    plot_confusion_matrix,
    plot_feature_search,
    plot_pca,
    plot_permutation_violins,
    select_features,
)

np.random.seed(88)

RF = RandomForestClassifier(
    n_jobs=-1, n_estimators=100, random_state=88, class_weight="balanced"
)
LR = LogisticRegression(solver="lbfgs", multi_class="auto")
BRF = BalancedRandomForestClassifier(n_jobs=-1, random_state=88)


def select(config):
    select_features(LR, config_file=config)
    evaluate_feature_search(config_file=config)


def permtest(config):
    permutation_test(LR, config_file=config)
    permtest_evaluate(config_file=config)


def evaluate(config):
    output_predictions(config_file=config)
    plot_box(config_file=config)
    plot_pca(config_file=config)
    plot_confusion_matrix(config_file=config)
    if load_config(config)["ESTIMATOR"]["CODE"] == "LR":
        plot_coefficients(config_file=config)


@click.command()
@click.option("--command", help="The command to execute.")
@click.option("--config", help="The command to execute.")
def interface(command, config):
    if command == "select":
        select(config=config)
    elif command == "permtest":
        permtest(config=config)
    elif command == "evaluate":
        evaluate(config=config)
    else:
        click.echo(
            'You must specify a command! Options: ["select", "permtest", "evaluate"]'
        )


if __name__ == "__main__":
    interface()
