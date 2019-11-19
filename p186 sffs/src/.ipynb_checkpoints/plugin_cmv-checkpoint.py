from datetime import datetime

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class CMVFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects CMV-specific antigens only.
    
    If igm=False, all IgM features are removed.
    """

    def __init__(self, jason=True, igm=False):
        self.jason = jason
        self.igm = igm
        if self.jason == True:
            self.cmv_antigens = ["prefusion gB", "postfusion gB", "CG1", "CG2", "pentamer - McLellan", "pentamer", "gB"]
        else:
            self.cmv_antigens = ["CG1", "CG2", "pentamer", "gB"]
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pair_tuples = [x.split("_") for x in X.columns]
        cmv_tuples = [x for x in pair_tuples if x[1] in self.cmv_antigens]
        cmv_cols = ["_".join(x) for x in cmv_tuples]
        if self.igm == False:
            cmv_cols = [x for x in cmv_cols if x.startswith("IgM") == False]
        return X[cmv_cols].copy()

def make_datasets(df_path):
    df = pd.read_csv(df_path)
    df = df.loc[df.Cohort.isin(["PP", "NP", "PL", "NL", "PP_Erasmus"])].copy()
    df.Sample = df.Sample.apply(sample_cleaner)
    # drops columns with missing values
    # comment out following line if you do not want this!
    df = df.dropna(axis=1)
    ERASMUS, LONGITUDINAL, MAIN, TRAIN, TEST = split_dataset(df)
    print("Erasmus set saved at: {}".format(ERASMUS))
    print("Longitudinal set saved at: {}".format(LONGITUDINAL))
    print("Main set saved at: {}".format(MAIN))
    print("Train set (subset of main) saved at: {}".format(TRAIN))
    print("Test set (subset of main) saved at: {}".format(TEST))
    return ERASMUS, LONGITUDINAL, MAIN, TRAIN, TEST

def sample_cleaner(sample):
    """Standardizes sample names."""
    sample = sample.replace("-", "").lower().strip()
    return sample


def drop_missval_cols(df):
    """Drops columns with an excessive number of missing values."""
    na_series = df.isna().sum().sort_values(ascending=False)
    na_labels = na_series[na_series > 25].index
    new_df = df.drop(na_labels, axis=1)
    return new_df


def find_final_visits(value):
    """Determines whether a measurement is for the 
    final visit of a sample in a longitudinal pairing."""
    if value.endswith("v") or value.endswith("v4"):
        return True
    else:
        return False


def find_longitudinal_samples(df):
    """Returns list of longitudinal samples from CMV dataset."""
    final_visits = list(df[df.Sample.apply(find_final_visits)].Sample)
    first_visits = [x.split("v")[0] for x in final_visits]
    return first_visits, final_visits

def split_dataset(df):
    """Splits main CMV dataset into three parts:
    1. Erasmus Samples
    2. Longitudinal Pavia Samples
    3. Non-Longitudinal Pavia Samples

    Datasets are saved in the data/processed folder.
    """
    # get samples by group
    first_visits, final_visits = find_longitudinal_samples(df)
    erasmus_samples = list(df.loc[df.Cohort == "PP_Erasmus", "Sample"])
    # split datasets
    erasmus = df.loc[df.Sample.isin(erasmus_samples)].copy()
    longitudinal = df.loc[df.Sample.isin(first_visits + final_visits)].copy()
    main = df.loc[~df.Sample.isin(erasmus_samples + first_visits + final_visits)].copy()
    train, test = train_test_split(main, test_size=.20, stratify=main['Cohort'])
    # create filenames for saving
    now = datetime.now().strftime("%Y%m%d")
    erasmus_filename = "../data/processed/{}_cmv_erasmus.csv".format(now)
    longitudinal_filename = "../data/processed/{}_cmv_longitudinal.csv".format(now)
    main_filename = "../data/processed/{}_cmv_main.csv".format(now)
    train_filename = "../data/processed/{}_cmv_train.csv".format(now)
    test_filename = "../data/processed/{}_cmv_test.csv".format(now)
    # save datasets
    erasmus.to_csv(erasmus_filename, index=False)
    longitudinal.to_csv(longitudinal_filename, index=False)
    main.to_csv(main_filename, index=False)
    train.to_csv(train_filename, index=False)
    test.to_csv(test_filename, index=False)
    return erasmus_filename, longitudinal_filename, main_filename, train_filename, test_filename

def binarize_cohorts(df):
    """Transforms cohort labels to 'Primary' and 'Latent' only."""
    mapper = {
        "PP": "Primary",
        "NP": "Primary",
        "PL": "Latent",
        "NL": "Latent",
        "PP_Erasmus": "Primary",
    }
    df.Cohort = df.Cohort.map(mapper)
    return df
