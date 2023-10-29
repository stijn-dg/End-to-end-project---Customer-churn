import pickle
import re
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit)
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake case.

    Parameters
    ----------
    name : str
        The string to convert.

    Returns
    -------
    str
        The converted string.
    """
    name = name.replace(' ', '_')
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower().strip()


def build_column_transformer_for_df(train_x: pd.DataFrame) -> ColumnTransformer:
    """Builds a column transformer for a pandas dataframe."""
    # Get the categorical and numerical columns
    categorical_columns = train_x.select_dtypes(
        include='object').columns.to_list()
    numerical_columns = train_x.select_dtypes(
        include='number').columns.to_list()

    num_prep = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_prep = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ('num', num_prep, numerical_columns),
        ('cat', cat_prep, categorical_columns)
    ])

    return transformer


def build_sklearn_pipeline(df: pd.DataFrame, y_col_name: str, model_name: str, model: object, transformer: ColumnTransformer = None) -> Pipeline:
    """Builds a sklearn pipeline for churn prediction."""
    # Define the steps
    if transformer == None:
        transformer = build_column_transformer_for_df(
            df.drop(y_col_name, axis=1))

    steps = [
        ('preprocessor', transformer),
        ('under', RandomUnderSampler()),
        ('over', SMOTE(k_neighbors=5)),
        ('pca', PCA()),
        (model_name, model)
    ]
    # Create the pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


def sklearn_gridsearch_using_pipeline(
        train: pd.DataFrame, y_col_name: str,
        model_name: str, model: object,
        fit_le: LabelEncoder, param_grid: dict,
        verbose: int, randomized: bool = False,
        n_folds: int = 5, pipeline: Pipeline = None) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Performs a (randomized) grid search using a sklearn pipeline."""
    # Get the pipeline
    if pipeline == None:
        pipeline = build_sklearn_pipeline(
            train, y_col_name=y_col_name, model=model, model_name=model_name)

    # define stratiefied shuffle split:
    sss = StratifiedShuffleSplit(
        n_splits=n_folds, test_size=0.2, random_state=0)

    # Define the hyperparameter grid
    default_pca_n_components = [15, 20, 25, 30, 35, 50, 65]
    # example from fin_churn
    default_undesampling_rates = [0.3, 0.5, 0.7, 1]
    default_oversampling_rates = [0.3, 0.5, 0.7, 1]
    param_grid = param_grid
    default_param_grid = {
        "pca__n_components": default_pca_n_components,
        "under__sampling_strategy": default_undesampling_rates,
        "over__sampling_strategy": default_oversampling_rates
    }

    for param in default_param_grid.keys():
        if param not in param_grid.keys():
            param_grid[param] = default_param_grid[param]

    # Perform the grid search
    common_params = {
        'estimator': pipeline,
        'scoring': "roc_auc",
        'verbose': verbose,
        'n_jobs': -1,
        'cv': sss
    }

    if randomized:
        common_params["param_distributions"] = param_grid
        common_params["n_iter"] = 10
        common_params["random_state"] = 0
        grid = RandomizedSearchCV(**common_params)
    else:
        common_params["param_grid"] = param_grid
        grid = GridSearchCV(**common_params)

    encoded_labels = fit_le.transform(train[y_col_name])
    grid.fit(train.drop(y_col_name, axis=1), encoded_labels)
    # Print the results
    print('Best score:', grid.best_score_)
    print('Best parameters:', grid.best_params_)

    return grid


def evaluate_model(
        best_pipeline: Pipeline,
        fit_le: LabelEncoder,
        test: pd.DataFrame,
        y_col_name: str,
        model_name: str) -> None:
    """
    Evaluates a model using a test set.

    Parameters
    ----------
    best_pipeline : Pipeline
        The best pipeline found by the grid search.
    fit_le : LabelEncoder
        The label encoder fitted on the training set.
    test : pd.DataFrame 
        The test set.
    y_col_name : str
        The name of the target column.
    """
    clf = best_pipeline[model_name]

    test_predictions = best_pipeline.predict(
        test.drop(y_col_name, axis=1))
    test_predictions_proba = best_pipeline.predict_proba(
        test.drop(y_col_name, axis=1))

    test_y_encoded = fit_le.transform(test[y_col_name])
    decoded_labels = fit_le.inverse_transform(clf.classes_)
    cm = confusion_matrix(
        test_y_encoded, test_predictions, labels=clf.classes_)

    _fig, _ax = plt.subplots(figsize=(7.5, 7.5))
    sn.heatmap(cm, annot=True, fmt="d", xticklabels=decoded_labels,
               yticklabels=decoded_labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

    # only get predictions from the positive class (=churn)
    PrecisionRecallDisplay.from_predictions(
        test_y_encoded, test_predictions_proba[:, 1], pos_label=1)
    plt.show()


def write_pipeline(pipeline: Pipeline, model_name: str, dataset_name: str) -> None:
    """
    Writes the pipeline to a pickle file

    Args:
        pipeline (Pipeline): the pipeline to be written
        model_name (str): the name of the model
        dataset_name (str): the name of the dataset

    """
    pipeline_base_dir = f"../models/{dataset_name}/"
    Path(pipeline_base_dir).mkdir(parents=True, exist_ok=True)
    file = f"{model_name}.pkl"
    pipeline_path = pipeline_base_dir + file
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline(model_name: str, dataset_name: str) -> Pipeline:
    """
    Loads the pipeline from a pickle file

    Args:
        model_name (str): the name of the model
        dataset_name (str): the name of the dataset

    Returns:
        Pipeline: the loaded pipeline
    """
    pipeline_base_dir = f"../models/{dataset_name}/"
    file = f"{model_name}.pkl"
    pipeline_path = pipeline_base_dir + file
    with open(pipeline_path, 'rb') as f:
        best_pipeline = pickle.load(f)
    return best_pipeline


def add_eap_ep(train:pd.DataFrame,test:pd.DataFrame, y_col_name:str, best_pipeline_log_reg:Pipeline, cb_column: str) -> pd.DataFrame:
    """
    Adds the EAP column to the test set.
    cb_column is de aggregated cost benefit, benefit - cost
    
    Parameters

    


    """


    impute = SimpleImputer(strategy='median')
    test[cb_column] = impute.fit_transform(
        test[cb_column].to_frame())[:, 0]
    test['prob_churn'] = best_pipeline_log_reg.predict_proba(test.drop(y_col_name, axis=1))[:, 1]
   

    
    counts = train[y_col_name].value_counts()
    estimated_p_1 = counts.loc[1] / counts.sum()
    estimated_p_0 = counts.loc[0] / counts.sum()


    test_actual_label_0 = test.loc[test[y_col_name] == 0]
    test_actual_label_0['EAP']= test_actual_label_0['prob_churn']*test_actual_label_0['FP']+(1-test_actual_label_0['prob_churn'])*test_actual_label_0['TN']
    test_actual_label_0['EP'] = estimated_p_1*test_actual_label_0['FP']+estimated_p_0*test_actual_label_0['TN']
    test_actual_label_1 = test.loc[test[y_col_name] == 1]
    test_actual_label_1['EAP']= test_actual_label_1['prob_churn']*test_actual_label_1['TP']+(1-test_actual_label_1['prob_churn'])*test_actual_label_1['FN']
    test_actual_label_1['EP']= estimated_p_1 *test_actual_label_1['TP']+ estimated_p_0 *test_actual_label_1['FN']
    return pd.concat([test_actual_label_0, test_actual_label_1])
