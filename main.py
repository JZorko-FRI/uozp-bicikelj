# Imports
import gc
import multiprocessing as mp
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import processing
import prediction
import constants

np.set_printoptions(suppress=True)

if __name__ == '__main__':

    # Usage
    if len(sys.argv) < 2:
        print('Usage:')
        print('\tPass "evaluate" as argument to evaluate model on train data.')
        print('\tPass "predict" as argument to make predictions on test data.')
        exit(1)

    # Read data
    df_test_raw = pd.read_csv('data/bicikelj_test.csv', sep=',')
    df_train_raw = pd.read_csv('data/bicikelj_train.csv', sep=',')
    df_metadata = pd.read_csv('data/bicikelj_metadata.csv', sep='\t')

    original_column_order = list(df_train_raw.columns)
    original_column_order.remove(constants.TIMESTAMP)

    # Split data by station
    df_test_split = processing.split(df_test_raw)
    df_train_split = processing.split(df_train_raw)

    # Preprocess data and add features
    df_test = processing.preprocess_test(df_test_split, df_metadata)
    df_train = processing.preprocess_train(df_train_split, df_metadata)

    # Set the used prediction method
    model = prediction.PredictionModel(
        Ridge(alpha=1, random_state=42),
        split=[constants.STATION, 'DayOfWeek']
    )
    # model = prediction.PredictionModel(
    #     XGBRegressor(n_estimators=50, nthread=1, random_state=42),
    #     split=[constants.STATION, 'DayOfWeek'], parallel=True
    # )
    # model = prediction.PredictionModel(
    #     MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, n_iter_no_change=10,
    #                  verbose=0, random_state=42),
    #     split=[constants.STATION, 'DayOfWeek']
    # )

    # TODO build separate simple model across all routes, use it to prune outliers

    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    TRAIN_HOURS = {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23}
    EVAL_HOURS = {8, 20, 21}

    # Local evaluation on one month
    if 'evaluate' in sys.argv:
        print('Evaluating on labeled data')
        # Use one month as test data and the rest for training
        df_tested_month = df_train[df_train[constants.TIMESTAMP].dt.hour.isin(EVAL_HOURS)].copy().reset_index()
        df_train_months = df_train[df_train[constants.TIMESTAMP].dt.hour.isin(TRAIN_HOURS)].copy().reset_index()
        model(df_train_months, df_tested_month, labeled=True, verbose=5.0)

    # Predictions on proper test set
    if 'predict' in sys.argv:
        print('Predicting on unlabeled data')
        train_months = df_train[constants.TIMESTAMP].dt.hour.isin(TRAIN_HOURS)
        eval_months = df_train[constants.TIMESTAMP].dt.hour.isin(EVAL_HOURS)
        train_and_eval = train_months | eval_months
        df_train_months = df_train[train_and_eval].copy().reset_index()
        df_test[constants.TARGET] = model(df_train_months, df_test, verbose=True)
        recombined = processing.combine(df_test.filter([constants.TIMESTAMP, constants.STATION, constants.TARGET]))
        recombined = recombined[original_column_order]
        recombined.to_csv('predictions.txt')
