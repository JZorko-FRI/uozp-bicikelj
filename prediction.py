import gc
import multiprocessing as mp

import numpy as np
import pandas as pd

import features
import utils
import constants

# TODO check if needed
import scipy.sparse as sp

# TODO route handling currently not compatible with data

def _predict_route(df_train_split, df_test_split, lr, predictions,
                   split, test_mask, labeled, verbose, i, total_splits):
    """
    Perform prediction for a single split.
    """
    # Clear predictions because some workers do multiple splits
    predictions[constants.TARGET] = 0
    df_train_extended = features.construct_features(df_train_split, df_train_split)
    X_train = features.select_features(df_train_extended)
    # TODO see if sparse matrix is needed
    X_train_sp = sp.csr_matrix(X_train)
    # Prepare train features
    y_train = features.select_target(df_train_extended)
    # Construct model
    model = lr.fit(X_train.values, y_train.values)
    # Prepare test features
    df_test_extended = features.construct_features(df_test_split, df_train_split)
    df_test_features = features.select_features(df_test_extended)
    # Predict
    # y_pred = df_test_features.apply(lambda row: model.predict(np.array(row)), axis=1)
    y_pred = model.predict(df_test_features.values)
    # Store predictions
    predictions.loc[test_mask, constants.TARGET] = y_pred.tolist()
    # Perform evaluation if working with labeled data
    if labeled and verbose:
        # Prepare test features
        y_true = features.select_target(df_test_extended)
        # Evaluate separately
        mae = utils.MAE(y_true, y_pred)
        # If verbose is float then treat it as threshold for printing
        if verbose is True or (isinstance(verbose, float) and mae > verbose):
            # Need to print all at once to avoid race conditions
            to_print = []
            to_print.append(f'{split} ({i}/{total_splits}):')
            to_print.append(f'\tMAE: {mae}')
            to_print.append(f'\tNum samples: {df_train_split.shape[0]}')
            print('\n'.join(to_print))
    elif verbose:
        print(f'({i:03}/{total_splits}) ', end='\r')
    gc.collect()
    return predictions[constants.TARGET]

def _predict_separate(df_train, df_test, lr, split, labeled, verbose, parralel):
    """
    Perform prediction for each route separately.
    """
    print('Splitting by:', split)
    # Prepare prediction dataset
    predictions = df_test.filter([constants.TIMESTAMP, constants.TARGET]).copy()
    predictions[constants.TARGET] = 0
    combinations = df_train.groupby(split).count().filter(split).reset_index()
    total_splits = len(combinations)

    # Prepare arguments for each route
    args = []
    for i, split in combinations.iterrows():
        # Prepare query for exctacting current split
        query = utils.construct_query(split, split.index)
        
        # We need the test mask later, so we store it here
        df_test_split_mask = df_test.eval(query)
        df_test_split = df_test[df_test_split_mask]
        if df_test_split.shape[0] == 0:
            # If there are no samples for this split, skip it
            continue

        # Train can be queried directly
        df_train_split = df_train.query(query)
        # Append to args
        args.append((df_train_split, df_test_split, lr, predictions, query, 
                     df_test_split_mask,
                     labeled, verbose, i, total_splits))

    if isinstance(verbose, float):
        print(f'Printing splits with MAE above: {verbose}')

    # Run predictions for all args, either in parallel or sequentially
    if parralel:
        # Parallel
        with mp.Pool() as pool:
            separate_predictions = pool.starmap(_predict_route, args)
    else:
        # Sequential
        separate_predictions = [_predict_route(*arg) for arg in args]

    print(f'Prediction complete, processed {total_splits} splits')

    # Join predections from each route
    df_separate_predictions = pd.concat(separate_predictions, axis=1)
    return df_separate_predictions.sum(axis=1)

def _predict_together(df_train, df_test, lr, labeled, verbose):
    """
    Perform prediction for all routes together.
    """
    # Prepare prediction dataset
    predictions = df_test.filter([constants.TIMESTAMP]).copy()
    # Construct features
    df_train_extended = features.construct_features(df_train, df_train)
    X_train = features.select_features(df_train_extended)
    X_train_sp = sp.csr_matrix(X_train)
    # Prepare train features
    y_train = features.select_features(df_train_extended)
    # Train model
    if verbose:
        print('Training model...')
    model = lr.fit(X_train_sp, y_train)
    if verbose:
        print('Model trained, starting prediction...')
    # Prepare test features
    df_test_extended = features.construct_features(df_test, df_train)
    df_test_features = features.select_features(df_test_extended)
    # Predict
    predictions[constants.TARGET] = model.predict(df_test_features)
    if verbose:
        print('Prediction complete')
    return predictions[constants.TARGET]

def predict(df_train, df_test, lr, *, split=None, labeled=False,
            verbose=False, parralel=True):
    """
    Returns the predictions for the test data based on the train data.
    Uses features returned from construct_features to predict features in select_features.

    Parameters
    ----------
    df_train : pandas.DataFrame
        The train data.
    df_test : pandas.DataFrame
        The test data.
    lr : PredictionModel
        Wrapper of the used prediction model.
    split : str, list[str], optional
        Columns to split the data on and use separate models for each split.
    labeled : bool, optional
        Whether the data is labeled and can be used for evaluation.
    verbose : bool, int, optional
        Whether to print the intermediate results and the treshold for printed route MAE.
    parralel : bool, optional
        Whether to use parralel processing.

    Returns
    -------
    pandas.Series
        The predictions for the test data.
    """
    # Basic checks
    if df_train.shape[0] == 0 or df_test.shape[0] == 0:
        print('No test or train data, aborting')
        return None

    # Either split by route or train on all data
    if split:
        predictions = _predict_separate(df_train, df_test, lr, split, labeled, verbose, parralel)
    else:
        predictions = _predict_together(df_train, df_test, lr, labeled, verbose)

    # Round predictions to three decimals
    predictions = predictions.round(3)

    # Replace negative predictions with 0
    predictions[predictions < 0] = 0

    # Evaluate end result
    if labeled:
        y_true = features.select_target(df_test)
        if (y_true[y_true < 0]).sum() > 0:
            print('Warning: negative predictions')
        print('Overall MAE:', utils.MAE(y_true, predictions))
    
    # Required output format is microseconds
    return predictions

class PredictionModel():
    """
    Class for standardising usage of different prediction models.
    """
    def __init__(self, model, split=None, parallel=True) -> None:
        self.model = model
        if split and type(split) is not list:
            raise ValueError('Parameter split must be a list of strings')
        self.split = split
        self.parallel = parallel

    def __call__(self, df_train, df_test, labeled=False, verbose=False):
        return predict(df_train, df_test, self.model,
                       split=self.split, labeled=labeled,
                       verbose=verbose, parralel=self.parallel)
