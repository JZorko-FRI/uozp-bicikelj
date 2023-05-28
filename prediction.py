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

def _predict_route(df_train_route, df_test_route, lr, predictions,
                   route, labeled, verbose, i, total_routes):
    """
    Perform prediction for a single route.
    """
    # Clear predictions because some workers do multiple routes
    predictions[constants.TARGET] = 0
    df_train_extended = features.construct_features(df_train_route, df_train_route)
    X_train = features.select_features(df_train_extended)
    X_train_sp = sp.csr_matrix(X_train)
    # Prepare train features
    y_train = features.select_features(df_train_extended)
    # Construct model
    model = lr.fit(X_train_sp, y_train)
    # Prepare test features
    df_test_extended = features.construct_features(df_test_route, df_train_route)
    df_test_features = features.select_features(df_test_extended)
    # Predict
    y_pred = df_test_features.apply(lambda row: model.predict(np.array(row).reshape(1, -1)), axis=1)
    # Store predictions
    predictions.loc[predictions['RouteID'] == route, constants.TARGET] = y_pred.tolist()
    # Perform evaluation if working with labeled data
    if labeled and verbose:
        # Prepare test features
        y_true = features.select_features(df_test_extended)
        # Evaluate separately
        mae = utils.MAE(y_true, y_pred)
        if verbose is True or (not isinstance(verbose, bool) and mae > verbose):
            print(route, f'({i}/{total_routes}):')
            print(f'\tMAE: {mae}')
            print(f'\tNum samples: {df_train_route.shape[0]}')
    elif verbose:
        print(f'({i:03}/{total_routes}) ', end='\r')
    gc.collect()
    return predictions[constants.TARGET]

def _predict_separate(df_train, df_test, lr, *, labeled=False, verbose=False, parralel=True):
    """
    Perform prediction for each route separately.
    """
    # Prepare prediction dataset
    predictions = df_test.filter(['Departure time', 'RouteID']).copy()
    predictions[constants.TARGET] = 0
    test_unique_directions = df_test['RouteID'].unique()
    train_unique_directions = df_train['RouteID'].unique()
    train_unique_routes = df_train['Route'].unique()
    total_routes = len(test_unique_directions)

    # Prepare arguments for each route
    args = []
    for i, route in enumerate(test_unique_directions):
        # Prepare train features with data of the route
        if route in train_unique_directions:
            df_train_route = df_train[df_train['RouteID'] == route]
        else:
            # If RouteID not in training data, try to use route without direction as training
            route_without_direction = df_test.loc[df_test['RouteID'] == route, 'Route'].iloc[0] or False
            if route_without_direction and route_without_direction in train_unique_routes:
                df_train_route = df_train[df_train['Route'] == route_without_direction]
                if verbose:
                    print(f'Direction {route} not available in training data, using route {route_without_direction}')
            else:
                # If route also not available, use random subset
                df_train_route = df_train.sample(frac=(3 / len(train_unique_directions)),
                                                 random_state=42)
                if verbose:
                    print(f'Direction {route} and its route not available in training data, using random subset')
        df_test_route = df_test[df_test['RouteID'] == route]
        args.append((df_train_route, df_test_route, lr, predictions, route,
                     labeled, verbose, i, total_routes))

    if not isinstance(verbose, bool):
        print(f'Printing routes with MAE above: {verbose}')

    # Run predictions for all args, either in parallel or sequentially
    if parralel:
        # Parallel
        with mp.Pool() as pool:
            separate_predictions = pool.starmap(_predict_route, args)
    else:
        # Sequential
        separate_predictions = [_predict_route(*arg) for arg in args]

    print(f'Prediction complete, processed {total_routes} routes')

    # Join predections from each route
    df_separate_predictions = pd.concat(separate_predictions, axis=1)
    return df_separate_predictions.sum(axis=1)

def _predict_together(df_train, df_test, lr, labeled=False, verbose=False):
    """
    Perform prediction for all routes together.
    """
    # Prepare prediction dataset
    predictions = df_test.filter(['Departure time']).copy()
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

def predict(df_train, df_test, lr, *, split=False, labeled=False,
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
        predictions = _predict_separate(df_train, df_test, lr, labeled=labeled, verbose=verbose, parralel=parralel)
    else:
        predictions = _predict_together(df_train, df_test, lr, labeled=labeled, verbose=verbose)

    # TODO evaluate
    # # Do corrections using median value
    # median = np.median(predictions)

    # difference = np.abs(predictions - median)
    # # Take mean if difference is above threshold, or if prediction is negative
    # predictions[(difference > 2000) | (predictions < 0)] = median

    # Evaluate end result
    if labeled:
        y_true = features.select_features(df_test)
        print(y_true[y_true < 0])
        print('Overall MAE:', utils.MAE(y_true, predictions))

    # Return arrival time computed from predicted duration
    final_predictions = df_test['Departure time'] + utils.number_to_timedelta(predictions)
    
    # Required output format is microseconds
    return final_predictions.astype('datetime64[μs]')

class PredictionModel():
    """
    Class for standardising usage of different prediction models.
    """
    def __init__(self, model, split=constants.STATION) -> None:
        self.model = model
        if split and type(split) not in (str, list[str], None):
            raise ValueError('Parameter split must be a string or list of strings')
        self.split = split

    def __call__(self, df_train, df_test, labeled=False, verbose=False, parallel=True):
        return predict(df_train, df_test, self.model,
                       split=self.split, labeled=labeled,
                       verbose=verbose, parralel=parallel)
