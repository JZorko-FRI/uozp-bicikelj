import numpy as np
import pandas as pd

import processing as utils
import features
import constants

def split(df : pd.DataFrame):
    """
    Split initial dataset where each column is a station
    into a dictionary of datasets of one station each.
    """
    # Don't modify original dataframe
    df = df.copy()
    # Construct dictionary of dataframes
    by_station = []
    for col in df.columns:
        if col != constants.TIMESTAMP:
            # Extract column and rename it to target
            df_station = df.filter([constants.TIMESTAMP, col])
            df_station.rename(columns={col: constants.TARGET}, inplace=True)
            # Add column with station name so metadata can be added
            df_station[constants.STATION] = col
            # Add dataframe to dictionary
            by_station.append(df_station)
    # Concatenate all dataframes into one
    return pd.concat(by_station, ignore_index=True)

def combine(df : pd.DataFrame):
    """
    Undo split of dataset where each column is a station
    into a single dataframe with stations as columns.
    """
    return df.pivot(index=constants.TIMESTAMP, columns=constants.STATION, values=constants.TARGET)

def add_metadata(df : pd.DataFrame, df_metadata : pd.DataFrame):
    """
    Add metadata to dataframe.
    """
    df = df.merge(df_metadata, on=constants.STATION)
    return df

def add_weather_data(df : pd.DataFrame, df_weather : pd.DataFrame):
    """
    Add weather data to dataframe.
    """
    df['date'] = df[constants.TIMESTAMP].dt.strftime('%Y-%m-%d')
    return df.merge(df_weather, left_on='date', right_on='valid')

def add_distance_data(df : pd.DataFrame, df_distances : pd.DataFrame):
    """
    Add distance data to dataframe.
    Includes distance to n nearest stations.
    This is done here to avoid passing around all 80+ columns.
    """
    # Extract n smallest values from each row
    n = constants.N_NEAREST
    def extract_n_smallest(row):
        # Ignore first column (station name)
        # Ignore first value (distance to self)
        smallest = row.values[np.argsort(row.values[1:])[1:n+1]]
        for i in range(n):
            row['Nearest_' + str(i)] = smallest[i]
        return row
    df_nearest = df_distances.apply(extract_n_smallest, axis=1)
    # Add distance data to dataframe
    df_nearest = df_distances.filter(['Nearest_' + str(i) for i in range(n)] + ['Station Name'])
    return df.merge(df_nearest, left_on=constants.STATION, right_on='Station Name')

# Preprocessing
def preprocess_test(df : pd.DataFrame, df_metadata : pd.DataFrame, df_weather, df_distances):
    # Add metadata
    df = add_metadata(df, df_metadata)
    # Add weather data
    # df = add_weather_data(df, df_weather)
    # Add distance data
    # df = add_distance_data(df, df_distances)

    # Construct this feature in preprocessing so it can be used for splitting
    df['DayOfWeek'] = features.get_day_of_week(df)
    df['Month'] = features.get_month(df)
    
    # Precompute some features so other features can be efficiently computed from them
    df['TimeOfDay'] = utils.get_time_of_day(df[constants.TIMESTAMP])
    df['TimeOfDayMin'] = utils.get_time_of_day_minutes(df[constants.TIMESTAMP])
    df['TimeOfDay5Min'] = df['TimeOfDayMin'] // (5 * 60)
    df['TimeOfDay10Min'] = df['TimeOfDayMin'] // (10 * 60)
    df['TimeOfDayQuarterHour'] = df['TimeOfDayMin'] // (15 * 60)
    df['TimeOfDayHour'] = df[constants.TIMESTAMP].dt.hour
    
    return df

def preprocess_train(df : pd.DataFrame, df_metadata : pd.DataFrame, df_weather, df_distances):
    # Repeat all preprocessing done on test data
    df = preprocess_test(df, df_metadata, df_weather, df_distances)
    return df

# Code from utils.py

import numpy as np
import pandas as pd

# Offset time of day by this many minutes
TIME_OFFSET_MIN = 0

def date_to_number(column):
    """
    Convert datetime column to seconds.
    """
    return column.astype(np.int64) // 10 ** 9

def number_to_date(column):
    """
    Convert seconds column to datetime.
    """
    return pd.to_datetime(column * 10 ** 9)

def number_to_timedelta(column):
    """
    Convert seconds column to timedelta.
    """
    return pd.to_timedelta(column * 10 ** 9)

def get_time_of_day(column):
    """
    Extract time of day in seconds from datetime column.
    """
    hour = column.dt.hour
    minute = column.dt.minute
    second = column.dt.second
    return hour * 3600 + minute * 60 + second - TIME_OFFSET_MIN * 60

def get_time_of_day_minutes(column):
    """
    Extract time of day in minutes from datetime column.
    """
    hour = column.dt.hour
    minute = column.dt.minute
    return hour * 60 + minute - TIME_OFFSET_MIN

def MAE(y_true, y_pred):
    """
    Calculates the mean absolute error.
    """
    mean = np.mean(np.abs(y_true - y_pred))
    # Sometimes return type is array
    if hasattr(mean, 'shape') and mean.shape != ():
        mean = mean[0]
    return round(mean, 3)

def construct_query(values, columns):
    """
    Construct query for selecting data from a split.
    """
    conditions = []
    for value, column in zip(values, columns):
        conditions.append(f'`{column}` == {repr(value)}')
    return ' & '.join(conditions)
