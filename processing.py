import numpy as np
import pandas as pd

import utils
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
    df = add_weather_data(df, df_weather)
    # Add distance data
    df = add_distance_data(df, df_distances)

    # Construct this feature in preprocessing so it can be used for splitting
    df['DayOfWeek'] = features.get_day_of_week(df)
    df['Month'] = features.get_month(df)
    
    # Precompute some features so other features can be efficiently computed from them
    df['TimeOfDay'] = utils.get_time_of_day(df[constants.TIMESTAMP])
    df['TimeOfDayMin'] = utils.get_time_of_day_minutes(df[constants.TIMESTAMP])
    df['TimeOfDay5Min'] = df['TimeOfDayMin'] // (5 * 60)
    df['TimeOfDayQuarterHour'] = df['TimeOfDayMin'] // (15 * 60)
    df['TimeOfDayHour'] = df[constants.TIMESTAMP].dt.hour
    
    return df

def preprocess_train(df : pd.DataFrame, df_metadata : pd.DataFrame, df_weather, df_distances):
    # Repeat all preprocessing done on test data
    df = preprocess_test(df, df_metadata, df_weather, df_distances)
    return df
