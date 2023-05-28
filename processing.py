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

# Preprocessing
def preprocess_test(df : pd.DataFrame, df_metadata : pd.DataFrame, df_weather):
    # Add metadata
    df = add_metadata(df, df_metadata)
    # Add weather data
    df = add_weather_data(df, df_weather)

    # Construct this feature in preprocessing so it can be used for splitting
    df['DayOfWeek'] = features.get_day_of_week(df)
    
    # Precompute some features so other features can be efficiently computed from them
    df['TimeOfDay'] = utils.get_time_of_day(df[constants.TIMESTAMP])
    df['TimeOfDayMin'] = utils.get_time_of_day_minutes(df[constants.TIMESTAMP])
    df['TimeOfDay5Min'] = df['TimeOfDayMin'] // (5 * 60)
    df['TimeOfDayQuarterHour'] = df['TimeOfDayMin'] // (15 * 60)
    df['TimeOfDayHour'] = df[constants.TIMESTAMP].dt.hour
    
    return df

    # # Set columns with repeating values to categorical
    # # Compute time of day in seconds for easier computation
    # df['Departure TimeOfDayMin'] = get_time_of_day_minutes(df[constants.TIMESTAMP])
    # df['Departure hour'] = df[constants.TIMESTAMP].dt.hour
    # df['Departure quarterhour'] = df['Departure TimeOfDay'] // (15 * 60)
    # df['Departure 5min'] = df['Departure TimeOfDay'] // (5 * 60)
    # df['Departure 1min'] = df['Departure TimeOfDay'] // (60)
    # # Construct column that identifies routes
    # df['RouteID'] = df[ID_COLS[0]].astype(str).str.strip()
    # for column in ID_COLS[1:]:
    #     df['RouteID'] += ";" + df[column].astype(str).str.strip()
    # month = df['Departure time'].dt.month
    # df['Season'] = month.apply(lambda month: (month - 3) // 3 % 4)
    # # Convert column types to conserve memory
    # for column in CATEGORICAL_COLS:
    #     df[column] = df[column].astype('category')
    # return df

def preprocess_train(df : pd.DataFrame, df_metadata : pd.DataFrame, df_weather):
    # Repeat all preprocessing done on test data
    df = preprocess_test(df, df_metadata, df_weather)
    return df
    
    
    # # Convert arrival time strings to datetime
    # df['Arrival time'] = pd.to_datetime(df['Arrival time'])
    # # Compute arrival time of day
    # df['Arrival TimeOfDay'] = get_time_of_day(df['Arrival time'])
    # # Compute travel duration
    # df['Duration'] = (df['Arrival TimeOfDay'] - df['Departure TimeOfDay']) % (86400)
