import numpy as np
import pandas as pd

import utils
import features
import constants

def split(df):
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
            df_station[constants.STATION] = pd.Categorical(col)
            # Add dataframe to dictionary
            by_station.append(df_station)
    # Concatenate all dataframes into one
    return pd.concat(by_station, ignore_index=True)

def combine(df):
    """
    Undo split of dataset where each column is a station
    into a single dataframe with stations as columns.
    """
    pivot = df.pivot(index=constants.TIMESTAMP, columns=constants.STATION, values=constants.TARGET)
    return pivot.set_index('timestamp')

def add_metadata(df, df_metadata):
    """
    Add metadata to dataframe.
    """
    df = df.merge(df_metadata, on=constants.STATION)
    return df

# Preprocessing
def preprocess_test(df, df_metadata):
    # Add metadata
    df = add_metadata(df, df_metadata)

    return df

    # # Set columns with repeating values to categorical
    # # Convert departure time strings to datetime
    # df['Departure time'] = pd.to_datetime(df['Departure time'])
    # # Compute time of day in seconds for easier computation
    # df['Departure TimeOfDay'] = get_time_of_day(df['Departure time'])
    # df['Departure TimeOfDayMin'] = get_time_of_day_minutes(df['Departure time'])
    # df['Departure hour'] = df['Departure time'].dt.hour
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

def preprocess_train(df, df_metadata):
    # Repeat all preprocessing done on test data
    df = preprocess_test(df, df_metadata)
    return df
    
    
    # # Convert arrival time strings to datetime
    # df['Arrival time'] = pd.to_datetime(df['Arrival time'])
    # # Compute arrival time of day
    # df['Arrival TimeOfDay'] = get_time_of_day(df['Arrival time'])
    # # Compute travel duration
    # df['Duration'] = (df['Arrival TimeOfDay'] - df['Departure TimeOfDay']) % (86400)
