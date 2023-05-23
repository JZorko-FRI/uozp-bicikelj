import numpy as np
import pandas as pd

from utils import *

CATEGORICAL_COLS = ['Registration', 'Driver ID', 'Route', 'Route Direction', 'Season',
                    'First station', 'Last station', 'Route description', 'RouteID']
ID_COLS = ['Route Direction', 'First station', 'Last station']

def select_target(df):
    """
    Returns the target column.
    """
    return df['Duration']

# Preprocessing
def preprocess_test(df, df_metadata):

    # TODO transform 


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
    df = preprocess_test(df)
    # Convert arrival time strings to datetime
    df['Arrival time'] = pd.to_datetime(df['Arrival time'])
    # Compute arrival time of day
    df['Arrival TimeOfDay'] = get_time_of_day(df['Arrival time'])
    # Compute travel duration
    df['Duration'] = (df['Arrival TimeOfDay'] - df['Departure TimeOfDay']) % (86400)
    return df
