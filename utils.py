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
    if mean.shape != ():
        mean = mean[0]
    return round(mean, 3)
