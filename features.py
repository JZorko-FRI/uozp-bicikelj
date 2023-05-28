import pandas as pd
import numpy as np

import constants

GENERAL_HOLIDAYS = {
    (1, 1), (2, 1), (8, 2), (18, 4), (27, 4), (1, 5), (2, 5),
    (25, 6), (15, 8), (31, 10), (1, 11), (25, 12), (26, 12),
}

# Feature definitions
def get_month(df, df_train=None):
    """
    Returns the numerical month of the departure time.
    """
    return df[constants.TIMESTAMP].dt.month

def is_general_holiday(df, df_train=None):
    """
    Returns 1 if the departure date is a general holiday, 0 otherwise.
    """
    return df[constants.TIMESTAMP].map(lambda time: (time.month, time.day) in GENERAL_HOLIDAYS).astype(int)

def is_weekend(df, df_train=None):
    """
    Returns 1 if the departure date is on weekend, 0 otherwise.
    """
    return df[constants.TIMESTAMP].dt.dayofweek.isin({5, 6}).astype(int)

def is_workday(df, df_train=None):
    """
    Returns 1 if the departure date is on workday, 0 otherwise.
    """
    weekends = is_weekend(df)
    holidays = df[constants.TIMESTAMP].map(lambda time: (time.month, time.day) in GENERAL_HOLIDAYS)
    return (~weekends & ~holidays).astype(int)

def get_day_of_week(df, df_train=None):
    """
    Returns the numerical day of the week.
    """
    return df[constants.TIMESTAMP].dt.dayofweek

def get_day_of_month(df, df_train=None):
    """
    Returns the numerical day of the month.
    """
    return df[constants.TIMESTAMP].dt.day

def get_time_of_day_poly(df, df_train=None):
    """
    Returns the departure time of day in seconds.
    """
    timeOfDay = df['TimeOfDay']
    num_rows = df.shape[0]
    powers = [3]
    num_cols = len(powers)
    matrix = np.zeros((num_rows, num_cols))
    for i, power in enumerate(powers):
        matrix[:, i ] = np.power(timeOfDay, power)
    return pd.DataFrame(matrix, columns=powers)

def get_time_of_day_min_poly(df, df_train=None):
    """
    Returns the departure time of day in minutes.
    """
    timeOfDay = df['TimeOfDayMin']
    num_rows = df.shape[0]
    powers = [3]
    num_cols = len(powers)
    matrix = np.zeros((num_rows, num_cols))
    for i, power in enumerate(powers):
        matrix[:, i ] = np.power(timeOfDay, power)
    return pd.DataFrame(matrix, columns=powers)

def get_routes_average(df, df_train):
    """
    Returns the routes average speeds.
    """
    average_speeds = df_train.groupby(constants.STATION).aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter([constants.STATION]), average_speeds, on=constants.STATION, how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_hour_average(df, df_train):
    """
    Returns the hour average speeds.
    """
    average_speeds = df_train.groupby('TimeOfDayHour').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['TimeOfDayHour']), average_speeds,
                         on='TimeOfDayHour', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_quarterhour_average(df, df_train):
    """
    Returns the hour average speeds.
    """
    average_speeds = df_train.groupby('TimeOfDayQuarterHour').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['TimeOfDayQuarterHour']), average_speeds,
                         on='TimeOfDayQuarterHour', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_5min_average(df, df_train):
    """
    Returns the 5min average speeds.
    """
    average_speeds = df_train.groupby('TimeOfDay5Min').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['TimeOfDay5Min']), average_speeds,
                         on='TimeOfDay5Min', how='left')[constants.TARGET]
    quarter_average = get_quarterhour_average(df, df_train)
    durations[durations.isna()] = quarter_average[durations.isna()]
    durations[durations.isna()] = total_average_speed
    return durations

def get_1min_average(df, df_train):
    """
    Returns the 5min average speeds.
    """
    average_speeds = df_train.groupby('TimeOfDayMin').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['TimeOfDayMin']), average_speeds,
                         on='TimeOfDayMin', how='left')[constants.TARGET]
    five_min_average = get_5min_average(df, df_train)
    durations[durations.isna()] = five_min_average[durations.isna()]
    durations[durations.isna()] = total_average_speed
    return durations

def get_traffic_factor(df, df_train=None):
    """
    Returns the traffic factor (how much traffic there is) of the departure time.
    """
    time_of_day = df['TimeOfDay']
    morning = ((6 * 3600 < time_of_day) & (time_of_day < 8 * 3600)).astype(float)
    midday = ((8 * 3600 < time_of_day) & (time_of_day < 14.5 * 3600)).astype(float)
    rush = ((14.5 * 3600 < time_of_day) & (time_of_day < 15.5 * 3600)).astype(float)
    afternoon = ((15.5 * 3600 < time_of_day) & (time_of_day < 17.5 * 3600)).astype(float)
    evening = ((17.5 * 3600 < time_of_day) & (time_of_day < 21 * 3600)).astype(float)
    # Return weighted sum, weights derived empirically
    return morning * 4 + midday * 1 + rush * 4 + afternoon * 3 + evening * 1

def get_binary_part_of_day(df, df_train=None):
    """
    Returns the binarised part of the day (morning, midday, afternoon...),
    one column for each part of the day.
    """
    columns = {
        'earlyNight': (0, 6),
        'morning': (6, 8),
        'midday': (8, 14.5),
        'rush': (14.5, 15.5),
        'afternoon': (15.5, 17.5),
        'evening': (17.5, 20),
        'lateNight': (20, 24)
    }
    time_of_day = df['TimeOfDay']
    num_rows = df.shape[0]
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, (part, (start, end)) in enumerate(columns.items()):
        matrix[:, i] = ((start * 3600 < time_of_day) & (time_of_day < end * 3600)).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_worktime(df, df_train=None):
    """
    Returns 1 if the departure time is within usual work hours, 0 otherwise.
    """
    time_of_day = df['TimeOfDay']
    worktime = ((4 * 3600 < time_of_day) & (time_of_day < 18 * 3600))
    # Times derived empirically
    return (worktime).astype(int)

def get_binary_hours(df, df_train=None):
    """
    Returns the binarised hours of the day, one column for each hour.
    """
    num_rows = df.shape[0]
    columns = list(range(0, 24))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for hour in columns:
        matrix[:, hour - 1] = (df[constants.TIMESTAMP].dt.hour == hour).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_binary_quarterhours(df, df_train=None):
    """
    Returns the binarised quarterhours of the day, one column for each hour.
    """
    num_rows = df.shape[0]
    columns = list(range(0, 24 * 4))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for quarter in columns:
        matrix[:, quarter - 1] = (df['TimeOfDayQuarterHour'] == quarter).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)


def get_binary_5min(df, df_train=None):
    """
    Returns the binarised 5min increments of the day, one column for each hour.
    """
    num_rows = df.shape[0]
    columns = list(range(0, 24 * 12))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for mins in columns:
        matrix[:, mins - 1] = (df['TimeOfDay5Min'] == mins).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_binary_days(df, df_train=None):
    """
    Returns the binarised representation of the days of the week, one column for each day.
    """
    num_rows = df.shape[0]
    columns = list(range(1, 8))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for day in columns:
        matrix[:, day - 1] = (df[constants.TIMESTAMP].dt.dayofweek == day).astype(int)
    df_out = pd.DataFrame(matrix, columns=columns, dtype=int)
    return df_out

def get_binary_months(df, df_train=None):
    """
    Returns the binarised representation of the months of the week, one column for each month.
    """
    num_rows = df.shape[0]
    columns = list(range(1, 13))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for month in columns:
        matrix[:, month - 1] = (df[constants.TIMESTAMP].dt.month == month).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_60_min_ago(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] + pd.Timedelta(minutes=60)
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))
    return merged[constants.TARGET + '_y']

def get_90_min_ago(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] + pd.Timedelta(minutes=90)
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))
    return merged[constants.TARGET + '_y']

def get_120_min_ago(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] + pd.Timedelta(minutes=120)
    df_train['FoundTimestamp'] = df_train[constants.TIMESTAMP]
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))

    return merged[constants.TARGET + '_y']

def get_8_hours_ago(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] + pd.Timedelta(hours=8)
    df_train['FoundTimestamp'] = df_train[constants.TIMESTAMP]
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))

    return merged[constants.TARGET + '_y']

def get_1_week_ago(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] + pd.Timedelta(days=7)
    df_train['FoundTimestamp'] = df_train[constants.TIMESTAMP]
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))

    return merged[constants.TARGET + '_y']

def get_1_week_ahead(df : pd.DataFrame, df_train=None):
    """
    Returns the amount of bikes on this station 60 minutes ago.
    """
    relevant = [constants.TIMESTAMP, constants.TARGET]
    df = df.copy().filter(relevant)
    df_train = df_train.copy().filter(relevant)
    df_train[constants.TIMESTAMP] = df_train[constants.TIMESTAMP] - pd.Timedelta(days=7)
    df_train['FoundTimestamp'] = df_train[constants.TIMESTAMP]
    merged = pd.merge_asof(df, df_train, on=constants.TIMESTAMP, direction='nearest', suffixes=('', '_y'))

    return merged[constants.TARGET + '_y']

def get_weather_data(df, df_train=False):
    # Available: količina padavin [mm],dež,nevihta,sneg,sodra,poledica,padavine,snežna odeja
    boolean = [
        'dež',
        'nevihta',
        'sneg',
        # 'sodra',
        'poledica',
        # 'padavine', 
        # 'snežna odeja'
    ]
    df_weather = df.filter([
        # 'količina padavin [mm]',
    ] + boolean)
    for col in boolean:
        df_weather[col] = df_weather[col].astype(int)
    return df_weather

def get_distances(df, df_train=False):
    n = constants.N_NEAREST
    selected = ['Nearest_' + str(i) for i in range(n)]
    return df.filter(selected)
                    
features = {
    # 'Month': get_month,
    # 'DayOfWeek': get_day_of_week,
    # 'DayOfMonth': get_day_of_month,
    # 'TimeOfDayPoly': get_time_of_day_poly,
    # 'TimeOfDayMinPoly': get_time_of_day_min_poly,
    'GeneralHoliday': is_general_holiday,
    # 'Weekend': is_weekend,
    # 'Workday': is_workday,
    # 'Worktime': get_worktime,
    'TrafficFactor': get_traffic_factor,
    # 'HourAverage': get_hour_average,
    'QuarterHourAverage': get_quarterhour_average,
    # '5minAverage': get_5min_average,
    # '1minAverage': get_1min_average,
    'RouteAverage': get_routes_average,
    '60MinAgo': get_60_min_ago,
    '90MinAgo': get_90_min_ago,
    '120MinAgo': get_120_min_ago,
    # '8HoursAgo': get_8_hours_ago,
    '1WeekAgo': get_1_week_ago,
    '1WeekAhead': get_1_week_ahead,
}
multi_features = {
    'BinaryDayOfWeek': get_binary_days,
    'BinaryMonth': get_binary_months,
    # 'BinaryHour': get_binary_hours,
    'BinaryQuarterHour': get_binary_quarterhours,
    # 'Binary5min': get_binary_5min,
    'BinaryPartOfDay': get_binary_part_of_day,
    # 'TimeOfDayPoly': get_time_of_day_poly,
    # 'TimeOfDayMin': get_time_of_day_min_poly,
    'Weather': get_weather_data,
    'Distances': get_distances,
}
combination_features = {
    'BinaryQuarterHourWeek': ['BinaryQuarterHour', 'BinaryDayOfWeek'],
}
precomputed_features = {
    # 'TimeOfDay',
    # 'DayOfWeek',
    'total_space',
}


# Feature construction
used_features = set()

def construct_features(df, df_train):
    """
    Construct features on the given dataframe.
    Needs the training dataframe to match vocabulary, as test might not include all values.
    """
    df = df.copy().reset_index(drop=True)
    # Features that add single column
    for name, feature in features.items():
        df[name] = feature(df, df_train)
        used_features.add(name)
    # Features that add multiple columns
    for name, feature in multi_features.items():
        # Get the columns created by the feature
        new_cols = feature(df, df_train).add_prefix(name)
        df = pd.concat([df, new_cols], axis=1)
        used_features.update(new_cols.columns)
    # Features that combine other features
    i = 0
    for name, features_to_combine in combination_features.items():
        # Take existing columns
        feature_dfs = []
        for feature_name in features_to_combine:
            feature_cols = [feature for feature in df.columns if feature.startswith(feature_name)]
            feature_df = df.filter(feature_cols)
            feature_dfs.append(feature_df)
        # Construct all combinations of features
        for combined_feature_name, combined_feature_vals in _feature_combinations(feature_dfs):
            new_name = f'{name}_{combined_feature_name}'
            df[new_name] = combined_feature_vals
            used_features.update(new_name)
            i += 1
            if i % 50 == 0:
                df = df.copy()
    used_features.update(precomputed_features)
    return df

def _feature_combinations(dfs):
    """
    Recursively generate combinations of passed features.
    If we pass a df A with columns 1 and 2, and df B with columns 3 and 4,
    we get the following combinations:
        1_3, 1_4, 2_3, 2_4

    Parameters
    ----------
    dfs : list of pd.DataFrame

    Returns
    -------
    iterable of tuples : (column_name, pd.Series)
    """
    for feature_name, feature_vals in dfs[0].items():
        if len(dfs) == 2:
            other_features = dfs[1].items()
        else:
            other_features = _feature_combinations(dfs[1:])
        for other_feature_name, other_feature_vals in other_features:
            name = f'{feature_name}_{other_feature_name}'
            combined = feature_vals & other_feature_vals
            yield name, combined

def select_features(df):
    """
    Returns the columns used for training.
    """
    return df.filter(items=list(used_features))

def select_target(df):
    """
    Returns the target column.
    """
    return df[constants.TARGET]