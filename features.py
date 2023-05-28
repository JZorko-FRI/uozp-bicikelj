import pandas as pd
import numpy as np

import constants

GENERAL_HOLIDAYS = {
    (1, 1), (2, 1), (8, 2), (8, 4), (9, 4), (27, 4), (1, 5), (2, 5),
    (25, 6), (15, 8), (31, 10), (1, 11), (25, 12), (26, 12),
}

SCHOOL_HOLIDAYS = {
    (20, 2), (21, 2), (22, 2), (23, 2), (24, 2),
    (30, 4), (29, 10), (30, 10), (2, 11), (24, 12),
    (27, 12), (28, 12), (29, 12), (30, 12), (31, 12)
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

def is_school_holiday(df, df_train=None):
    """
    Returns 1 if the departure date is a school holiday, 0 otherwise.
    """
    other_holidays = df[constants.TIMESTAMP].map(lambda time: (time.month, time.day) in SCHOOL_HOLIDAYS)
    summer_holidays = df[constants.TIMESTAMP].dt.month.isin([7, 8, 9])
    return (other_holidays | summer_holidays).astype(int)

def is_school_day(df, df_train=None):
    """
    Returns 1 if the departure date is a school day, 0 otherwise.
    """
    weekends = is_weekend(df)
    school_holidays = is_school_holiday(df)
    general_holidays = is_general_holiday(df)
    holidays = school_holidays | general_holidays
    return (~weekends & ~holidays).astype(int)

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
    columns = list(range(1, 2))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, power in enumerate(columns):
        matrix[:, i ] = np.power(timeOfDay, power)
    return pd.DataFrame(matrix, columns=columns)

def get_time_of_day_min_poly(df, df_train=None):
    """
    Returns the departure time of day in minutes.
    """
    timeOfDay = df['TimeOfDay'] // 60
    num_rows = df.shape[0]
    columns = list(range(1, 2))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, power in enumerate(columns):
        matrix[:, i ] = np.power(timeOfDay, power)
    return pd.DataFrame(matrix, columns=columns)

def get_binary_drivers(df, df_train):
    """
    Returns the binarised drivers.
    """
    num_rows = df.shape[0]
    columns = list(df_train['Driver ID'].unique())
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, driver in enumerate(columns):
        matrix[:, i] = (df['Driver ID'] == driver).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_binary_routes(df, df_train):
    """
    Returns the binarised routes.
    """
    num_rows = df.shape[0]
    columns = list(df_train['Route'].unique())
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, route in enumerate(columns):
        matrix[:, i] = (df['Route'] == route).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_binary_directions(df, df_train):
    """
    Returns the binarised directions.
    """
    num_rows = df.shape[0]
    columns = list(df_train['Route Direction'].unique())
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, route in enumerate(columns):
        matrix[:, i] = (df['Route Direction'] == route).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_routes_average_speed(df, df_train):
    """
    Returns the routes average speeds.
    """
    average_speeds = df_train.groupby('Route').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Route']), average_speeds, on='Route', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_routeid_average_speed(df, df_train):
    """
    Returns the direction average speeds.
    """
    average_speeds = df_train.groupby('RouteID').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['RouteID']), average_speeds, on='RouteID', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_drivers_average_speed(df, df_train):
    """
    Returns the drivers average speeds.
    """
    average_speeds = df_train.groupby('Driver ID').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Driver ID']), average_speeds, on='Driver ID', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_bus_average_speed(df, df_train):
    """
    Returns the bus average speeds.
    """
    average_speeds = df_train.groupby('Registration').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Registration']), average_speeds,
                         on='Registration', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_hour_average_speed(df, df_train):
    """
    Returns the hour average speeds.
    """
    average_speeds = df_train.groupby('Departure hour').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Departure hour']), average_speeds,
                         on='Departure hour', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_quarterhour_average_speed(df, df_train):
    """
    Returns the hour average speeds.
    """
    average_speeds = df_train.groupby('Departure quarterhour').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Departure quarterhour']), average_speeds,
                         on='Departure quarterhour', how='left')[constants.TARGET]
    durations[durations.isna()] = total_average_speed
    return durations

def get_5min_average_speed(df, df_train):
    """
    Returns the 5min average speeds.
    """
    average_speeds = df_train.groupby('Departure 5min').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Departure 5min']), average_speeds,
                         on='Departure 5min', how='left')[constants.TARGET]
    quarter_average = get_quarterhour_average_speed(df, df_train)
    durations[durations.isna()] = quarter_average[durations.isna()]
    durations[durations.isna()] = total_average_speed
    return durations

def get_1min_average_speed(df, df_train):
    """
    Returns the 5min average speeds.
    """
    average_speeds = df_train.groupby('Departure 1min').aggregate({constants.TARGET: np.mean})
    total_average_speed = average_speeds[constants.TARGET].mean()
    durations = pd.merge(df.filter(['Departure 1min']), average_speeds,
                         on='Departure 1min', how='left')[constants.TARGET]
    five_min_average = get_5min_average_speed(df, df_train)
    durations[durations.isna()] = five_min_average[durations.isna()]
    durations[durations.isna()] = total_average_speed
    return durations

def get_binary_busses(df, df_train):
    """
    Returns the binarised busses.
    """
    num_rows = df.shape[0]
    columns = list(df_train['Registration'].unique())
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, bus in enumerate(columns):
        matrix[:, i] = (df['Registration'] == bus).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_season(df, df_train=None):
    """
    Returns the season of the departure time.
    """
    return df['Season']

def get_binary_season(df, df_train=None):
    """
    Returns the binarised season.
    """
    num_rows = df.shape[0]
    columns = list(range(4))
    num_cols = len(columns)
    matrix = np.zeros((num_rows, num_cols))
    for i, season in enumerate(columns):
        matrix[:, i] = (df['Season'] == season).astype(int)
    return pd.DataFrame(matrix, columns=columns, dtype=int)

def get_traffic_factor(df, df_train=None):
    """
    Returns the traffic factor (how much traffic there is) of the departure time.
    """
    time_of_day = df['TimeOfDay']
    morning = ((5 * 3600 < time_of_day) & (time_of_day < 8 * 3600)).astype(float)
    midday = ((8 * 3600 < time_of_day) & (time_of_day < 13 * 3600)).astype(float)
    afternoon = ((13 * 3600 < time_of_day) & (time_of_day < 17 * 3600)).astype(float)
    evening = ((17 * 3600 < time_of_day) & (time_of_day < 21 * 3600)).astype(float)
    # Return weighted sum, weights derived empirically
    return morning * 4 + midday * 0.5 + afternoon * 3 + evening * 1

def get_binary_part_of_day(df, df_train=None):
    """
    Returns the binarised part of the day (morning, midday, afternoon...),
    one column for each part of the day.
    """
    columns = {
        'earlyNight': (0, 6),
        'morning': (6, 9),
        'midday': (9, 15),
        'afternoon': (15, 18),
        'evening': (18, 21),
        'lateNight': (21, 24)
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
        matrix[:, quarter - 1] = (df['Departure quarterhour'] == quarter).astype(int)
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
        matrix[:, mins - 1] = (df['Departure 5min'] == mins).astype(int)
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

features = {
    'Month': get_month,
    'DayOfWeek': get_day_of_week,
    # 'DayOfMonth': get_day_of_month,
    'TimeOfDayMin': get_time_of_day_min_poly,
    'TimeOfDaySec': get_time_of_day_poly,
    # 'GeneralHoliday': is_general_holiday,
    # 'SchoolHoliday': is_school_holiday,
    # 'SchoolDay': is_school_day,
    # 'Weekend': is_weekend,
    # 'Workday': is_workday,
    # 'Worktime': get_worktime,
    # 'TrafficFactor': get_traffic_factor,
    # 'DriverAverageSpeed': get_drivers_average_speed,
    # 'BusAverageSpeed': get_bus_average_speed,
    # 'HourAverageSpeed': get_hour_average_speed,
    # 'QuarterHourAverageSpeed': get_quarterhour_average_speed,
    # '5minAverageSpeed': get_5min_average_speed,
    # '1minAverageSpeed': get_1min_average_speed,
    # 'RouteAverageSeed': get_routes_average_speed,
    # 'DirectionAverageSpeed': get_routeid_average_speed,
}
multi_features = {
    # 'BinaryDayOfWeek': get_binary_days,
    # 'BinaryMonth': get_binary_months,
    # 'BinaryHour': get_binary_hours,
    # 'BinaryQuarterHour': get_binary_quarterhours,
    # 'Binary5min': get_binary_5min,
    # 'BinaryPartOfDay': get_binary_part_of_day,
    # 'TimeOfDayPoly': get_time_of_day_poly,
    # 'BinaryDriver': get_binary_drivers,
    # 'BinaryBus': get_binary_busses,
    # 'BinaryRoute': get_binary_routes,
    # 'BinaryDirection': get_binary_directions,
    # 'BinarySeason': get_binary_season,
    # 'TimeOfDayMin': get_time_of_day_min_poly,
}
combination_features = {
    # 'Binary5minWeek': ['Binary5min', 'BinaryDayOfWeek'],
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
    for feature_name, feature_vals in dfs[0].iteritems():
        if len(dfs) == 2:
            other_features = dfs[1].iteritems()
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