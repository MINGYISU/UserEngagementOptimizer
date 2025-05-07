import numpy as np
import pandas as pd
import os

TIME_COLS = ["client_upload_time", "event_time", "processed_time","server_received_time", "server_upload_time"]

MANY_UNIQUE_VALS = ["event_type", "user_properties"]

FEW_UNIQUE_VALS = ["data", "device_type", "language", "os_name", "os_version", "country"]

features = ['event_properties', 'user_properties', 'device_family', 'device_type', 'language', 'os_name', 'city', 'country', 'region']
out = ['event_type']

def get_feature_mapping(df, features):
    feature_mapping = {}
    for feature in features:
        unique_values = df[feature].astype(str).unique().tolist()
        for val in unique_values:
            sub_feature_mapping = {'__default__': 0}
            if val not in sub_feature_mapping:
                sub_feature_mapping[val] = len(feature_mapping)
        feature_mapping[feature] = sub_feature_mapping
    return feature_mapping

def get_mapping(row: pd.Series, mapping: dict):
    for feature in features:
        # if the feature needs to be mapped
        if feature in mapping:
            row[feature] = mapping[feature][row[feature]]
    return row

def count_missing(df: pd.DataFrame):
    missing_val_count = df.isnull().sum()
    total_cells = np.product(df.shape)
    total_missing = missing_val_count.sum()
    percentage_missing = (total_missing / total_cells) * 100
    return percentage_missing

def load_data_from_source(data_source_path='2025_csv'):
    directory = list(filter(lambda x: x.endswith(".csv"), os.listdir(data_source_path)))
    datasets = []
    for file_name in directory:
        file_path = os.path.join(data_source_path, file_name)
        temp_df = pd.read_csv(file_path)
        datasets.append(temp_df)
    df = pd.concat(datasets, ignore_index=True)
    if count_missing(df) < 10:
        return df.dropna()
    else:
        raise ValueError("Too many missing values")

def interpret_time(df: pd.DataFrame):
    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S.%f')
    return df

def rank_session_by_time(df: pd.DataFrame):
    session_durations = df.groupby('session_id')['event_time'].agg(lambda x: (max(x) - min(x)).total_seconds())
    session_durations = session_durations.to_frame(name='duration_seconds')
    session_durations = session_durations.sort_values('duration_seconds', ascending=False)
    return session_durations.reset_index()

def removed_zero_duration_sessions(df: pd.DataFrame):
    session_durations = df.groupby('session_id')['event_time'].agg(lambda x: (max(x) - min(x)).total_seconds())
    zero_duration_sessions = session_durations[session_durations == 0].index
    df = df[~df['session_id'].isin(zero_duration_sessions)]
    return df.reset_index(drop=True)

def split_event_type(df: pd.DataFrame):
    df['event_type'] = df['event_type'].map(lambda x: x.split(':'))
    df['event_type'] = df['event_type'].map(lambda lst: set(filter(None, lst)))
    return df

def convert_to_dict(df: pd.DataFrame, column):
    import ast
    df[column] = df[column].map(ast.literal_eval)
    df[column] = df[column].map(lambda x: {k: v for k, v in x.items() if 'ID' not in k and v != 'EMPTY'})
    df[column] = df[column].map(lambda x: x.values())
    def flatten_values(d):
        values = list(d)
        flattened = []
        for item in values:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened
    df[column] = df[column].map(flatten_values)
    df[column] = df[column].map(lambda x: set(x))
    return df

def get_df():
    df = load_data_from_source()
    df = interpret_time(df)
    df = rank_session_by_time(df)
    df = removed_zero_duration_sessions(df)
    df = split_event_type(df)
    df = convert_to_dict(df, 'event_properties')
    df = convert_to_dict(df, 'user_properties')
    return df

if __name__ == "__main__":
    df = get_df()
    print('Data Size: ' + str(df.shape))
    print(df.info())
    print(df.describe())
    print(count_missing(df))
    print(rank_session_by_time(df))