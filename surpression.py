# Load the data and libraries
import pandas as pd
import numpy as np

TAXI_DATASET = pd.read_csv('./preprocessed_trip_data.csv')

def is_k_anonymous(k, qis, df):
    for index, row in df.iterrows():
        query = ' & '.join([f'`{col}` == "{row[col]}"' for col in qis])
        rows = df.query(query)
        if (rows.shape[0] < k):
            return False
    return True

def is_k_anonymous_fast(k, qis, df):
    ret_false = False
    for group in df.groupby(qis):
        if len(group[1]) < k:
            print(group[0], "has only", len(group[1]), "rows")
            ret_false = True
    if ret_false:
        return False
    else:
        return True

is_k_anonymous_fast(4, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], df)

df2 = TAXI_DATASET.loc[:,['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]

def suppress_count(k, qis, df):
    count = 0
    for group in df.groupby(qis):
        if len(group[1]) < k:
            count += len(group[1])
    return count

assert suppress_count(4, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], df2) == 32561

