# Load the data and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DECIMAL_PRECISION = 6
TAXI_DATASET = pd.read_csv('./preprocessed_trip_data.csv')
EPSILON = 1.0
PICKUP_LOC = [float('-74.004639'), float('40.742039')] # pickup_longitude  pickup_latitude


def calculate_sensitivity(data):
    # Compute the sensitivity as the maximum absolute difference in fare amounts
    max_fare = np.max(data['fare_amount'])
    min_fare = np.min(data['fare_amount'])
    sensitivity = max_fare - min_fare
    return sensitivity

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def relative_error(priv, orig):
    return np.abs(1 - (priv/orig))

# def count_query_pickup(dataset):

def query_data():
    dataset = TAXI_DATASET[(TAXI_DATASET['name'] == 'Anita Ball') & (round(TAXI_DATASET['pickup_longitude'], DECIMAL_PRECISION) == PICKUP_LOC[0]) & (round(TAXI_DATASET['pickup_latitude'], DECIMAL_PRECISION) == PICKUP_LOC[1])]
    return dataset

def count_query_pickup():
    dataset = query_data()
    return len(dataset)

def get_user_locations(dataset):
    dataset = query_data()
    return [dataset['pickup_longitude'], dataset['pickup_latitude']]

def count_query_with_dp(epsilon):
    count = len(query_data())
    return laplace_mech(count, 1, epsilon)


def collect_lat_long_data(epsilons):
    rel_errs = []
    true_res = count_query_pickup()
    for eps in epsilons:
        for _ in range(500):
            rel_errs.append((eps, relative_error(count_query_with_dp(eps), true_res)))
    return pd.DataFrame(rel_errs, columns=['Epsilon', 'Relative error'])

epsilons = np.round(np.geomspace(0.001, 25, num=20), 3)
err_df = collect_lat_long_data(epsilons)

print(count_query_pickup())
print(count_query_with_dp(1))

g = sns.scatterplot(data=err_df, x='Epsilon', y='Relative error')
g.set_xscale('log')
sns.despine()
plt.show()
