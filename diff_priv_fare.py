# Load the data and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DECIMAL_PRECISION = 6
TAXI_DATASET = pd.read_csv('./preprocessed_trip_data.csv')
EPSILON = 1.0
PICKUP_LOC = [float('-74.004639'), float('40.742039')] # pickup_longitude  pickup_latitude
driver_name = 'Kelly Castaneda'



def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def relative_error(priv, orig):
    return np.abs(1 - (priv/orig))

# def count_query_pickup(dataset):

def query_data():
    dataset = TAXI_DATASET[(TAXI_DATASET['name'] == driver_name)]
    return dataset['fare_amount']

def count_query_pickup():
    mean = np.mean(query_data())
    return mean

def get_user_locations(dataset):
    dataset = query_data()
    return [dataset['pickup_longitude'], dataset['pickup_latitude']]

def mean_query_with_dp(epsilon):
    mean = np.mean(query_data())
    return laplace_mech(mean, 1, epsilon)


def collect_lat_long_data(epsilons):
    rel_errs = []
    true_res = count_query_pickup()
    for eps in epsilons:
        for _ in range(20):
            rel_errs.append((eps, relative_error(mean_query_with_dp(eps), true_res)))
    return pd.DataFrame(rel_errs, columns=['Epsilon', 'Relative error'])

epsilons = np.round(np.geomspace(0.001, 25, num=20), 3)
err_df = collect_lat_long_data(epsilons)

g = sns.scatterplot(data=err_df, x='Epsilon', y='Relative error')
g.set_xscale('log')
sns.despine()
plt.show()
#0.07