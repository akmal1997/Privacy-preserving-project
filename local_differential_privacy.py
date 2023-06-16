import numpy as np
from numpy.random import laplace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TAXI_DATASET = pd.read_csv('./preprocessed_trip_data.csv')

# Define the privacy parameter (epsilon) and sensitivity of the query
epsilon = 0.11
sensitivity = 1

def filter_dataset(dataset, driver_name):
    filtered_dataset = dataset[(dataset['name'] == driver_name)]
    return filtered_dataset

def add_noise(data, epsilon, sensitivity):
    noisy_data = data.apply(lambda x: x + np.random.laplace(loc=0, scale=sensitivity/epsilon))
    return noisy_data

def compute_mean(data):
    # Compute the mean of the data
    mean = np.mean(data)
    return mean

def calculate_sensitivity(data):
    # Compute the sensitivity as the maximum absolute difference in fare amounts
    max_fare = np.max(data['fare_amount'])
    min_fare = np.min(data['fare_amount'])
    sensitivity = max_fare - min_fare
    return sensitivity

def query_average_fare_ldp(dataset, driver_name, epsilon, sensitivity):
    fares = dataset['fare_amount']
    noisy_fares = add_noise(fares, epsilon, sensitivity)
    
    noisy_mean = compute_mean(noisy_fares)
    
    return noisy_mean

def relative_error(priv, orig):
    return np.abs(1 - (priv/orig))

def collect_lat_long_data(filtered_dataset,original_mean,epsilons):
    rel_errs = []
    for eps in epsilons:
        for _ in range(500):
            noisy_mean = query_average_fare_ldp(filtered_dataset, driver_name, eps, sensitivity)
            rel_errs.append((eps, relative_error(noisy_mean, original_mean)))
    return pd.DataFrame(rel_errs, columns=['Epsilon', 'Relative error'])

# Example usage:
driver_name = 'Kelly Castaneda'
epsilons = np.round(np.geomspace(0.001, 25, num=20), 3)
filtered_dataset = filter_dataset(TAXI_DATASET, driver_name)
original_mean = compute_mean(filtered_dataset['fare_amount'])
err_df = collect_lat_long_data(filtered_dataset, original_mean, epsilons)

g = sns.scatterplot(data=err_df, x='Epsilon', y='Relative error')
g.set_xscale('log')
sns.despine()
plt.show()

# 0.07