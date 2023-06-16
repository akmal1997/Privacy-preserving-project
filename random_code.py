# Load the data and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DECIMAL_PRECISION = 6
TARGET_NAME = 'Anita Ball'
TAXI_DATASET = pd.read_csv('./preprocessed_trip_data.csv')
EPSILON = 1.0
PICKUP_LOC = [float('-74.004639'), float('40.742039')] # pickup_longitude  pickup_latitude
# PICKUP_LOC = [float('-73.993896'), float('40.750111')] # pickup_longitude  pickup_latitude
DROPOFF_LOC = [float('-74.005486'), float('40.745808')] # dropoff_longitude  dropoff_latitude
TAXI_DATASET.head(5)

def count_query():
    dataset = TAXI_DATASET[(TAXI_DATASET['name'] == 'Anita Ball') & (round(TAXI_DATASET['pickup_longitude'], DECIMAL_PRECISION) == PICKUP_LOC[0]) & (round(TAXI_DATASET['pickup_latitude'], DECIMAL_PRECISION) == PICKUP_LOC[1])]
    return len(dataset)

def relative_error(priv, orig):
    return np.abs(1 - (priv/orig))

# # Step 2: Define the Query or Computation
def get_user_locations(dataset):
    dataset = dataset[(round(dataset['pickup_longitude'], DECIMAL_PRECISION) == LOCATION_FLOAT[0]) & (round(dataset['pickup_latitude'], DECIMAL_PRECISION) == LOCATION_FLOAT[1])]
    return [dataset['pickup_longitude'], dataset['pickup_latitude']]

# Step 3: Determine Sensitivity
def calculate_sensitivity(dataset):
    pickup_longitude, pickup_latitude = get_user_locations(dataset)
    max_longitude = np.max(pickup_longitude)
    min_longitude = np.min(pickup_longitude)
    max_latitude = np.max(pickup_latitude)
    min_latitude = np.min(pickup_latitude)
    
    return max(max_longitude - min_longitude, max_latitude - min_latitude)

# Step 4: Choose Privacy Budget

def add_noise_to_locations(locations, sensitivity, EPSILON):
    scale = sensitivity / EPSILON
    noisy_locations = np.copy(locations)  # Create a copy of the locations array
    for i in range(len(locations[0])):
        noise = np.random.laplace(0.0, scale)
        noisy_locations[0][i] += noise
    
    for i in range(len(locations[1])):
        noise = np.random.laplace(0.0, scale)
        noisy_locations[1][i] += noise
    return noisy_locations

# Step 6: Implement the Differential Privacy Mechanism
def differentially_private_user_locations(EPSILON):
    user_locations = get_user_locations(TAXI_DATASET)
    sensitivity = calculate_sensitivity(TAXI_DATASET)
    noisy_locations = add_noise_to_locations(user_locations, sensitivity, EPSILON)
    return noisy_locations


def collect_over_30_data(epsilons):
    rel_errs = []
    true_res = get_user_locations(TAXI_DATASET)
    for eps in epsilons:
        for _ in range(500):
            dp_locations = differentially_private_user_locations(eps)
            relative_errors = []
            for location in dp_locations:
                relative_errors.append((relative_error(location[0], true_res[0]), relative_error(location[1], true_res[1])))
                rel_errs.append((eps, (relative_error(location[0], true_res[0]), relative_error(location[1], true_res[1]))))
    
    df_err =  pd.DataFrame(rel_errs, columns=['Epsilon', 'Relative error'])


    g = sns.scatterplot(data=df_err, x='Epsilon', y='Relative error')
    g.set_xscale('log')
    sns.despine()

epsilons = np.round(np.geomspace(0.001, 25, num=40), 3)
err_TAXI_DATASET = collect_over_30_data(epsilons)


#0.11