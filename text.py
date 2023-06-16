import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NYC taxi dataset
TAXI_DATASET = pd.read_csv('../archive/yellow_tripdata_2015-01.csv', usecols=['trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
DECIMAL_PRECISION = 4
LOCATION_FLOAT = [round(float('-73.990372'), DECIMAL_PRECISION), round(float('40.73470'), DECIMAL_PRECISION)]
EPSILON = 1.0

# Define the total privacy budget (epsilon)
epsilon = 1.0

# Define the specific location of interest (pickup latitude and longitude)
location_latitude = round(float('40.73470'), DECIMAL_PRECISION)
location_longitude = round(float('-73.990372'), DECIMAL_PRECISION)

def count_query_pickup(dataset):
    count = len(dataset[(round(dataset['pickup_longitude'], DECIMAL_PRECISION) == LOCATION_FLOAT[0]) & (round(dataset['pickup_latitude'], DECIMAL_PRECISION) == LOCATION_FLOAT[1])])
    return count

# Laplace Mechanism for Count Query
def laplace_mech(v, sensitivity, epsilon):
    scale = sensitivity / epsilon
    return v + np.random.laplace(loc=0, scale=scale)

# Local Differential Privacy Mechanism
def local_dp_mech(v, sensitivity, epsilon):
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon)
    return v + noise

# Renyi Differential Privacy Mechanism
def renyi_dp_mech(v, sensitivity, epsilon):
    alpha = 2.0  # Renyi divergence parameter
    sigma = sensitivity * np.sqrt(2 * np.log(1.0 / epsilon)) / (alpha - 1)
    noise = np.random.normal(loc=0, scale=sigma)
    return v + noise

# Count Query with Differential Privacy
def count_trips_with_dp(df, location_latitude, location_longitude, epsilon, mechanism):
    # Apply privacy mechanisms to the count query
    count = len(df[(round(df['dropoff_latitude'], DECIMAL_PRECISION) == location_latitude) & (round(df['dropoff_longitude'], DECIMAL_PRECISION) == location_longitude)])
    print(count)
    sensitivity = 1  # Sensitivity of the count query
    if mechanism == 'laplace':
        noisy_count = laplace_mech(count, sensitivity, epsilon)
    elif mechanism == 'local_dp':
        noisy_count = local_dp_mech(count, sensitivity, epsilon)
    elif mechanism == 'renyi_dp':
        noisy_count = renyi_dp_mech(count, sensitivity, epsilon)
    else:
        raise ValueError("Invalid mechanism specified")
    return noisy_count

# Calculate the count of trips from the specified location with differential privacy
mechanisms = ['laplace', 'local_dp', 'renyi_dp']
noisy_trip_counts = {}
relative_errors = {}
true_count = count_query_pickup(TAXI_DATASET)

for mechanism in mechanisms:
    noisy_trip_counts[mechanism] = count_trips_with_dp(TAXI_DATASET, location_latitude, location_longitude, epsilon, mechanism)
    relative_errors[mechanism] = np.abs(1 - (noisy_trip_counts[mechanism] / true_count))

print("Noisy Trip Counts:", noisy_trip_counts)
print("Relative Errors:", relative_errors)

# Plotting the Relative Error Graph
sns.barplot(x=mechanisms, y=list(relative_errors.values()))
plt.xlabel("Mechanism")
plt.ylabel("Relative Error")
plt.title("Relative Error for Differential Privacy Mechanisms")
plt.show()

# Finding the mechanism with the lowest relative error
best_mechanism = min(relative_errors, key=relative_errors.get)
print("Best Mechanism:", best_mechanism)