# Load the data and libraries
import pandas as pd
import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# plt.style.use/('seaborn-whitegrid')

# Step 1: Load the data set
df = pd.read_csv('../archive/combined_file.csv', usecols=['trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

# Step 2: Define the Query or Computation
def get_user_locations(dataset):
    pickup_longitude = dataset['pickup_longitude']
    pickup_latitude = dataset['pickup_latitude']
    dropoff_longitude = dataset['dropoff_longitude']
    dropoff_latitude = dataset['dropoff_latitude']
    return pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude

# Step 3: Determine Sensitivity
def calculate_sensitivity(dataset):
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude = get_user_locations(dataset)
    max_longitude = max(np.max(pickup_longitude), np.max(dropoff_longitude))
    min_longitude = min(np.min(pickup_longitude), np.min(dropoff_longitude))
    max_latitude = max(np.max(pickup_latitude), np.max(dropoff_latitude))
    min_latitude = min(np.min(pickup_latitude), np.min(dropoff_latitude))
    return max(max_longitude - min_longitude, max_latitude - min_latitude)

# Step 4: Choose Privacy Budget
epsilon = 1.0

def add_noise_to_locations(locations, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noisy_locations = np.copy(locations)  # Create a copy of the locations array

    for i in range(len(locations)):
        noise = np.random.laplace(0.0, scale)
        noisy_locations[i] += noise

    return noisy_locations

# Step 6: Implement the Differential Privacy Mechanism
def differentially_private_user_locations(dataset, epsilon):
    user_locations = get_user_locations(dataset)
    sensitivity = calculate_sensitivity(dataset)
    noisy_locations = add_noise_to_locations(user_locations, sensitivity, epsilon)
    return noisy_locations

# Compute the non-private user locations
non_private_locations = get_user_locations(df)

# Compute the differentially private user locations
differentially_private_locations = differentially_private_user_locations(df, epsilon)

print("Non-private user locations:")
print(non_private_locations)

print("Differentially private user locations:")
print(differentially_private_locations)

# Step 8: Fine-tuning (Adjust epsilon and evaluate the impact on privacy and utility)