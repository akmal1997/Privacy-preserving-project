from faker import Faker
import pandas as pd
import numpy as np
import random
output_file = 'preprocessed_trip_data.csv'
TAXI_DATASET = pd.read_csv('../archive/yellow_tripdata_2015-01.csv', usecols=['fare_amount','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
trip_df = TAXI_DATASET[(TAXI_DATASET['pickup_latitude'] != TAXI_DATASET['dropoff_latitude']) & 
                                        (TAXI_DATASET['pickup_longitude'] != TAXI_DATASET['dropoff_longitude']) &
                                        (TAXI_DATASET['pickup_latitude'] != 0) & 
                                        (TAXI_DATASET['pickup_longitude'] != 0) &
                                        (TAXI_DATASET['dropoff_longitude'] != 0) & 
                                        (TAXI_DATASET['dropoff_latitude'] != 0)
                                        ]


fake = Faker()
fake_names = [fake.name() for i in range(100)]
names = []
for i in range(12403492):
    names.append(random.choice(fake_names))

trip_df['name'] = [random.choice(fake_names) for _ in range(len(trip_df))]
trip_df.to_csv(output_file, index=False)
print("Data written to", output_file)