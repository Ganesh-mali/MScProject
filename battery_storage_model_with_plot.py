import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# For illustration, we will create artificial data
n_samples = 1000
ambient_temperature = np.random.normal(20, 5, n_samples)
load_demand = np.random.uniform(0, 5, n_samples)
battery_capacity = np.random.uniform(50, 100, n_samples)
soc = 0.2 * ambient_temperature + 0.8 * load_demand + 0.05 * battery_capacity + np.random.normal(0, 2, n_samples)
data = pd.DataFrame({'ambient_temperature': ambient_temperature,
                     'load_demand': load_demand,
                     'battery_capacity': battery_capacity,
                     'soc': soc})

# Plot the data
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].scatter(data['ambient_temperature'], data['soc'])
axs[0, 0].set_xlabel('Ambient temperature')
axs[0, 0].set_ylabel('State of charge')

axs[0, 1].scatter(data['load_demand'], data['soc'])
axs[0, 1].set_xlabel('Load demand')
axs[0, 1].set_ylabel('State of charge')

axs[1, 0].scatter(data['battery_capacity'], data['soc'])
axs[1, 0].set_xlabel('Battery capacity')
axs[1, 0].set_ylabel('State of charge')

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train the model
X_train = train_data[['ambient_temperature', 'load_demand', 'battery_capacity']]
y_train = train_data['soc']
reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# Evaluate the model
X_test = test_data[['ambient_temperature', 'load_demand', 'battery_capacity']]
y_test = test_data['soc']
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared: ", r2)

axs[1, 1].scatter(y_test, y_pred)
axs[1, 1].set_xlabel('True SOC')
axs[1, 1].set_ylabel('Predicted SOC')

# Add title and plot labels
plt.suptitle('Battery Storage Model')
plt.tight_layout()
plt.show()
