import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create artificial dataset
np.random.seed(42)
n_samples = 1000

ambient_temperature = np.random.normal(20, 5, n_samples)
building_volume = np.random.uniform(1000, 5000, n_samples)
insulation_level = np.random.uniform(0.5, 2, n_samples)

# Assume a simple linear relationship for this example
heat_demand = 0.5 * ambient_temperature - 0.2 * building_volume + 0.3 * insulation_level + np.random.normal(0, 10, n_samples)

data = pd.DataFrame({'ambient_temperature': ambient_temperature,
                     'building_volume': building_volume,
                     'insulation_level': insulation_level,
                     'heat_demand': heat_demand})

# Plot the data
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

axs[0].scatter(data['ambient_temperature'], data['heat_demand'])
axs[0].set_xlabel('Ambient temperature')
axs[0].set_ylabel('Heat demand')

axs[1].scatter(data['building_volume'], data['heat_demand'])
axs[1].set_xlabel('Building volume')
axs[1].set_ylabel('Heat demand')

axs[2].scatter(data['insulation_level'], data['heat_demand'])
axs[2].set_xlabel('Insulation level')
axs[2].set_ylabel('Heat demand')

plt.tight_layout()
plt.show()

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train the model
X_train = train_data[['ambient_temperature', 'building_volume', 'insulation_level']]
y_train = train_data['heat_demand']
reg = LinearRegression().fit(X_train, y_train)

# Evaluate the model
X_test = test_data[['ambient_temperature', 'building_volume', 'insulation_level']]
y_test = test_data['heat_demand']
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared: ", r2)

# Plot the predicted vs actual heat demand values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual heat demand")
plt.ylabel("Predicted heat demand")
plt.title("Predicted vs actual heat demand")
plt.show()
