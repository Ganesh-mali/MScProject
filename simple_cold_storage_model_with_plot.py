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
food_mass = np.random.uniform(100, 500, n_samples)
cooling_capacity = np.random.uniform(50, 100, n_samples)

# Assume a simple linear relationship for this example
cold_energy_demand = 0.3 * ambient_temperature + 0.5 * food_mass + 0.2 * cooling_capacity + np.random.normal(0, 10, n_samples)

data = pd.DataFrame({'ambient_temperature': ambient_temperature,
                     'food_mass': food_mass,
                     'cooling_capacity': cooling_capacity,
                     'cold_energy_demand': cold_energy_demand})

# Plot the data
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

axs[0].scatter(data['ambient_temperature'], data['cold_energy_demand'])
axs[0].set_xlabel('Ambient temperature')
axs[0].set_ylabel('Cold energy demand')

axs[1].scatter(data['food_mass'], data['cold_energy_demand'])
axs[1].set_xlabel('Food mass')
axs[1].set_ylabel('Cold energy demand')

axs[2].scatter(data['cooling_capacity'], data['cold_energy_demand'])
axs[2].set_xlabel('Cooling capacity')
axs[2].set_ylabel('Cold energy demand')

plt.tight_layout()
plt.show()

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train the model
X_train = train_data[['ambient_temperature', 'food_mass', 'cooling_capacity']]
y_train = train_data['cold_energy_demand']
reg = LinearRegression().fit(X_train, y_train)

# Evaluate the model
X_test = test_data[['ambient_temperature', 'food_mass', 'cooling_capacity']]
y_test = test_data['cold_energy_demand']
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared: ", r2)

# Plot the predicted values vs. the true values
plt.scatter(y_test, y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Predicted vs. true values')
plt.show()
