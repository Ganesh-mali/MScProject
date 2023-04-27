

# This code defines a total_cost() function that calculates the total cost of a system using capital cost per kWh, operating cost per kWh, capacity in kWh, and system lifetime. The example uses sample data for battery storage and cold storage systems and calculates the total cost for each system.

# In a real-world scenario, I need to replace the sample data with actual cost data from research, industry partners, or other sources. Also I need to refine the cost model by considering additional factors like inflation, discount rates, or variable costs.

# The levelized cost of energy (LCOE) can be another useful metric for comparing the costs of different energy systems.

import pandas as pd
import matplotlib.pyplot as plt

# Sample data for battery storage and cold storage systems
data = {
    "battery_storage": {
        "capital_cost_per_kWh": 200,  # Capital cost in $/kWh
        "operating_cost_per_kWh": 5,  # Annual operating cost in $/kWh
        "lifetime": 10,  # Lifetime in years
    },
    "cold_storage": {
        "capital_cost_per_kWh": 150,  # Capital cost in $/kWh
        "operating_cost_per_kWh": 8,  # Annual operating cost in $/kWh
        "lifetime": 15,  # Lifetime in years
    }
}

cost_data = pd.DataFrame(data)


def total_cost(capital_cost_per_kWh, operating_cost_per_kWh, capacity_kWh, lifetime):
    """
    Calculate the total cost of a system using a simple linear cost model.
    :param capital_cost_per_kWh: Capital cost per kWh in $
    :param operating_cost_per_kWh: Annual operating cost per kWh in $
    :param capacity_kWh: Capacity of the system in kWh
    :param lifetime: Lifetime of the system in years
    :return: Total cost in $
    """
    capital_cost = capital_cost_per_kWh * capacity_kWh
    operating_cost = operating_cost_per_kWh * capacity_kWh * lifetime
    return capital_cost + operating_cost


# Example usage
battery_storage_capacity = 100  # Capacity in kWh
cold_storage_capacity = 80  # Capacity in kWh

battery_storage_total_cost = total_cost(
    cost_data.loc["capital_cost_per_kWh", "battery_storage"],
    cost_data.loc["operating_cost_per_kWh", "battery_storage"],
    battery_storage_capacity,
    cost_data.loc["lifetime", "battery_storage"]
)

cold_storage_total_cost = total_cost(
    cost_data.loc["capital_cost_per_kWh", "cold_storage"],
    cost_data.loc["operating_cost_per_kWh", "cold_storage"],
    cold_storage_capacity,
    cost_data.loc["lifetime", "cold_storage"]
)

print(f"Battery storage total cost: ${battery_storage_total_cost:.2f}")
print(f"Cold storage total cost: ${cold_storage_total_cost:.2f}")

# Plot the total costs
plt.bar(["Battery storage", "Cold storage"], [battery_storage_total_cost, cold_storage_total_cost])
plt.title("Total costs of battery storage and cold storage systems")
plt.ylabel("Total cost ($)")
plt.show()
