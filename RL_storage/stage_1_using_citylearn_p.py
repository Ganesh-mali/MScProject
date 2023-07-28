from gym import spaces
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO, A2C, DQN
from pathlib import Path
import numpy as np

class CustomRewardFunction:
    def __init__(self, env):
        self.env = env
        print("CustomRewardFunction has been initialized!")

    def get_reward(self):
        # Reward for maximizing self-consumption
        reward_self_consumption = self.env.building.electricity_production - abs(self.env.building.electricity_consumption - self.env.building.electricity_production)

        # Penalty for importing or exporting energy
        penalty_import_export = abs(self.env.building.grid_import) + abs(self.env.building.grid_export)

        # Penalty for unnecessary charge and discharge cycles
        penalty_cycles = abs(self.env.building.battery.state_of_charge - self.env.building.battery.state_of_charge_previous_step)

        print("Calculating reward with CustomRewardFunction...")

        return reward_self_consumption - penalty_import_export - penalty_cycles

class CityLearnWrapper(CityLearnEnv):
    def __init__(self, schema_filepath):
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,)) # update with the correct value
        self._action_space = spaces.Discrete(2) # update with the correct value
        super().__init__(schema_filepath)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        obs = super().reset()
        if obs is not None:
            return np.concatenate(obs)

        else:
            return None

    def step(self, action):
        print(f"Action before conversion: {action}, Type: {type(action)}")
        action = float(action)  # convert action to float
        print(f"Action after conversion: {action}, Type: {type(action)}")
        action_dict = {'Building_1': {'electrical_storage': action}}
        obs, reward, done, info = super().step(action_dict)
        return np.concatenate(list(obs.values())), reward, done, info

# Load the environment from the schema file
schema_filepath = Path(r"C:\Kings_College_Subject_Docs\MSc_Individual_Project\PAID_PROJECT\week_2_update\RL+algo_practice\city_learn_simulation\submission\schema.json")
env = CityLearnWrapper(schema_filepath)

# Define the models
ppo_model = PPO('MlpPolicy', env, verbose=0)
a2c_model = A2C('MlpPolicy', env, verbose=0)
dqn_model = DQN('MlpPolicy', env, verbose=0)

# Train the models for one year
ppo_model.learn(total_timesteps=8760)
a2c_model.learn(total_timesteps=8760)
dqn_model.learn(total_timesteps=8760)

# Save the models
ppo_model.save('ppo_citylearn')
a2c_model.save('a2c_citylearn')
dqn_model.save('dqn_citylearn')
