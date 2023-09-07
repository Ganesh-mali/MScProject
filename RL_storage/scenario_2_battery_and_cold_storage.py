from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
import pandas as pd
import os

# Create directories for saving models and logs
models_dir_2 = "models/PPO_2"
logdir_2 = "logs_2"

if not os.path.exists(models_dir_2):
    os.makedirs(models_dir_2)

if not os.path.exists(logdir_2):
    os.makedirs(logdir_2)

# Define custom Gym environment for battery + cold storage energy system
class EnergyStorageEnv(gym.Env):
    def __init__(self):
        super(EnergyStorageEnv, self).__init__()
        
        # Load data from Excel sheet
        df = pd.read_excel('C:\\Kings_College_Subject_Docs\\MSc_Individual_Project\\PAID_PROJECT\\week_4_update\\energy_storage_data_aug.xlsx')
        self.data = df['load_profile'] - df['adjusted_solar_profile']
        
        # Define action space
        self.action_space = spaces.MultiDiscrete([3, 3])
        
        # Define observation space
        self.observation_space = spaces.Box(low=np.array([-100, 0, 0]), high=np.array([100, 100, 100]), dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.battery_state = 50.0
        self.cold_storage_state = 50.0
        return [self.data.iloc[self.current_step], self.battery_state, self.cold_storage_state]
    
    def step(self, action):
        
        if self.current_step >= len(self.data) - 1:
            done = True
            return [0, self.battery_state, self.cold_storage_state], 0, done, {}
        
        self.current_step += 1
        net_load = self.data.iloc[self.current_step]
        
        new_battery_state = np.clip(self.battery_state + 10 * (action[0] - 1), 0, 100)
        new_cold_storage_state = np.clip(self.cold_storage_state + 10 * (action[1] - 1), 0, 100)
        
        new_net_load = net_load - 10 * (action[0] - 1) - 10 * (action[1] - 1)
        
        reward = -abs(new_net_load) - 5 * abs(action[0] - 1) - 1 * abs(action[1] - 1) + 0.5 * (action[1] - 1)

        
        self.battery_state = new_battery_state
        self.cold_storage_state = new_cold_storage_state
        
        done = False if self.current_step < len(self.data) - 1 else True
        
        return [new_net_load, new_battery_state, new_cold_storage_state], reward, done, {}

# Initialize model and environment
learning_rate_2 = 1e-3
n_steps_2 = 2048

env_2 = EnergyStorageEnv()
model_2 = PPO("MlpPolicy", env_2, verbose=1, tensorboard_log=logdir_2, learning_rate=learning_rate_2, n_steps=n_steps_2)

# Early stopping variables
best_avg_reward_2 = -float('inf')
no_improvement_epochs_2 = 0
MAX_NO_IMPROVEMENT_EPOCHS_2 = 10

# Training loop
TIMESTEPS = 100000
reward_list_2 = []

for epoch in range(1, 100):
    model_2.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_2")
    model_2.save(f"{models_dir_2}/{TIMESTEPS*epoch}")
    
    obs = env_2.reset()
    episode_rewards_2 = 0
    for i in range(len(env_2.data)):
        action, _states = model_2.predict(obs)
        obs, reward, done, info = env_2.step(action)
        episode_rewards_2 += reward
        if done:
            break
    
    avg_reward_2 = episode_rewards_2 / len(env_2.data)
    reward_list_2.append(avg_reward_2)
    
    if avg_reward_2 > best_avg_reward_2:
        best_avg_reward_2 = avg_reward_2
        no_improvement_epochs_2 = 0
    else:
        no_improvement_epochs_2 += 1
    
    if no_improvement_epochs_2 >= MAX_NO_IMPROVEMENT_EPOCHS_2:
        print(f"Stopping training for Scenario-2. No improvement after {MAX_NO_IMPROVEMENT_EPOCHS_2} epochs.")
        break


# Model Evaluation Function
def evaluate_model(model, env, num_episodes=10):
    all_episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = 0
        for step in range(len(env.data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        all_episode_rewards.append(episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward

# Evaluate the trained model for Scenario-2
mean_reward_2 = evaluate_model(model_2, env_2, num_episodes=10)
print(f"Mean reward over 10 episodes for Scenario-2: {mean_reward_2}")    