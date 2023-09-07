from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
import pandas as pd
import os

# Create directories for models and logs
models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Custom Gym environment for battery-only energy storage system
class BatteryOnlyEnv(gym.Env):
    def __init__(self):
        super(BatteryOnlyEnv, self).__init__()
        
        # Load data from Excel sheet
        df = pd.read_excel('C:\\Kings_College_Subject_Docs\\MSc_Individual_Project\\PAID_PROJECT\\week_4_update\\energy_storage_data_aug.xlsx')
        self.data = df['load_profile'] - df['adjusted_solar_profile']
        
        # Define action space
        self.action_space = spaces.Discrete(3)
        
        # Define state space
        self.observation_space = spaces.Box(low=np.array([-100, 0]), high=np.array([100, 100]), dtype=np.float32)
        
        # Reset environment variables
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.battery_state = 50.0
        return [self.data.iloc[self.current_step], self.battery_state]
    
    def step(self, action):
        
        if self.current_step >= len(self.data) - 1:
            done = True
            return [0, self.battery_state], 0, done, {}
        
        # Move to the next time step
        self.current_step += 1
        net_load = self.data.iloc[self.current_step]  # Added this line to get the net_load
        
        # Update battery state based on action
        new_battery_state = np.clip(self.battery_state + 10 * (action - 1), 0, 100)
        
        # Calculate new net_load after action
        new_net_load = net_load - 10 * (action - 1)
        
        # Reward function
        reward = -np.square(new_net_load) - 5 * abs(action - 1)
        
        # Update the battery state
        self.battery_state = new_battery_state
        
        done = False if self.current_step < len(self.data) - 1 else True
        
        return [new_net_load, new_battery_state], reward, done, {}

# Initialize model and environment
# Modified Hyperparameters
learning_rate_1 = 1e-3  # Adjusted learning rate
n_steps_1 = 2048  # Adjusted number of steps

env = BatteryOnlyEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate_1, n_steps=n_steps_1)

# Early stopping parameters
best_avg_reward = -float('inf')
no_improvement_epochs = 0
MAX_NO_IMPROVEMENT_EPOCHS = 10  

# Training loop
TIMESTEPS = 100000
reward_list = []

for epoch in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*epoch}")
    
    obs = env.reset()
    episode_rewards = 0
    for i in range(len(env.data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    
    avg_reward = episode_rewards / len(env.data)
    reward_list.append(avg_reward)
    
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
    
    if no_improvement_epochs >= MAX_NO_IMPROVEMENT_EPOCHS:
        print(f"Stopping training for Scenario-1. No improvement after {MAX_NO_IMPROVEMENT_EPOCHS} epochs.")
        break

# Model Evaluation
def evaluate_model(model, env, num_episodes=10):
    all_episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = 0
        for step in range(len(env.data)):
            action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        all_episode_rewards.append(episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward

# Evaluate the trained model
mean_reward = evaluate_model(model, env, num_episodes=10)
print(f"Mean reward over 10 episodes: {mean_reward}")