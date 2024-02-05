import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
from typing import Dict
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch as th
import os



class UnderwaterFetchEnv(gym.Wrapper):
    def __init__(self, env_name, move_interval=15, velocity_magnitude=1):

      env = gym.make('FetchPickAndPlaceDense-v2')
      super().__init__(env)
      self.env = env
      self.move_interval = 5
      self.force_magnitude = 70
      self.step_count = 0
      self.time_step = 0.2 
      self.rho = 1000
      self.s_object = 0.01
      self.Cd = 0.3
      self.distance_threshold = 0.05
   

    def step(self, action):

      if self.step_count % self.move_interval == 0:

         v_water = np.random.uniform(0.5, -0.5, size=3)

         P = 0.5 * rho * v_water**2

         random_force = P * s_object # set a random force
            # extend the force 
         extended_force = np.concatenate([random_force, np.zeros(3)])
            # set force
         object_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
            
         data = self.env.unwrapped.data
         data.xfrc_applied[object_id] = extended_force

      v_object = self._utils.get_site_xvelp(self.model, self.data, "robot0")
      v_rel = v_object - v_water

      F_resistance = 0.5 * Cd * rho * A * v_rel**2
      data.xfrc_applied[object_id] = extended_force

      

      self.r1 = -d
      self.r2 = (1-(d > self.distance_threshold).astype(np.float32))*30
      previous_d = None
      if previous_d is not None:  # Skip comparison for the first iteration
        if d > previous_d:
            self.r3 = -0.005
        elif d < previous_d:
            self.r3 = 0.005
        else:
            self.r3 = 0
      previous_d = d

      reward = self.r1 + self.r2 + self.r3

      self.step_count += 1

      return self.env.step(action)

    def reset(self, seed=None):
        self.step_count = 0
        return self.env.reset()

   
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_env(env_name):
    def _init():
        env = UnderwaterFetchEnv(env_name)
        return env
    return _init



class Train_Type:

   def sac_train():
      loaded_model = SAC.load('underwater_save/SAC.FetchPickAndPlace_Underwater_hybrid')

      loaded_model.load_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_hybrid')

      vec_env = DummyVecEnv([make_env('underwater_env') for _ in range(8)])

      loaded_model.set_env(vec_env)

      mean_params = iteration(loaded_model)

      loaded_model.policy.load_state_dict(mean_params, strict=False)

      loaded_model.learn(50000, callback=eval_callback)


      loaded_model.save('underwater_save/SAC.FetchPickAndPlace_Underwater_hybrid')

      loaded_model.save_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_hybrid')

         # save policy from the model
      policy = loaded_model.policy
      policy.save('underwater_save/SAC.PolicyPickAndPlace_Underwater_hybrid')


         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)


      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")




   

   def ddpg_train():

      loaded_model = DDPG.load('underwater_save/DDPG.FetchPickAndPlace_Underwater_hybrid')

      loaded_model.load_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_hybrid')

      vec_env = DummyVecEnv([make_env('underwater_env') for _ in range(8)])

      loaded_model.set_env(vec_env)

      loaded_model.learn(50000, callback=eval_callback)


      loaded_model.save('underwater_save/DDPG.FetchPickAndPlace_Underwater_hybrid')

      loaded_model.save_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_hybrid')

         # save policy from the model
      policy = loaded_model.policy

      policy.save('underwater_save/DDPG.PolicyPickAndPlace_Underwater_hybrid')

         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
      

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



Train = 1


underwater_env = UnderwaterFetchEnv('FetchPickAndPlaceDense-v2')
env_id = 'FetchPickAndPlaceDense-v2'
log_dir = './fetch_pickandplace_tensorboard/'
eval_callback = EvalCallback(underwater_env, log_path=log_dir, eval_freq=500)

if Train:
   state_file_path = 'training_state.txt'


   total_trainings = 200

   

   # check if the model has be trained
   if os.path.exists(state_file_path):
      already_trained = True
   else:
      already_trained = False

      with open(state_file_path, 'w') as f:
         f.write('MODEL_HAS_TRAINED')
   rewards = []

   for i in range(total_trainings):
      if not already_trained:

 
         vec_env = DummyVecEnv([lambda: underwater_env])
      
         model = DDPG( policy='MultiInputPolicy', env=make_vec_env(env_id, n_envs=8),learning_rate=1e-3, buffer_size=10_0000, learning_starts=100, batch_size=256, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1, tensorboard_log=log_dir)
         model.learn(50000, callback=eval_callback)
         model.save('underwater_save/DDPG.FetchPickAndPlace_Underwater_hybrid')
         model.save_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_hybrid')

         model = SAC( policy='MultiInputPolicy', env=make_vec_env(env_id, n_envs=8) , learning_rate=1e-3, buffer_size=10_0000, learning_starts=100, batch_size=256, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1, tensorboard_log=log_dir)
         model.learn(50000, callback=eval_callback)
         model.save('underwater_save/SAC.FetchPickAndPlace_Underwater_hybrid')
         model.save_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_hybrid')

         already_trained = True
      else:
         Train_Type.sac_train()
         Train_Type.ddpg_train()






















