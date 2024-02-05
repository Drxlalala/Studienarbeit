import gymnasium as gym
import gymnasium_robotics
from typing import Dict
from stable_baselines3.common.callbacks import EvalCallback
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
import Interation


class NormalEnv(gym.Wrapper):
    def __init__(self, env_name, move_interval=15, velocity_magnitude=1):

      env = gym.make('FetchReach-v2')
      super().__init__(env)
      self.env = env
      self.distance_threshold = 0.05
   

    def step(self, action):

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

      return self.env.step(action, reward)

    def reset(self, seed=None):
        self.step_count = 0
        return self.env.reset()

   
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_env(env_name):
    def _init():
        env = NormalEnv(env_name)
        return env
    return _init



Train = 1



class Train_Type:

   def sac_train():
      loaded_model = SAC.load('underwater_save/SAC.curriculum_Underwater',tensorboard_log=log_dir)

      loaded_model.load_replay_buffer('save/SAC.ReplayBuffer')

      loaded_model.set_env(DummyVecEnv([lambda: gym.make('normal_env') for _ in range(8)]))


      loaded_model.learn(50000, callback=eval_callback)


      loaded_model.save('save/SAC.model_curriculum')

      loaded_model.save_replay_buffer('save/SAC.ReplayBuffer_curriculum')

         # save policy from the model
      policy = loaded_model.policy
      policy.save('save/SAC.policy_curriculum')


         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

      total_mean_reward.append((mean_reward,std_reward))

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

      del loaded_model




   

   def ddpg_train():

      loaded_model = DDPG.load('save/DDPG.curriculum_Underwater',tensorboard_log=log_dir)

      loaded_model.load_replay_buffer('save/DDPG.ReplayBuffer_curriculum')

      loaded_model.set_env(DummyVecEnv([lambda: gym.make('normal_env') for _ in range(8)]))

      loaded_model.learn(50000, callback=eval_callback)

      loaded_model.save('save/DDPG.model_curriculum')

      loaded_model.save_replay_buffer('save/DDPG.ReplayBuffer_curriculum')

         # save policy from the model
      policy = loaded_model.policy

      policy.save('save/DDPG.policy_curriculum')

         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
      

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



normal_env = NoramlEnv('FetchReach', render_mode="human", max_episode_steps=100)
env_id = 'FetchReach'
log_dir = './fetch_curriculum/'
eval_callback = EvalCallback(normal_env, log_path=log_dir, eval_freq=500)

if Train:
   state_file_path = 'training_state.txt'


   total_trainings = 1

   

   # check if the model has be trained
   if os.path.exists(state_file_path):
      already_trained = True
   else:
      already_trained = False

      with open(state_file_path, 'w') as f:
         f.write('MODEL_HAS_TRAINED')

   for training in range(total_trainings):
      if not already_trained:
         model = DDPG( policy='MultiInputPolicy', env=make_vec_env(normal_env, n_envs=8), replay_buffer_class=HerReplayBuffer, learning_rate=1e-3, buffer_size=10_0000,learning_starts=100, batch_size=512, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1, tensorboard_log=log_dir)
         model = SAC( policy='MultiInputPolicy', env=make_vec_env(normal_env, n_envs=8), replay_buffer_class=HerReplayBuffer, learning_rate=1e-3, buffer_size=10_0000, learning_starts=100, batch_size=256, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1, tensorboard_log=log_dir)
         model.learn(50000, progress_bar=True)

         model.save('save/DDPG.model_curriculum')
         model.save_replay_buffer('save/DDPG.ReplayBuffer_curriculum')

         model.save('save/SAC.model_curriculum')
         model.save_replay_buffer('save/SAC.ReplayBuffer_curriculum')

         already_trained = True

      else:
         # Not the first time to train the model, use the saved model to train again

         Train_Type.sac_train()

         Train_Type.ddpg_train()
