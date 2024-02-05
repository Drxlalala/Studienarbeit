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
import Interation


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


def mutate(params: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
   return dict((name, param + th.rand_like(param)) for name, param in params.items())
# add the noise to the params

def iteration(model):
   # load the params from the model
   mean_params = dict(
      {key, value}
      for key, value in model.policy.state_dict().items()
      if ("policy" in key or "shared_net" in key or "action" in key)
   )

   # population size of 50 invdiduals
   pop_size = 50
   # keep top 10%
   n_elite = pop_size // 10
   # Retrieve the env
   vec_env = model.get_env()

   for iteration in range(5):

      # create population of candidates and evaluate them
      population = []
      for population_i in range(pop_size):
         candidate = mutate(mean_params)

         model.policy.load_state_dict(candidate, strict=False)

         fitness, _ = evaluate_policy(model, vec_env)
         population.append((candidate, fitness))

      top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:n_elite]

      # get the better params(mean_params) and load it into the model
      mean_params = dict(
         (
            name,
            th.stack([candidate[0][name] for candidate in top_candidates]).mean(dim=0),
         )
         for name in mean_params.keys()
      )
      mean_fitness = sum(candidate[1] for candidate in top_candidates) / n_elite
      print(f"Iteration {iteration + 1:<3} Mean top fitness: {mean_fitness: .2f}")
      print(f"Best fitness: {top_candidates[0][1]:.2f}")

   return mean_params


class Train_Type:

   def sac_train():
      loaded_model = SAC.load('underwater_save/SAC.sparse')

      loaded_model.load_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_sparse')

      vec_env = DummyVecEnv([make_env('underwater_env') for _ in range(8)])

      loaded_model.set_env(vec_env)

      mean_params = iteration(loaded_model)

      loaded_model.policy.load_state_dict(mean_params, strict=False)

      loaded_model.learn(50000, callback=eval_callback)


      loaded_model.save('underwater_save/SAC.sparse')

      loaded_model.save_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_sparse')

         # save policy from the model
      policy = loaded_model.policy
      policy.save('underwater_save/SAC.PolicyPickAndPlace_Underwater_sparse')


         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)


      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")




   

   def ddpg_train():

      loaded_model = DDPG.load('underwater_save/DDPG.sparse')

      loaded_model.load_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_sparse')

      vec_env = DummyVecEnv([make_env('underwater_env') for _ in range(8)])

      loaded_model.set_env(vec_env)

      loaded_model.learn(50000, callback=eval_callback)


      loaded_model.save('underwater_save/DDPG.sparse')

      loaded_model.save_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_sparse')

         # save policy from the model
      policy = loaded_model.policy

      policy.save('underwater_save/DDPG.PolicyPickAndPlace_Underwater_sparse')

         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
      

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



Train = 1


underwater_env = UnderwaterFetchEnv('FetchPickAndPlace-v2')
env_id = 'FetchPickAndPlace-v2'
log_dir = './fetch_pickandplace_tensorboard_sparse/'
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
         model.save('underwater_save/DDPG.sparse')
         model.save_replay_buffer('underwater_save/DDPG.ReplayBuffer_Underwater_sparse')

         model = SAC( policy='MultiInputPolicy', env=make_vec_env(env_id, n_envs=8) , learning_rate=1e-3, buffer_size=10_0000, learning_starts=100, batch_size=256, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1, tensorboard_log=log_dir)
         model.learn(50000, callback=eval_callback)
         model.save('underwater_save/SAC.sparse')
         model.save_replay_buffer('underwater_save/SAC.ReplayBuffer_Underwater_sparse')

         already_trained = True
      else:
         Train_Type.sac_train()
         Train_Type.ddpg_train()






















