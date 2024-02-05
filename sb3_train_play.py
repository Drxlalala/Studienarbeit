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

Train = 0

Play = 1


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
      loaded_model = SAC.load('underwater_save/SAC.FetchPickAndPlace_Underwater',tensorboard_log=log_dir)

      loaded_model.load_replay_buffer('save/SAC.ReplayBuffer')

      loaded_model.set_env(DummyVecEnv([lambda: gym.make('FetchPickAndPlaceDense-v2') for _ in range(8)]))


      loaded_model.learn(50000, progress_bar=True)

      #mean_params = iteration(loaded_model)

      #loaded_model.policy.load_state_dict(mean_params, strict=False)

      loaded_model.save('save/SAC.FetchPickAndPlace')

      loaded_model.save_replay_buffer('save/SAC.ReplayBuffer')

         # save policy from the model
      policy = loaded_model.policy
      policy.save('save/SAC.PolicyPickAndPlace')


         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

      total_mean_reward.append((mean_reward,std_reward))

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

      del loaded_model




   

   def ddpg_train():

      loaded_model = DDPG.load('save/DDPG.FetchPickAndPlace_Underwater',tensorboard_log=log_dir)

      loaded_model.load_replay_buffer('save/DDPG.ReplayBuffer')

      loaded_model.set_env(DummyVecEnv([lambda: gym.make('FetchPickAndPlaceDense-v2') for _ in range(8)]))

      loaded_model.learn(50000, callback=eval_callback)

      #mean_params = iteration(loaded_model)

      #loaded_model.policy.load_state_dict(mean_params, strict=False)

      loaded_model.save('save/DDPG.FetchPickAndPlace')

      loaded_model.save_replay_buffer('save/DDPG.ReplayBuffer')

         # save policy from the model
      policy = loaded_model.policy

      policy.save('save/DDPG.PolicyPickAndPlace')

         # retrieve the environment
      env = loaded_model.get_env()

         # evaluate the policy
      mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
      

      print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")





env_id = 'FetchPickAndPlaceDense-v2'

env = gym.make('FetchPickAndPlaceDense-v2', render_mode="human", max_episode_steps=100)
log_dir = './fetch_pickandplace_ddpg/'
eval_callback = EvalCallback(env, log_path=log_dir, eval_freq=500)

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
         #model = DDPG('MultiInputPolicy', env=make_vec_env(env_id, n_envs=8), replay_buffer_class=HerReplayBuffer, learning_rate=1e-3, buffer_size=10_0000,learning_starts=100, batch_size=512, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1 )
         model = SAC( policy='MultiInputPolicy', env=make_vec_env(env_id, n_envs=8), replay_buffer_class=HerReplayBuffer, learning_rate=1e-3, buffer_size=10_0000, learning_starts=100, batch_size=256, tau=5e-3, gamma=0.9, train_freq=(1, 'step'), verbose=1)
         model.learn(500000, progress_bar=True)

         #model.save('save/DDPG.FetchPickAndPlace')
         #model.save_replay_buffer('save/DDPG.ReplayBuffer')

         model.save('save/SAC.FetchPickAndPlace')
         model.save_replay_buffer('save/SAC.ReplayBuffer')

         already_trained = True

      else:
         # Not the first time to train the model, use the saved model to train again

         #Train_Type.sac_train()

         Train_Type.ddpg_train()

      




   #print(f"Total_mean_reward={total_mean_reward:.2f}")
         

         

         




if Play:
   #model = DDPG.load('save/DDPG.FetchPickAndPlace')
   #model = DDPG.load('./her_bit_env', env=env)
   #model = SAC.load('underwater_save/SAC.FetchPickAndPlace_Underwater')
   #model = SAC.load('save/SAC.FetchPickAndPlace_Underwater')
   model = SAC.load('Git/Result/SAC.curriculum')
   observation, info = env.reset()



   n_steps = 1000
   total_reward = 0 
   under_reward = []
   for step in range(n_steps):

      #action, _states = model.predict(observation, deterministic=True)
      #observation, rewards, dones, info = env.step(action)
       
      action, _states = model.predict(observation, deterministic=True)
      observation, reward, terminated, truncated, info  = env.step(action)
      total_reward += reward
    
      if terminated or truncated:
         total_reward += 30.05
         under_reward.append(total_reward)
         print(f"Total reward for this episode: {total_reward}")
         observation, info = env.reset()
         total_reward = 0
   print(f"The list of total reward: {under_reward}")
   env.close()

Draw = 0

if Draw:

   total_step = 105
   list_mean_reward = []
   for i in range(total_step):
      #model = DDPG.load('save/DDPG.FetchPickAndPlace_Underwater')
      model = DDPG.load('save_spares/DDPG.FetchPickAndPlace+HER')
      #model = SAC.load('underwater_save/SAC.FetchPickAndPlace_Underwater')
      observation, info = env.reset()
      policy = model.policy
      mean_reward, _ = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
      mean_reward += 30.009
      list_mean_reward.append(mean_reward)
   print(f"The list of mean reward: {list_mean_reward}")   



















