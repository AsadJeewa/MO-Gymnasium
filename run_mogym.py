import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import matplotlib.pyplot as plt

# ASAD DEBUG START
import os
import sys
os.environ["SDL_VIDEODRIVER"] = "x11"
#ASAD DEBUG END

# It follows the original Gymnasium API ...
env = mo_gym.make('four-room-v0',render_mode="human") 
# env = mo_gym.make('mo-reacher-v4',render_mode="rgb_array") #mo-reacher-v4 based on reacher-v4
# Optionally, you can scalarize the reward function with the LinearReward wrapper
#env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))

obs, info = env.reset()

observation, info = env.reset(seed=42)
for _ in range(1000000):
   #print(_)
   # render = lambda : plt.imshow(env.render(mode='rgb_array'))
   # plt.show()
   # render()
   env.render()
   action = env.action_space.sample()  # this is where you would insert your policy
   # but vector_reward is a numpy array!
   next_obs, vector_reward, terminated, truncated, info = env.step(action)   
   print(vector_reward)
   # exit()
   if terminated or truncated:
      print("TERM OR TRUNC")
      obs, info = env.reset()
env.close()


