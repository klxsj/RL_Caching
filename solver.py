import numpy as np
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from env.cache_env import cache_env

import pandas as pd
r = []

df = pd.read_csv('./data/requests.csv')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: cache_env(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.load('ppo2_positive_rewards.zip')
#%%
#model.learn(total_timesteps=40000)
#%%


#%%
model.learn(total_timesteps=100000)
model.save("PPO2_new")
#%%
r=[]
obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    r.append(rewards)
    env.render()

#%%
import matplotlib.pyplot as plt
plt.plot(r)

r2= np.array(r)
r2 = np.delete(r2, 0)
r2= np.append(r2,0)
r2= r2.reshape(-1,1)
print(r2 - r)