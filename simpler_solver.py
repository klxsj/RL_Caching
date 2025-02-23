import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from pandas.plotting import register_matplotlib_converters
from stable_baselines.common.vec_env import DummyVecEnv
register_matplotlib_converters()

from env.cache_env import cache_env

import pandas as pd

df = pd.read_csv('./data/requests.csv')

env = DummyVecEnv([lambda: cache_env(df)])

#%%
model = A2C('MlpPolicy', env)
model.learn(total_timesteps=20000)
model.save('A2C_new')
#%%
#model = A2C('MlpPolicy', env)
#model.load('A2C_cache.zip')

#%%
#r=[]
#rewards=[]

obs = env.reset()
for i in range(65):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #r.append(rewards)
    env.render()

#import matplotlib.pyplot as plt
#plt.plot(r)