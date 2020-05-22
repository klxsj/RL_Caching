import gym

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from pandas.plotting import register_matplotlib_converters
from stable_baselines.common.vec_env import DummyVecEnv
register_matplotlib_converters()

from env.cache_env import cache_env

import pandas as pd

df = pd.read_csv('./data/requests.csv')

# env = cache_env(df)

env = DummyVecEnv([lambda: cache_env(df)])

model = A2C('MlpLstmPolicy', env, gamma= 0.88, n_steps= 18, learning_rate=0.004, lr_schedule='linear')
# model.load('A2C_cache.zip')
#%%
model.learn(total_timesteps=40000)
# model.save('A2C_optimize')
#%%
#model = A2C('MlpPolicy', env)
#model.load('A2C_cache.zip')

#%%
#r=[]
#rewards=[]

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #r.append(rewards)
    env.render()

#import matplotlib.pyplot as plt
#plt.plot(r)