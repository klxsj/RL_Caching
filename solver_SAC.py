import gym
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from env.cache_env import cache_env

import pandas as pd

df = pd.read_csv('./data/requests.csv')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: cache_env(df)])

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=40000)

#
model.save("SAC_new")
#%% 
#
# load model
#model = SAC.load("SAC")
obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
