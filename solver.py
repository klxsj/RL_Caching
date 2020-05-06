import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from env.cache_env import cache_env

import pandas as pd

df = pd.read_csv('./data/requests.csv')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: cache_env(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=40000)

obs = env.reset()
for i in range(20000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
