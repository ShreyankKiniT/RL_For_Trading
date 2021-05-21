# Importing packages 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

#Data Ingest
df = pd.read_csv('D:/DATASCIENCE/Python/RL_FOR_TRADING/Data/tesla.csv')
#print(df.head(5))

#print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'])
#print(df.dtypes)

#setting dtae column as index
df.set_index('Date',inplace=True)
print(df.head(10),"\n")

env = gym.make('stocks-v0',df=df, frame_bound=(5,100), window_size=5) #5th to 100th time step with batch size as 5

print(env.prices,"\n")
print(env.signal_features,"\n")

#Build Env
env.action_space

state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


#Build & Train Environment
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C('MlpPolicy', env, verbose=1) 
model.learn(total_timesteps=1000000)

# Test & Evaluation
env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break


plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

# download crypto data and replace to see the FALL lol :(
    # https://www.marketwatch.com/investing/cryptocurrency/btcusd/download-data?mod=mw_quote_tab

