import gym
from stable_baselines3 import PPO
from lake_rl_rollover import Lake
from lake_rl_daily import Lake as Daily_Lake

env = Daily_Lake()
env.reset()


model = PPO.load(r'C:\Users\stefa\PycharmProjects\Loon\models\1674530838\200000.zip', env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)



