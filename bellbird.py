from lake_rl_daily import Lake as Lake
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
import sys


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.booked_return = 0

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/booked_return", self.booked_return)
        self.booked_return = 0

    def _on_step(self) -> bool:
        self.booked_return += self.training_env.get_attr("booked_return")[0]
        return True


env = Lake()

check_env(env)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=3e-5, seed=11) # ent_coef=0.01,
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 50000
iters = 0

returns_callback = TensorboardCallback()

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=returns_callback)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
    model.env.render()
