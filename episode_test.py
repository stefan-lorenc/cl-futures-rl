from lake_rl_rollover import Lake as Roll_Lake
from lake_rl_daily import Lake as Daily_Lake
from tqdm import tqdm



# env = Roll_Lake()
env = Daily_Lake()

episodes = 1

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:
		random_action = env.action_space.sample()
		obs, reward, done, info = env.step(0)
		print(reward, info['max_p_l'], info['cur_p_l'])
	env.render()
		# print(random_action, reward, info['account bal'], info['openpl'], info['curpr'], info['pospr'])