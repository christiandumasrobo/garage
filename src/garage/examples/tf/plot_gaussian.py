import csv
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

def plot_run(progress_file):
    total_episodes = 0
    episodes = []
    total_reward = 0
    rewards = []
    with open(progress_file, 'r') as csv_file:
        csv_data = read_csv(csv_file)
        first_row = True
        for episode_count in csv_data['Evaluation/NumEpisodes']:
            num_episodes = float(episode_count)
            total_episodes += num_episodes
            episodes.append(total_episodes)
        #for av_reward in csv_data['Evaluation/AverageReturn']:
        for av_reward in csv_data['Evaluation/MaxReturn']:
            reward = float(av_reward)
            total_reward += reward
            rewards.append(reward)
    return episodes, rewards


episode_list = []
reward_list = []

experiments_dir = sys.argv[1] + '/local/experiment/'

for directory in os.listdir(experiments_dir):
    #if not sys.argv[2] in directory:
        #continue
    episodes, rewards = plot_run(experiments_dir + directory + '/progress.csv')
    episode_list.append(episodes)
    reward_list.append(rewards)

el = np.array(episode_list)
rl = np.array(reward_list)

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(el, rl)
rl_pred, sigma = gp.predict(el, return_std=True)
#import pdb; pdb.set_trace()

plt.figure()
el = el.mean(axis=0)
rlm = rl.mean(axis=0)
rls = rl.std(axis=0)
plt.plot(el, rlm)
plt.title(sys.argv[2] + ' average sample efficiency')
plt.xlabel('Total episodes learning')
plt.ylabel('Reward per episode')

std_factor = 1.0

plt.fill(np.concatenate([el, el[::-1]]),
         np.concatenate([rlm - std_factor * rls,
                        (rlm + std_factor * rls)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.show()
exit()
