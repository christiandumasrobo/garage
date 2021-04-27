import csv
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
import os
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



ep_avs = []
re_avs = []
ex_names = []
for experiment in os.listdir(sys.argv[1]):
    if sys.argv[2] not in experiment:
        continue
    episode_list = []
    reward_list = []
    experiments_dir = sys.argv[1] + experiment + '/local/experiment/'

    for directory in os.listdir(experiments_dir):
        #if not sys.argv[2] in directory:
            #continue
        episodes, rewards = plot_run(experiments_dir + directory + '/progress.csv')
        episode_list.append(episodes)
        reward_list.append(rewards)
    ep_avs.append(np.array(episode_list).mean(axis=0))
    re_avs.append(np.array(reward_list).mean(axis=0))
    ex_names.append(experiment)

for i in range(len(ep_avs)):
    plt.plot(ep_avs[i], re_avs[i], label=ex_names[i])
plt.legend()
plt.title('Sample Efficiency of the ' + sys.argv[2] + ' task')
plt.xlabel('Total episodes learning')
plt.ylabel('Reward per episode')
plt.xscale('log')
plt.show()
