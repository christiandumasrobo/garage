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

maxl = 0
for av in re_avs:
    maxl = max(maxl, len(av))
orig_lens = []
for i in range(len(re_avs)):
    orig_lens.append(len(ep_avs[i]))
    ep_avs[i] = np.resize(ep_avs[i], maxl)
    re_avs[i] = np.resize(re_avs[i], maxl)
#import pdb; pdb.set_trace()
# Construct thresholds
xmin = 115
xmax = 10000
line_seg_x = np.array([xmin, xmax])
thresholds = []
minr = np.min(re_avs)
maxr = np.max(re_avs)
def ratiofy(mnr, mxr, rat):
    return (mxr - mnr) * rat + mnr
fracs = []
frac_names = []
for frac in np.arange(0.2, 1.0, 0.2):
    fracs.append(frac)
    frac_names.append(str(round(frac, 1)))
    thresholds.append(ratiofy(minr, maxr, frac))

# Plot thresholds
for threshold in thresholds:
    plt.plot(line_seg_x, np.array(2 * [threshold]), 'k')

first_crosses = []
first_crosses_unt = []
for i in range(len(ep_avs)):
    # Get the crossing points of the performance thresholds in # of episodes
    first_crosses.append([])
    first_crosses_unt.append([])
    for j in range(len(thresholds)):
        threshold = thresholds[j]
        thresh_frac = fracs[j]
        for ep in range(len(ep_avs[i])):
            if re_avs[i][ep] > threshold and ep_avs[i][ep] > xmin:
                first_crosses[i].append(re_avs[i][ep] / np.log(ep_avs[i][ep]))
                first_crosses_unt[i].append(ep_avs[i][ep])
                #first_crosses[i].append(1 / np.log(ep_avs[i][ep]))
                print(ex_names[i], thresh_frac, threshold, ep_avs[i][ep])
                break

    # Plot reward segments
    plt.plot(ep_avs[i][:orig_lens[i]], re_avs[i][:orig_lens[i]], label=ex_names[i])

# Formatting
plt.legend()
#plt.title(sys.argv[2] + ' task performance')
plt.xlabel('Cumulative Samples (episodes)')
plt.ylabel('Reward per episode')
plt.xlim([xmin, xmax])
plt.xscale('log')
#plt.show()

fig, ax = plt.subplots()
max_crosses = -1
for cross in first_crosses:
    max_crosses = max(len(cross), max_crosses)
#print(max_crosses)
xvals = np.array(list(range(max_crosses)))
width = 0.1
for i, cross in enumerate(first_crosses):
    if len(cross) > 0:
        ax.bar(xvals[:len(cross)] + i * width, cross, width, label=ex_names[i])
    print(ex_names[i], cross)
ax.set_xticks(xvals)
ax.set_xticklabels(frac_names)
#import pdb; pdb.set_trace()
ax.legend()
ax.set_xlabel('Performance Percentage')
ax.set_ylabel('Reward Scaled Inverse Log Sample Count')
#ax.set_title(sys.argv[2] + ' task sample efficiency')
#plt.show()


# Print a table of the crossover points
for i in range(max_crosses):
    print(frac_names[i], '&', end=' ')
print('\\\\\n\\hline')
for i, cross in enumerate(first_crosses_unt):
    print(ex_names[i], '&', end=' ')
    for cr_i, cr in enumerate(cross):
        print(int(cr), '&', end=' ')
    print('\\\\\n\\hline')
