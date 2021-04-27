#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
import argparse

from garage.envs import GymEnv
from garage.trainer import TFTrainer
from garage.experiment import Snapshotter
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('file_path',
        type=str,
        help='The file path to load from')
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load(args.file_path)
    policy = data['algo'].policy

    # You can also access other components of the experiment
    env = data['env']
    #env = GymEnv('LunarLander-v2')
    from garage import rollout
    rollout(env, policy, animated=True)
    exit()

    steps, max_steps = 0, 150
    done = False
    obs = env.reset()  # The initial observation
    policy.reset()

    while steps < max_steps and not done:
        obs, rew, done, _ = env.step(policy.get_action(obs))
        env.render()  # Render the environment to see what's going on (optional)
        steps += 1

    env.close()
