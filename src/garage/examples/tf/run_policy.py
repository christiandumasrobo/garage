#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
import argparse

from garage.envs import GymEnv
from garage.trainer import TFTrainer

parser = argparse.ArgumentParser()
parser.add_argument('file_path',
        type=str,
        help='The file path to load from')
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = GymEnv('LunarLander-v2')

# Reset the environment and launch the viewer
env.reset()
env.visualize()

step_count = 0
es = env.step(env.action_space.sample())

class snp:
    def __init__(self):
        self.snapshot_dir = args.file_path
        self.snapshot_mode = 'last'
        self.snapshot_gap = 1

with TFTrainer(snapshot_config=snp()) as trainer:
    trainer.restore(args.file_path)
    while not es.last and step_count < args.n_steps:
        es = env.step(env.action_space.sample())
        step_count += 1

env.close()
