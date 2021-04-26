#!/usr/bin/env python3
"""This is an example to train a task with ERWR algorithm.

Here it runs CartpoleEnv on ERWR with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 34
"""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import ERWR
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy
from garage.trainer import TFTrainer
from garage.tf.optimizers import LBFGSOptimizer
import json
import sys

global_params = None

@wrap_experiment
def pre_trained_erwr_bipedal(ctxt=None,
    snapshot_dir=sys.argv[1],
    seed=1):
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(snapshot_dir)
        trainer.resume(n_epochs=30, batch_size=10000)
        print('Success')

pre_trained_erwr_bipedal()
