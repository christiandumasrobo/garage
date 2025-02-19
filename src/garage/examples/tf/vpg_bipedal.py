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
from garage.tf.algos import VPG
from garage.tf.algos import ERWR
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy
from garage.trainer import TFTrainer
from garage.tf.optimizers import LBFGSOptimizer
import json
import sys

global_params = None

@wrap_experiment
def vpg_bipedal(ctxt=None, seed=1):
    """Train with ERWR on CartPole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('BipedalWalker-v3')

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(256, 256, 256))

        '''
        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))
                                      '''

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo=algo, env=env)

        trainer.train(n_epochs=100, batch_size=10000, plot=False)


vpg_bipedal(seed=1)

