# Shared experiment functions
import atexit
import argparse
import random
import os

from dfp.experiment import Experiment
from dfp import defaults
from dfp import util

from minos.lib.util.measures import *
from minos.config import sim_config
from minos.config.sim_args import add_sim_args


def run_exp_parse_args():
    parser = argparse.ArgumentParser(description='MINOS DFP experiment')
    add_sim_args(parser)
    parser.add_argument('mode', help='Experiment mode (train|test|show)')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
    parser.add_argument('--test_checkpoint', type=str,
                        default='',
                        help='Checkpoint directory to load for testing')
    parser.add_argument('--test_checkpoint_range', type=str,
                        default=None,
                        help='Checkpoint range filter: e.g. [100000-150000]')
    parser.add_argument('--test_dataset', type=str,
                        default='val',
                        help='Dataset on which to perform testing')
    parser.add_argument('--episodes_per_scene_test', type=int,
                        default=1,
                        help='Episodes per scene during testing')
    parser.set_defaults(width=84, height=84, color_encoding='gray', num_simulators=4)
    args = parser.parse_args()
    if args.mode == 'test' or args.mode == 'show':
        args.num_simulators = 1
    if args.log_action_trace:
        args.actionTraceLogFields = ['forces']
    args = sim_config.get(args.env_config, vars(args))
    return args


def start_experiment(gdict):
    my_rand = random.Random(123456789)
    gdict['test_experience_args'] = gdict['train_experience_args'].copy()

    # merge experiment params with default values
    target_maker_args = util.merge_two_dicts(defaults.target_maker_args, gdict['target_maker_args'])
    train_experience_args = util.merge_two_dicts(defaults.train_experience_args, gdict['train_experience_args'])
    test_experience_args = util.merge_two_dicts(defaults.test_experience_args, gdict['test_experience_args'])
    agent_args = util.merge_two_dicts(defaults.agent_args, gdict['agent_args'])
    experiment_args = defaults.experiment_args

    # some augmentation / copying between args
    simulator_args = gdict['simulator_args']
    agent_args['log_dir'] = os.path.split(simulator_args['logdir'])[-2]
    agent_args['model_dir'] = os.path.split(simulator_args['logdir'])[-1]
    simulator_args['modalities'] = agent_args['modalities']
    simulator_args['outputs'] = agent_args['modalities'][:] + ['rewards', 'terminals']
    experiment_args['test_objective_coeffs_meas'] = agent_args['objective_coeffs_meas'],
    experiment_args['test_checkpoint'] = simulator_args['test_checkpoint']

    experiment = Experiment(
        target_maker_args=target_maker_args,
        simulator_args=simulator_args,
        train_experience_args=train_experience_args,
        test_experience_args=test_experience_args,
        agent_args=agent_args,
        experiment_args=experiment_args,
        my_rand=my_rand)
    atexit.register(experiment.stop)
    experiment.run(gdict['simulator_args'])
