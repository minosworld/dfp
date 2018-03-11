# Shared experiment functions
import atexit
import argparse
import random
import os

from dfp.experiment import Experiment
from dfp import defaults
from dfp import util

from minos.lib.util.measures import *
from minos.config.sim_config import resolve_relative_path, get_scene_params


def run_exp_parse_args():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('mode', help='Experiment mode (train or show)')
    parser.add_argument('--save_video',
                        default=None, type=str,
                        help='Video filename to save frames to')
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
    parser.add_argument('--log_action_trace', action='store_true',
                        default=False,
                        help='Whether to log action and state traces')
    args = parser.parse_args()
    if args.log_action_trace:
        args.actionTraceLogFields = ['forces']
    return args


def start_experiment(gdict):
    my_rand = random.Random(123456789)
    gdict['simulator_args'] = util.merge_two_dicts(gdict['simulator_args'], vars(gdict['args']))
    gdict['test_experience_args'] = gdict['train_experience_args'].copy()

    # merge experiment params with default values
    target_maker_args = util.merge_two_dicts(defaults.target_maker_args, gdict['target_maker_args'])
    simulator_args = util.merge_two_dicts(defaults.simulator_args, gdict['simulator_args'])
    train_experience_args = util.merge_two_dicts(defaults.train_experience_args, gdict['train_experience_args'])
    test_experience_args = util.merge_two_dicts(defaults.test_experience_args, gdict['test_experience_args'])
    agent_args = util.merge_two_dicts(defaults.agent_args, gdict['agent_args'])
    experiment_args = util.merge_two_dicts(defaults.experiment_args, gdict['experiment_args'])

    # some augmentation / copying between args
    simulator_args['scenes_file'] = resolve_relative_path(simulator_args['scenes_file'])
    simulator_args['states_file'] = resolve_relative_path(simulator_args['states_file'])
    simulator_args['roomtypes_file'] = resolve_relative_path(simulator_args['roomtypes_file'])
    simulator_args['logdir'] = os.path.join(agent_args['log_dir'], agent_args['model_dir'])
    simulator_args['modalities'] = agent_args['modalities']
    simulator_args['outputs'] = agent_args['modalities'][:] + ['rewards', 'terminals']

    experiment = Experiment(
        target_maker_args=target_maker_args,
        simulator_args=simulator_args,
        train_experience_args=train_experience_args,
        test_experience_args=test_experience_args,
        agent_args=agent_args,
        experiment_args=experiment_args,
        my_rand=my_rand)
    atexit.register(experiment.stop)
    experiment.run(gdict['args'])
