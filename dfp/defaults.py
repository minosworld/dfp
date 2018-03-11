import time

import numpy as np


## Target maker
target_maker_args = {}
#target_maker_args['future_steps'] = [1,2,4,8,16,32]
#target_maker_args['meas_to_predict'] = [0,1,2]
target_maker_args['min_num_targs'] = 3
target_maker_args['rwrd_schedule_type'] = 'exp'
target_maker_args['gammas'] = []
#target_maker_args['invalid_targets_replacement'] = 'nan'
target_maker_args['invalid_targets_replacement'] = 'last_valid'

## Simulator
simulator_args = {}
simulator_args['task'] = 'room_goal'
simulator_args['host'] = 'localhost'
simulator_args['log_action_trace'] = False
#simulator_args['goal'] = {'categories': ['arch', 'door'], 'select': 'random'}
simulator_args['goal'] = {'roomTypes': 'any', 'select': 'random'}
simulator_args['simulator'] = 'room_simulator'
simulator_args['auto_start'] = True
simulator_args['config'] = ''
simulator_args['width'] = 84
simulator_args['height'] = 84
simulator_args['frame_skip'] = 1
simulator_args['collision_detection'] = {'mode': 'navgrid'}
simulator_args['color_mode'] = 'GRAY'
simulator_args['maps'] = ['MAP01']
simulator_args['switch_maps'] = False
simulator_args['num_simulators'] = 1
simulator_args['num_episodes_per_restart'] = 1000
simulator_args['num_episodes_per_scene'] = 10
simulator_args['game_args'] = ""
simulator_args['scenes_file'] = '../data/scenes.multiroom.csv'
#simulator_args['states_files'] = {'train': '', 'val': '../data/episode_states.val.csv', 'test': '../data/episode_states.test.csv'}
simulator_args['states_file'] = '../data/episode_states.suncg.csv.bz2'
simulator_args['roomtypes_file'] = '../data/roomTypes.suncg.csv'
simulator_args['max_states_per_scene'] = 1
simulator_args['navmap'] = {'refineGrid': True, 'autoUpdate': True, 'allowDiagonalMoves': True, 'reverseEdgeOrder': False},
simulator_args['reward_type'] = 'dist_time'

## Experience
# Train experience
train_experience_args = {}
train_experience_args['memory_capacity'] = 10000  # TODO: automatically set as num_simulators*2500
train_experience_args['default_history_length'] = 1
train_experience_args['history_lengths'] = {}
train_experience_args['history_step'] = 1
train_experience_args['action_format'] = 'enumerate'
train_experience_args['shared'] = False
train_experience_args['meas_statistics_gamma'] = 0.
train_experience_args['num_prev_acts_to_return'] = 0

# Test policy experience
test_experience_args = train_experience_args.copy()
test_experience_args['memory_capacity'] = 60000  # NOTE has to be more than maximum possible test policy steps
#test_experience_args['memory_capacity'] = 20000

## Agent
agent_args = {}

# agent type
agent_args['agent_type'] = 'advantage'

# preprocessing
agent_args['preprocess_sensory'] = {'color': lambda x: x / 255. - 0.5,
                                    'measurements': lambda x: x,
                                    'audio': lambda x: x / 255. - 0.5,
                                    'audiopath': lambda x: x,
                                    'force': lambda x: x / 700,
                                    'actions': lambda x: x,
                                    'depth': lambda x: x / 10.0 - 0.5,
                                    'roomType': lambda x: x,
                                    'goalRoomType': lambda x: x}
agent_args['preprocess_input_targets'] = lambda x: x
agent_args['postprocess_predictions'] = lambda x: x
agent_args['discrete_controls_manual'] = []
agent_args['opposite_button_pairs'] = [(0,1)]
agent_args['onehot_actions_only'] = True

# agent properties
agent_args['objective_coeffs_temporal'] = np.array([0., 0., 0., 0.5, 0.5, 1.])
agent_args['new_memories_per_batch'] = 8
agent_args['add_experiences_every'] = 1
agent_args['random_objective_coeffs'] = False
agent_args['objective_coeffs_distribution'] = 'none'
agent_args['random_exploration_schedule'] = lambda step: (0.02 + 72500. / (float(step) + 75000.))

# optimization parameters
agent_args['batch_size'] = 64
agent_args['init_learning_rate'] = 0.0002
#agent_args['lr_step_size'] = 300000
agent_args['lr_step_size'] = 125000
agent_args['lr_decay_factor'] = 0.3
agent_args['adam_beta1'] = 0.95
agent_args['adam_epsilon'] = 1e-4
agent_args['optimizer'] = 'Adam'
agent_args['reset_iter_count'] = True
agent_args['clip_gradient'] = 0

# directories
agent_args['checkpoint_dir'] = 'checkpoints'
agent_args['init_model'] = ''
agent_args['model_name'] = "predictor.model"

# logging and testing
agent_args['print_err_every'] = 50
agent_args['detailed_summary_every'] = 1000
agent_args['test_policy_every'] = 10000
agent_args['checkpoint_every'] = 10000
agent_args['save_param_histograms_every'] = 5000
agent_args['test_random_policy_before_training'] = True

# net parameters
agent_args['img_conv_params'] = np.array([(32,8,4), (64,4,2), (64,3,1)], dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
agent_args['img_fc_params']   = np.array([(512,)], dtype = [('out_dims',int)])
agent_args['depth_conv_params'] = np.array([(32,8,4), (64,4,2), (64,3,1)], dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
agent_args['depth_fc_params']   = np.array([(512,)], dtype = [('out_dims',int)])
agent_args['audio_fc_params']   = np.array([(512,)], dtype = [('out_dims',int)])
agent_args['goalroomtype_fc_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)])
agent_args['roomtype_fc_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)])
agent_args['infer_roomtype_fc_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
agent_args['actions_fc_params']   = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)])
agent_args['obj_fc_params']  = None
#np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
agent_args['meas_fc_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
agent_args['force_fc_params'] = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)])
agent_args['audiopath_fc_params'] = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)])
agent_args['joint_fc_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
agent_args['infer_meas_fc_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
agent_args['weight_decay'] = 0.00000

## Experiment
experiment_args = {}
experiment_args['num_train_iterations'] = 410000 #820000
experiment_args['test_random_prob'] = 0.1
experiment_args['test_policy_num_steps'] = 1000
experiment_args['test_objective_coeffs_temporal'] = agent_args['objective_coeffs_temporal']
experiment_args['show_predictions'] = True
experiment_args['meas_for_manual'] = [] # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7]
