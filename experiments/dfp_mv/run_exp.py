#!/usr/bin/env python3
import sys
import numpy as np
sys.path = ['../..'] + sys.path
from dfp import expt_common

simulator_args = expt_common.run_exp_parse_args()

target_maker_args = {
    'future_steps': [1, 2, 4, 8, 16, 32],
    'meas_to_predict': list(range(simulator_args['measure_fun'].num_meas))
}

targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([1., 1., 1., 1.]), 1)
                                    * np.ones((1, len(target_maker_args['future_steps'])))).flatten(), 0)
agent_args = {
    'modalities': ['color', 'measurements'],
    'preprocess_input_targets': lambda x: x / targ_scale_coeffs,
    'postprocess_predictions': lambda x: x * targ_scale_coeffs,
    'objective_coeffs_meas': np.array([-1., 0., 0., -1.])
}

train_experience_args = {
    'default_history_length': 4,
    'history_step': 1,
    'history_lengths': {'actions': 12}
}

expt_common.start_experiment(globals())
