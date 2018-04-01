import copy
import os
import pprint

import numpy as np
import tensorflow as tf

from dfp import util as my_util
from dfp.agent_multimodal_advantage import AgentMultimodalAdvantage
from dfp.future_target_maker import FutureTargetMaker
from dfp.experience_memory import ExperienceMemory
from dfp.multi_simulator import MultiSimulator


class Experiment:
    def __init__(self, target_maker_args, simulator_args, train_experience_args, test_experience_args,
                 agent_args, experiment_args, my_rand):
        self.logdir = simulator_args['logdir']
        self.target_maker = FutureTargetMaker(target_maker_args)
        self.show_predictions = experiment_args['show_predictions']
        agent_args['target_dim'] = self.target_maker.target_dim
        agent_args['target_names'] = self.target_maker.target_names
        agent_args['num_steps_per_policy_test'] = test_experience_args['memory_capacity'] / simulator_args['num_simulators']

        # replicate simulator args
        replicated_simulator_args = []
        for i in range(simulator_args['num_simulators']):
            simargs = copy.deepcopy(simulator_args)
            simargs['id'] = 'sim%02d' % i
            simargs['logdir'] = os.path.join(simargs['logdir'], simargs['id'])
            simargs['seed'] = my_rand.random()
            replicated_simulator_args.append(simargs)
        self.train_multi_simulator = MultiSimulator(replicated_simulator_args)
        test_sim_args = copy.deepcopy(simulator_args)
        test_sim_args['num_simulators'] = 1
        test_sim_args['id'] = 'sim10'
        test_sim_args['logdir'] = os.path.join(test_sim_args['logdir'], test_sim_args['id'])
        test_sim_args['seed'] = my_rand.random()
        test_sim_args['log_action_trace'] = True
        self.test_multi_simulator = MultiSimulator([test_sim_args])

        agent_args['discrete_controls'] = self.train_multi_simulator.discrete_controls
        agent_args['continuous_controls'] = self.train_multi_simulator.continuous_controls

        agent_args['objective_indices'], agent_args['objective_coeffs'] = my_util.make_objective_indices_and_coeffs(agent_args['objective_coeffs_temporal'],
                                                                                                                    agent_args['objective_coeffs_meas'])

        train_experience_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        test_experience_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        self.train_experience = ExperienceMemory(train_experience_args, multi_simulator=self.train_multi_simulator,
                                                 target_maker=self.target_maker)
        agent_args['state_sensory_shapes'] = self.train_experience.state_sensory_shapes
        agent_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        agent_args['num_simulators'] = self.train_multi_simulator.num_simulators

        # this seems to be unused now and may not work well. Can be needed if we use some of the measurements for the net and others for a hand-designed controller.
        if 'meas_for_net' in experiment_args:
            agent_args['meas_for_net'] = []
            for ns in range(self.train_experience.history_lengths['measurements']):
                agent_args['meas_for_net'] += [i + self.train_multi_simulator.num_meas * ns for i in experiment_args['meas_for_net']] # we want these measurements from all timesteps
            agent_args['meas_for_net'] = np.array(agent_args['meas_for_net'])
        else:
            agent_args['meas_for_net'] = np.arange(self.train_experience.state_sensory_shapes['measurements'][0])
        if len(experiment_args['meas_for_manual']) > 0:
            agent_args['meas_for_manual'] = np.array([i + self.train_multi_simulator.num_meas*(self.train_experience.history_lengths['measurements']-1) for i in experiment_args['meas_for_manual']]) # current timestep is the last in the stack
        else:
            agent_args['meas_for_manual'] = []
        agent_args['state_sensory_shapes']['measurements'] = (len(agent_args['meas_for_net']),)
        self.agent_type = agent_args['agent_type']

        if agent_args['random_objective_coeffs']:
            assert('fc_obj_params' in agent_args)

        self.test_experience = ExperienceMemory(test_experience_args, multi_simulator=self.test_multi_simulator,
                                                target_maker=self.target_maker)

        if simulator_args['gpu']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # avoid using all gpu memory
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
        else:
            self.sess = tf.Session(config=tf.ConfigProto())

        if self.agent_type == 'advantage':
            self.ag = AgentMultimodalAdvantage(self.sess, agent_args)
        else:
            raise Exception('Unknown agent type', self.agent_type)

        self.num_train_iterations = experiment_args['num_train_iterations']
        _, self.test_objective_coeffs = my_util.make_objective_indices_and_coeffs(experiment_args['test_objective_coeffs_temporal'],
                                                                                  experiment_args['test_objective_coeffs_meas'])
        self.test_random_prob = experiment_args['test_random_prob']
        self.test_checkpoint = experiment_args['test_checkpoint']
        self.test_policy_num_steps = experiment_args['test_policy_num_steps']

        print('Initialized experiment with PARAMS:')
        all_args = {'target_maker_args': target_maker_args,
                    'simulator_args': simulator_args,
                    'train_experience_args': train_experience_args,
                    'test_experience_args': test_experience_args,
                    'agent_args': agent_args,
                    'experiment_args': experiment_args}
        pprint.pprint(all_args)

    def run(self, args):
        test_dataset = args.get('test_dataset', 'val')
        if args['mode'] == 'train':
            self.test_experience.log_prefix = os.path.join(self.logdir, 'log')
            self.ag.train(self.train_multi_simulator, self.train_experience, self.num_train_iterations,
                          test_simulator=self.test_multi_simulator, test_experience=self.test_experience,
                          test_random_prob=self.test_random_prob, test_dataset=test_dataset)
        elif args['mode'] == 'show':
            if not self.ag.load(self.test_checkpoint):
                raise Exception('Could not load the checkpoint ', self.test_checkpoint)
            self.test_experience.head_offset = self.test_policy_num_steps + 1
            self.test_experience.log_prefix = 'logs/log_test'
            num_steps = self.ag.test_policy(self.train_multi_simulator, self.test_experience,
                                            self.test_objective_coeffs,
                                            self.test_policy_num_steps,
                                            random_prob=self.test_random_prob, write_summary=False,
                                            write_predictions=True, test_dataset=test_dataset)
            self.test_experience.show(start_index=0,
                                      end_index=num_steps,
                                      display=True, write_imgs=False,
                                      show_predictions=self.show_predictions,
                                      net_discrete_actions = self.ag.net_discrete_actions)
        elif args['mode'] == 'test':
            self.test_experience.log_prefix = 'logs/log_test'
            all_checkpoints = my_util.list_checkpoints(self.test_checkpoint, args['test_checkpoint_range'])
            for step in sorted(all_checkpoints):
                ckpt = all_checkpoints[step]
                self.ag.load_checkpoint(ckpt)
                num_steps = self.ag.test_policy(self.train_multi_simulator, self.test_experience,
                                                self.test_objective_coeffs,
                                                num_steps=self.test_policy_num_steps,
                                                random_prob=self.test_random_prob, write_summary=False,
                                                write_predictions=False, test_dataset=test_dataset)
                if 'save_video' in args and args['save_video'] is not None:
                    videofile = args['save_video'] + '.%s.ckpt-%d.mp4' % (test_dataset, step)
                    print('Writing video to ' + videofile + ' ...')
                    self.test_experience.write_video(videofile, start_index=0, end_index=num_steps)
        else:
            print('Unknown mode', args['mode'])

    def stop(self):
        self.train_multi_simulator.close()
        self.test_multi_simulator.close()
