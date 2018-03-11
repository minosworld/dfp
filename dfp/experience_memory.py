# adapted from tensorflow_DQN TODO cite properly
"""
ExperienceMemory is a class for experience replay.
It stores experience samples and samples minibatches for training.
"""
import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np

import dfp.util as my_util


class ExperienceMemory:
    def __init__(self, args, multi_simulator = None, target_maker = None):
        """ Initialize empty experience dataset.
            Assumptions:
                - observations come in sequentially, and there is a terminal state in the end of each episode
                - every episode is shorter than the memory
        """

        # NOTE: the memory capacity is divided equally for each simulator. Each sim can span over the entire memory
        # capacity. Assuming that all simulators are stepped synchronously the memory span occupied by each shifts
        # predictably. If simulators are asynchronously stepped there can be collisions.
        self.capacity = int(args['memory_capacity'])
        self.default_history_length = int(args['default_history_length']) # history_lengths created after data_specs
        self.history_step = int(args['history_step'])
        self.shared = args['shared']

        self.num_heads = int(multi_simulator.num_simulators)
        self.target_maker = target_maker
        self.head_offset = int(self.capacity/self.num_heads)

        self.data_specs = {
            'color': {'type': np.uint8, 'shape': (multi_simulator.num_channels, multi_simulator.resolution[1], multi_simulator.resolution[0])},
            'depth': {'type': np.float32, 'shape': (1, multi_simulator.resolution[1], multi_simulator.resolution[0])},
            'measurements': {'type': np.float32, 'shape': (multi_simulator.num_meas,)},
            'force': {'type': np.float32, 'shape': (4,1)},   # TODO parameterize
            'audiopath': {'type': np.float32, 'shape': (8,1)},  # TODO parameterize
            'audio': {'type': np.uint8, 'shape': (1600,1)},  # TODO parameterize
            'rewards': {'type': np.float32, 'shape': ()},
            'terminals': {'type': np.bool, 'shape': ()},
            'actions': {'type': np.int8, 'shape': (multi_simulator.action_len,)},
            'roomType': {'type': np.int8, 'shape': (9,1)},  # TODO parameterize
            'goalRoomType': {'type': np.int, 'shape': (9,1)},  # TODO parameterize
            'objectives': {'type': np.float32, 'shape': args['obj_shape']}
        }

        # use the default history length if not otherwise defined
        self.history_lengths = {k: self.default_history_length for k in self.data_specs}
        self.history_lengths.update({k: int(v) for k,v in args['history_lengths'].items()})
        self.max_history_length = max(self.history_lengths.values())
        print('History lengths:', self.history_lengths)

        self.state_sensory_shapes = {
            'color': self.data_specs['color']['shape'][1:] + (self.history_lengths['color']*self.data_specs['color']['shape'][0],),
            'depth': self.data_specs['depth']['shape'][1:] + (self.history_lengths['depth']*self.data_specs['depth']['shape'][0],),
            'measurements': (self.history_lengths['measurements']*self.data_specs['measurements']['shape'][0],),
            'force': (self.history_lengths['force']*self.data_specs['force']['shape'][0],),
            'audiopath': (self.history_lengths['audiopath']*self.data_specs['audiopath']['shape'][0],),
            'audio': (self.history_lengths['audio']*self.data_specs['audio']['shape'][0],),
            'actions': (self.history_lengths['actions']*self.data_specs['actions']['shape'][0],),
            'roomType': (self.history_lengths['roomType']*self.data_specs['roomType']['shape'][0],),
            'goalRoomType': (self.history_lengths['goalRoomType']*self.data_specs['goalRoomType']['shape'][0],)
        }

        # initialize dataset
        self.reset()

    def reset(self):
        self._data = {}
        assert ('terminals' in self.data_specs), 'Experience memory has to contain terminals'
        assert ('rewards' in self.data_specs), 'Experience memory has to contain rewards'
        assert ('measurements' in self.data_specs), 'Experience memory has to contain measurements'
        for name,spec in self.data_specs.items():
            self._data[name] = my_util.make_array(shape=(self.capacity,) + spec['shape'], dtype=spec['type'], shared=self.shared, fill_val=0)

        # simulator-based episode count into memory
        self._n_episode = my_util.make_array(shape=(self.capacity,), dtype=np.uint64, shared=self.shared, fill_val=0) # this is needed to compute future targets efficiently
        # which simulator the memory is associated with
        self._n_head = my_util.make_array(shape=(self.capacity,), dtype=np.uint64, shared=self.shared, fill_val=0) # this is needed to compute future targets efficiently

        # per-simulator indices into memory
        self._curr_indices = np.arange(self.num_heads) * int(self.head_offset)
        # per-simulator episode count
        self._episode_counts = np.zeros(self.num_heads)

    def add(self, data_to_add):
        """ Add experience to dataset.

        Args:
            data: dictionary with data to be stored
        """

        for name,curr_data in data_to_add.items():
            self._data[name][self._curr_indices] = curr_data

        self._n_episode[self._curr_indices] = self._episode_counts
        self._n_head[self._curr_indices] = np.arange(self.num_heads)
        terminals = (np.array(data_to_add['terminals']) == True)
        terminals_where = np.where(terminals)[0]
        terminal_inds = self._curr_indices[terminals_where]
        self._n_episode[terminal_inds] = self._episode_counts[terminals_where]+100# so that the terminal step of the episode is not used as a target by target_maker

        self._episode_counts = self._episode_counts + (terminals)
        self._curr_indices = (self._curr_indices + 1) % self.capacity

        # TODO Alexey: I think this is not necessary any more, but have ot check
        #self._data['terminal'][self._curr_indices] = True # make the following state terminal, so that our current episode doesn't get stitched with the next one when sampling states

        return terminals

    def add_step(self, multi_simulator, acts = None, objs=None, preds=None):
        if acts == None:
            acts = multi_simulator.get_random_actions()
        data_to_add = multi_simulator.step(acts)
        data_to_add['actions'] = acts
        if not (objs is None):
            data_to_add['objectives'] = objs
        if not (preds is None):
            data_to_add['predictions'] = preds
        return self.add(data_to_add)

    def add_n_steps_with_actor(self, multi_simulator, num_steps, actor, verbose=False, write_predictions=False, write_logs = False, global_step=0):
        if write_predictions and not ('predictions' in self._data):
            self._data['predictions'] = my_util.make_array(shape=(self.capacity,) + actor.predictions_shape, dtype=np.float32, shared=self.shared, fill_val=0.)
        if verbose or write_logs:
            start_time = time.time()
        #write_logs = False
        if write_logs:
            log_dir = os.path.dirname(self.log_prefix)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_brief = open(self.log_prefix + '_brief.txt','a')
            log_detailed = open(self.log_prefix + '_detailed.txt','a')
            log_detailed.write('Step {0}\n'.format(global_step))
            start_times = time.time() * np.ones(multi_simulator.num_simulators)
            num_episode_steps = np.zeros(multi_simulator.num_simulators)
            accum_rewards = np.zeros(multi_simulator.num_simulators)
            accum_meas = np.zeros((multi_simulator.num_simulators,) + self.data_specs['measurements']['shape'])
            last_meas = np.zeros((multi_simulator.num_simulators,) + self.data_specs['measurements']['shape'])
            total_final_meas = np.zeros(self.data_specs['measurements']['shape'])
            total_avg_meas = np.zeros(self.data_specs['measurements']['shape'])
            total_accum_reward = 0
            total_start_time = time.time()
            num_episodes = 0
            meas_dim = np.prod(self.data_specs['measurements']['shape'])
            log_brief_format = ' '.join([('{' + str(n) + '}') for n in range(5)]) + ' | ' + \
                       ' '.join([('{' + str(n+5) + '}') for n in range(meas_dim)]) + ' | ' + \
                       ' '.join([('{' + str(n+5+meas_dim) + '}') for n in range(meas_dim)]) + '\n'
            log_detailed_format = ' '.join([('{' + str(n) + '}') for n in range(4)]) + ' | ' + \
                          ' '.join([('{' + str(n+4) + '}') for n in range(meas_dim)]) + ' | ' + \
                          ' '.join([('{' + str(n+4+meas_dim) + '}') for n in range(meas_dim)]) + '\n'
        for ns in range(int(num_steps)):
            if verbose and time.time() - start_time > 1:
                print('%d/%d' % (ns, num_steps))
                start_time = time.time()

            curr_act = actor.act_with_multi_memory(self)

            # actor has to return a np array of bools
            invalid_states = np.logical_not(np.array(self.curr_states_with_valid_history()))
            #print(invalid_states)
            #print(actor.random_objective_coeffs)
            if actor.random_objective_coeffs:
                actor.reset_objective_coeffs(np.where(invalid_states)[0].tolist())
            curr_act[invalid_states] = actor.random_actions(np.sum(invalid_states)) #np.array(multi_simulator.get_random_actions())[invalid_states]

            if write_predictions:
                self.add_step(multi_simulator, acts=curr_act.tolist(), objs=actor.objectives_to_write(), preds=actor.curr_predictions)
            else:
                self.add_step(multi_simulator, acts=curr_act.tolist(), objs=actor.objectives_to_write())
            if write_logs:
                last_indices = np.array(self.get_last_indices())
                last_rewards = self._data['rewards'][last_indices]
                prev_meas = last_meas
                last_meas = self._data['measurements'][last_indices]
                last_terminals = self._data['terminals'][last_indices]
                accum_rewards += last_rewards
                accum_meas += last_meas
                num_episode_steps = num_episode_steps + 1
                #print(last_terminals)
                #print(num_episode_steps)
                terminated_simulators = list(np.where(last_terminals)[0])
                for ns in terminated_simulators:
                    num_episodes += 1
                    episode_time = time.time() - start_times[ns]
                    avg_meas = accum_meas[ns]/float(num_episode_steps[ns])
                    total_avg_meas += avg_meas
                    total_final_meas += prev_meas[ns]
                    total_accum_reward += accum_rewards[ns]
                    start_times[ns] = time.time()
                    log_detailed.write(log_detailed_format.format(*([num_episodes, num_episode_steps[ns], episode_time, accum_rewards[ns]] + list(prev_meas[ns]) + list(avg_meas))))
                    accum_meas[ns] = 0
                    accum_rewards[ns] = 0
                    num_episode_steps[ns] = 0
                    start_times[ns] = time.time()
        if write_logs:
            if num_episodes == 0:
                num_episodes = 1
            log_brief.write(log_brief_format.format(*([global_step, time.time(), time.time() - total_start_time, num_episodes, total_accum_reward/float(num_episodes)] +
                                 list(total_final_meas / float(num_episodes)) + list(total_avg_meas / float(num_episodes)))))
            log_brief.close()
            log_detailed.close()

    def add_n_episodes_with_actor(self, multi_simulator, num_episodes, actor, write_predictions=False):
        if write_predictions and not ('predictions' in self._data):
            self._data['predictions'] = my_util.make_array(shape=(self.capacity,) + actor.predictions_shape,
                                                           dtype=np.float32, shared=self.shared, fill_val=0.)
        ne = 0
        num_steps = 0
        while ne < num_episodes:
            curr_act = actor.act_with_multi_memory(self)
            invalid_states = np.logical_not(np.array(self.curr_states_with_valid_history()))
            if actor.random_objective_coeffs:
                actor.reset_objective_coeffs(np.where(invalid_states)[0].tolist())
            curr_act[invalid_states] = actor.random_actions(np.sum(invalid_states)) #np.array(multi_simulator.get_random_actions())[invalid_states]

            # NOTE this doesn't really work well when multiple simulators used, need to handle when some simulators done earlier
            terminals = self.add_step(multi_simulator,
                                      acts=curr_act.tolist(),
                                      objs=actor.objectives_to_write(),
                                      preds=(actor.curr_predictions if write_predictions else None))
            num_steps += multi_simulator.num_simulators
            num_terminals = np.sum(terminals)
            ne += num_terminals
        return num_steps

    def get_states(self, indices, modalities):



        states = {}
        for modality in modalities:
            # prepare which indices to take (modality-dependent, since the history length is modality-dependent)
            history_length = self.history_lengths[modality]
            frames = np.zeros(len(indices)*history_length, dtype=np.int64)
            for (ni,index) in enumerate(indices):
                frame_slice = np.arange(int(index) - history_length*self.history_step + 1, (int(index) + 1), self.history_step) % self.capacity
                frames[ni*history_length:(ni+1)*history_length] = frame_slice
            
            if modality == 'color':
                reshape_size = (self.state_sensory_shapes['color'][2],) + self.state_sensory_shapes['color'][:2]
                states['color'] = np.transpose(np.reshape(np.take(self._data['color'], frames, axis=0),
                                                           (len(indices),) + reshape_size), [0,2,3,1]).astype(np.float32)
            if modality == 'depth':
                reshape_size = (self.state_sensory_shapes['depth'][2],) + self.state_sensory_shapes['depth'][:2]
                states['depth'] = np.transpose(np.reshape(np.take(self._data['depth'], frames, axis=0),
                                                          (len(indices),) + reshape_size), [0,2,3,1]).astype(np.float32)
            if modality == 'measurements':
                #print((len(indices),), self.state_sensory_shapes['measurements'])
                states['measurements'] = np.reshape(np.take(self._data['measurements'], frames, axis=0),
                                                    (len(indices),) + self.state_sensory_shapes['measurements']).astype(np.float32)
            if modality == 'force':
                states['force'] = np.reshape(np.take(self._data['force'], frames, axis=0),
                                             (len(indices),) + self.state_sensory_shapes['force']).astype(np.float32)
            if modality == 'audio':
                states['audio'] = np.reshape(np.take(self._data['audio'], frames, axis=0),
                                             (len(indices),) + self.state_sensory_shapes['audio']).astype(np.float32)
            if modality == 'audiopath':
                states['audiopath'] = np.reshape(np.take(self._data['audiopath'], frames, axis=0),
                                                 (len(indices),) + self.state_sensory_shapes['audiopath']).astype(np.float32)
            if modality == 'goalRoomType':
                states['goalRoomType'] = np.reshape(np.take(self._data['goalRoomType'], frames, axis=0),
                                             (len(indices),) + self.state_sensory_shapes['goalRoomType']).astype(np.int8)
            if modality == 'roomType':
                states['roomType'] = np.reshape(np.take(self._data['roomType'], frames, axis=0),
                                             (len(indices),) + self.state_sensory_shapes['roomType']).astype(np.int8)
            # In the memory at step i we store the observations from step i and actions from step i-1. Therefore it is fine to
            # take actions directly, in the same way as other modalities - there will be no "looking into the future"
            if modality == 'actions':
                states['actions'] = np.reshape(np.take(self._data['actions'], frames, axis=0),
                                                 (len(indices),) + self.state_sensory_shapes['actions']).astype(np.float32)

        return states

    def get_current_states(self, modalities):
        """  Return most recent observation sequence """
        return self.get_states(list((self._curr_indices-1)%self.capacity), modalities)

    def get_last_indices(self):
        """  Return most recent indices """
        return list((self._curr_indices-1)%self.capacity)

    def get_targets(self, indices):
        # TODO this 12345678 is a hack, but should be good enough
        return self.target_maker.make_targets(indices, self._data['measurements'], self._data['rewards'], self._n_episode + 12345678*self._n_head)

    def has_valid_history(self, index):
        prev_inds = np.arange(int(index) - self.max_history_length*self.history_step + 1, int(index)+1) % self.capacity
        return (self._n_episode[index] == self._n_episode[prev_inds]).all()

    def curr_states_with_valid_history(self):
        return [self.has_valid_history((ind - 1)%self.capacity) for ind in list(self._curr_indices)]

    def has_valid_target(self, index):
        next_inds = np.arange(index, index+self.target_maker.min_future_frames+1) % self.capacity
        return (self._n_episode[index] == self._n_episode[next_inds]).all()

    def is_valid_state(self, index):
        return self.has_valid_history(index) and self.has_valid_target(index)

    def get_observations(self, indices, modalities):
        indices_arr = np.array(indices)
        states = self.get_states((indices_arr - 1) % self.capacity, modalities)
        rwrds = self._data['rewards'][indices_arr]
        acts = self._data['actions'][indices_arr]
        terms = self._data['terminals'][indices_arr].astype(int)
        targs = self.get_targets((indices_arr - 1) % self.capacity)
        if 'objectives' in self._data:
            objs = self._data['objectives'][indices_arr]
        else:
            objs = None

        return states, rwrds, terms, acts, targs, objs

    def get_random_batch(self, batch_size, modalities):
        """ Sample minibatch of experiences for training """

        samples = [] # indices of the end of each sample

        counter = 0
        while len(samples) < batch_size:
            index = random.randrange(self.capacity)
            # check if there is enough history to make a state and enough future to make targets
            if self.is_valid_state(index):
                samples.append(index)
            else:
                counter += 1
            if counter > 1000:
                print('get_random_batch.py: Could not find a valid random batch, returning %d samples' % len(samples))
                break

        # create batch
        return self.get_observations(np.array(samples), modalities)

    def compute_avg_meas_and_rwrd(self, start_idx, end_idx):
        # compute average measurement values per episode, and average cumulative reward per episode
        if end_idx > self.capacity or start_idx > self.capacity:
            print('WARNING: Average measurement computation on overflowed memory, results inaccurate!')
        curr_num_obs = 0.
        curr_sum_meas = self._data['measurements'][0] * 0
        curr_sum_rwrd = self._data['rewards'][0] * 0
        num_episodes = 0.
        total_sum_meas = self._data['measurements'][0] * 0
        total_sum_rwrd = self._data['rewards'][0] * 0
        for index in range(int(start_idx), int(end_idx)):
            index = index % self.capacity
            curr_sum_rwrd += self._data['rewards'][index]
            if self._data['terminals'][index]:
                if curr_num_obs:
                    total_sum_meas += curr_sum_meas / curr_num_obs
                    total_sum_rwrd += curr_sum_rwrd
                    num_episodes += 1
                    #print(num_episodes,curr_sum_meas / curr_num_obs,curr_sum_rwrd)
                curr_sum_meas = self._data['measurements'][0] * 0
                curr_sum_rwrd = self._data['rewards'][0] * 0
                curr_num_obs = 0.
            else:
                curr_sum_meas += self._data['measurements'][index]
                curr_num_obs += 1

        if num_episodes == 0.:
            total_avg_meas = curr_sum_meas / curr_num_obs
            total_avg_rwrd = curr_sum_rwrd
        else:
            total_avg_meas = total_sum_meas / num_episodes
            total_avg_rwrd = total_sum_rwrd / num_episodes

        return total_avg_meas, total_avg_rwrd

    def write_video(self, filename='out.mp4', start_index=0, end_index=None):
        fig = plt.figure(figsize=(10, 10), dpi=50)
        axes = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        metadata = {'title': 'DFP-SIM', 'comment': 'comment'}
        writer = manimation.FFMpegWriter(fps=10, metadata=metadata)

        curr_index = start_index
        end_index = end_index or start_index
        im = None
        txt = None
        with writer.saving(fig, filename, dpi=50):
            while True:
                curr_img = np.transpose(self._data['color'][curr_index], [1,2,0])
                if curr_img.shape[2] == 1:
                    curr_img = np.tile(curr_img, (1,1,3))

                curr_meas = np.array_str(self._data['measurements'][curr_index], precision=3)
                if curr_index == start_index:
                    im = axes.imshow(curr_img)
                    txt = axes.text(0.01, 2.0, curr_meas, fontsize=20, color='red')
                else:
                    im.set_data(curr_img)
                    txt.set_text(curr_meas)

                writer.grab_frame()

                curr_index = (curr_index + 1) % self.capacity
                if curr_index == end_index:
                    break

    def show(self, start_index=0, end_index=None, display=True, write_imgs=False, show_predictions=False, net_discrete_actions=None):
        if show_predictions:
            assert 'predictions' in self._data, 'Need to write_predictions to show predictions'
        curr_index = start_index
        if not end_index:
            end_index = start_index
        inp = ''
        if write_imgs:
            os.makedirs('imgs')
        print('Press ENTER to go to the next observation, type "quit" or "q" or "exit" and press ENTER to quit')

        if display or write_imgs:
            fig_img = plt.figure(figsize=(10, 7), dpi=50, tight_layout=True)
            ax_img = plt.gca()
        if display:
            fig_img.show()

        if show_predictions and net_discrete_actions is not None and len(net_discrete_actions):
            action_labels = []
            for act in net_discrete_actions:
                action_labels.append(''.join(str(int(i)) for i in act))

        while True:
            curr_img = np.transpose(self._data['color'][curr_index], (1,2,0))
            if curr_img.shape[2] == 1:
                curr_img = np.tile(curr_img, (1,1,3))
            if show_predictions:
                preds = self._data['predictions'][curr_index]
                objs = np.sum(preds, axis=1)
                objs_argsort = np.argsort(-objs)
                curr_preds = np.transpose(preds[objs_argsort[:]])
                curr_labels = [action_labels[i] for i in objs_argsort[:]]

            curr_meas_str = np.array_str(self._data['measurements'][curr_index], precision=3)
            if curr_index == start_index:
                if display or write_imgs:
                    im = ax_img.imshow(curr_img)
                    txt = ax_img.text(0.01, 5.0, curr_meas_str, fontsize=20, color='red')
                if show_predictions:
                    all_objs = np.sum(self._data['predictions'], axis=2)
                    sbp = my_util.StackedBarPlot(curr_preds, ylim=[np.min(all_objs), np.max(all_objs)], labels=curr_labels)
                    del all_objs
                    if display:
                        sbp.show()
            else:
                if display or write_imgs:
                    im.set_data(curr_img)
                    txt.set_text(curr_meas_str)
                if show_predictions:
                    sbp.set_data(curr_preds, labels=curr_labels)
            if write_imgs:
                plt.savefig('imgs/%.5d.png' % curr_index, dpi=50)
            if display:
                fig_img.canvas.draw()
                if show_predictions:
                    sbp.draw()
                print('Index', curr_index)
                print('Measurements:', self._data['measurements'][curr_index])
                print('Rewards:', self._data['rewards'][curr_index])
                print('Action:', self._data['actions'][curr_index])
                print('Terminal:', self._data['terminals'][curr_index])
                inp = input()

            curr_index = (curr_index + 1) % self.capacity
            if curr_index == end_index or inp == 'q' or inp == 'quit' or inp == 'exit':
                break
