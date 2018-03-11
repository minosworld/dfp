import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from minos.lib.RoomSimulator import RoomSimulator
    from minos.lib import common
except ImportError as e:
    print('RoomSimulator unavailable.')
    print(e)
try:
    from .doom_simulator import DoomSimulator
except ImportError as e:
    print('DoomSimulator unavailable.')
    print(e)


class MultiSimulator:
    def __init__(self, all_args):
        self.num_simulators = len(all_args)
        self.simulators = []
        self.simulator_type = all_args[0]['simulator']
        for args in all_args:
            if self.simulator_type == 'room_simulator':
                self.simulators.append(RoomSimulator(args))
            else:
                self.simulators.append(DoomSimulator(args))
        self.modalities = all_args[0]['modalities']

        # TODO: outputs=modalities+rewards+terminals, needs to match observations (refactor to ensure consistency)
        self.outputs = all_args[0]['outputs']

        self.resolution = self.simulators[0].resolution
        self.num_channels = self.simulators[0].num_channels
        self.num_meas = self.simulators[0].num_meas
        self.action_len = self.simulators[0].num_buttons
        self.config = all_args[0]['config']
        self.maps = all_args[0]['maps']
        self.continuous_controls = self.simulators[0].continuous_controls
        self.discrete_controls = self.simulators[0].discrete_controls

    def __del__(self):
        self.close()

    def step(self, actions):
        """
        Action can be either the number of action or the actual list defining the action

        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            data_out - dictionary of lists with output data:
                images  - image after the step
                measurements - numpy array of returned additional measurements (e.g. health, ammo) after the step
                rewards - reward after the step
                terminals - if the state after the step is terminal
        """
        assert (len(actions) == len(self.simulators))

        data_out = {outp: self.num_simulators*[None] for outp in self.outputs}

        def act(idx, s):
            try:
                response = s.step(actions[idx])
                if self.simulator_type == 'room_simulator':
                    response = self._convert_observation(s, response, self.outputs)
                for outp in self.outputs:
                    data_out[outp][idx] = response[outp]
            except Exception as exc:
                print('Exception when stepping simulator with id: ' + str(s.sid))
                raise exc

        with ThreadPoolExecutor(max_workers=self.num_simulators) as executor:
            futures = []
            for i in range(self.num_simulators):
                future = executor.submit(act, i, self.simulators[i])
                futures.append(future)
            concurrent.futures.wait(futures)
            # check if any exception
            for f in futures:
                f.result()

        # data_out = {outp: [] for outp in self.outputs}
        #
        # for (sim, act) in zip(self.simulators, actions):
        #     data_one_sim = sim.step(act)
        #     for outp in self.outputs:
        #         data_out[outp].append(data_one_sim[outp])

        # print(data_out.keys())
        return data_out

    def num_actions(self, nsim):
        return self.simulators[nsim].num_actions

    def get_random_actions(self):
        acts = []
        for i in range(self.num_simulators):
            acts.append(self.simulators[i].get_random_action())
        return acts

    def close(self):
        for sim in self.simulators:
            sim.close()

    @staticmethod
    def _convert_observation(sim, response, outputs):
        observation = response['observation']
        sensors = observation.get('sensors')
        for outp in outputs:
            if outp == 'color':
                img = sensors.get('color').get('data')
                response[outp] = np.expand_dims(img, 0)  # add color channel dimension in front (single for gray)
            elif outp == 'depth':
                depth = sensors.get('depth').get('data')
                response[outp] = np.expand_dims(depth, 0)  # add "color" channel dimension in front (single for depth)
            elif outp == 'depth_clean':
                depth = sensors.get('depth').get('data_clean')
                response[outp] = np.expand_dims(depth, 0)  # add "color" channel dimension in front (single for depth)
            elif outp == 'forces':
                force = sensors.get('forces').get('data')
                response[outp] = np.expand_dims(force, 2)  # expand to 2D (add single column dimension)
            elif outp == 'audiopath':
                audiopath = sensors.get('audio').get('endpointShortestPaths')
                response[outp] = np.expand_dims(audiopath, 2)  # expand to 2D (add single column dimension)
            elif outp == 'audio':
                audio = sensors.get('audio').get('data')
                response[outp] = np.expand_dims(audio, 2)  # expand to 2D (add single column dimension)
            elif outp == 'roomType':
                rt = observation.get('roomInfo').get('roomTypeEncoded')
                response[outp] = np.expand_dims(rt, 2)
            elif outp == 'goalRoomType':
                rt = sim.start_config_this_episode['goal']['roomTypeEncoded']
                response[outp] = np.expand_dims(rt, 2)
        # print(response)

        return response
