import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.specs import tensor_spec

import numpy as np
import pandas as pd



class KlineHikePyEnvironment(py_environment.PyEnvironment):

    def __init__(self, data, scope=1024, view='CPA'):
        self.output_text = []
        self.scope = scope
        self.colums = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'poy', 'pod', 'pow']
        self.view = view
        self.data = data
        self.fiat = 1000
        self.crypto = 0
        self._state = data[next(iter(data))].data['open_time'].iloc(1)
        self.current_price = 0

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,1), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(data),len(self.colums) * self.scope), dtype=np.float32,
            name='observation')

        self._episode_ended = False
##
    def _sum_wallet(self):
        return float(1)
##
    def _make_observation(self):
        temp = pd.DataFrame()
        for frame in self.data:
            temp = pd.concat([temp,self.data[frame].get_view(self.view)], sort=False)

        return np.array(self.data[frame].get_view('CA')[self._state - self.scope: self._state].stack(),
                        dtype=np.float32)

    def get_state(self):
        return self._state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
##
    def _reset(self):
        self._state = 1 + self.scope
        self.fiat = 1000
        self.crypto = 0

        self._episode_ended = False
        return ts.restart(self._make_observation())
##
    def _step(self, action):
        action = (action - 0.5) * 2
        self._state += 1

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self.current_price = self.data['close'].iloc[self._state]

        if (action > 0.05 and self.fiat > 10):
            amount = self.fiat * action
            self.crypto += amount / self.current_price
            self.fiat -= amount
            print('bought {0} crypto at price {1} for total {2}'.format(amount / self.current_price, self.current_price,
                                                                        amount))

        elif (action < -0.05 and self.crypto / self.current_price > 10):
            amount = self.crypto * action * -1
            self.fiat += amount * self.current_price
            self.crypto -= amount
            print('sold {0} crypto at price {1} for total {2}'.format(amount, self.current_price,
                                                                      amount * self.current_price))

        if self._state == len(self.data.get_view('CA')['close']) - self.scope:
            self._episode_ended = True
            print('Terminating with {0} fiat and {1} for a total of {2}'.format(self.fiat, self.crypto,
                                                                                self._sum_wallet()))
            return ts.termination(self._make_observation(), reward=self._sum_wallet())

        else:
            return ts.transition(self._make_observation(), reward=self._sum_wallet(), discount=1.0)

