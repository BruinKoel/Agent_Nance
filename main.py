from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import time

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import os
import abc
import threading

import binance.exceptions
import tensorflow as tf
import numpy as np
import pandas as pd

from binance.client import Client
import datetime as dt
from keras import layers
from datetime import timedelta

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU


import pandas as pd
import os


import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import timedelta

from binance.client import Client
import datetime as dt
import datahandler
import env

api_key = ''
api_secret = ''

config_path = os.getcwd()+ '\\' + 'config.txt'


def load_config():
    with open(config_path) as file:
        lines = file.readlines()
        file.close()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()
    return Client(api_key,api_secret)


def setup():
    while True:
        try:
            client = load_config()
            print('succesfully loaded api')
            break
        except Exception as E:
            print('no valid config found , or worse....')
            print(E)

        print('no valid api key, secret combination in config')
        print('enter api key:')

        with open(config_path, 'w') as file:
            file.write(input() + '\n')
            print('enter api secret')
            file.write(input() + '\n')
            file.close()
    return client


def main():
    client = setup()
    sets = ['ETHUSDT','TRXUSDT','ADAUSDT','BTCUSDT','ARUSDT','SOLUSDT']
    data = datahandler.multi_load(sets,'3m',client)



    threads = []
    for line in sets:
        thread = threading.Thread(target=datahandler.Data.get_view, args=(data[line],'CPA'))
        thread.start()
        thread.name = line
        time.sleep(0.5)
        threads.append(thread)
    for thread in threads:
        thread.join()
        print(thread.name + " view generation ran to completion!")

    while True:
        try:
            environment = env.KlineHikePyEnvironment(data)
            utils.validate_py_environment(environment, episodes=5)
        except Exception as excption:
            print(excption)

    print('kek + {0}'.format(len(data)))


main()