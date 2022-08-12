import os
import abc

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

api_key = ''
api_secret = ''
client = None
config_path = os.getcwd()+ '\\' + 'config.txt'

def api_check(client):
    try:
        print('key owned by: ' + client.get_account())
        return True
    except binance.exceptions.BinanceAPIException:
        print('false api key, secret combination')
    return False

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

def main():
    setup()


    data = datahandler.Data('ETHUSDT','3m',client)
    data.get_historical_klines()
    temp = data.get_view('CPA')
    temp = data.get_view('CPA')

main()