import os
import abc
import tensorflow as tf
import numpy as np
import pandas as pd

from binance.client import Client as Client
from keras import layers
from datetime import timedelta

import pandas as pd
import os


import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import timedelta

import datetime as dt


class Data:


    def __init__(self, symbol, interval, client, data = [], working_file =''):
        if not os.path.exists(os.getcwd()+ '\\CSVData'):
            os.makedirs(os.getcwd()+ '\\CSVData')
        if working_file == '':
            self.working_file = os.getcwd()+ '\\CSVData\\' + symbol + 'Data.csv'
        else:
            self.working_file = working_file
        self.viewcache_folder = os.path.join(self.working_file, os.pardir) + '\\' + symbol + '\\'
        self.symbol = symbol# the coinpair symbol, ex: 'ETHUSDT' or 'TRXUSDT'
        self.interval = interval# the time interval to be used for binance Klinedata gathering
        self.data = data# base dataset, should only be replaced, updating is risky
        self.view = {}# cached views
        self.client = client# binance api client


        if data == []:
            self.get_historical_klines()

    #returns de dataset includig precessin operations based on given input key, also caches the views if they are already
    def get_view(self,key):
        key_file = self.viewcache_folder + key + '.csv'
        if key not in self.view:
            if os.path.exists(key_file):
                self.view[key] = pd.read_csv(key_file)
                print(key_file + ' cache found, loaded')
            else:
                if not os.path.exists(self.viewcache_folder):
                    os.mkdir(self.viewcache_folder)

                print( self.symbol + key +' view not cached, generating')
                self.view[key] = self.data.copy()

                for operation in key:

                   match operation:
                        case 'C':
                            self.view[key] = calculate_cycles(self.view[key])
                        case 'P':
                            self.view[key] = calculate_peaks(self.view[key])
                        case 'A':
                            self.view[key] = calculate_ascent(self.view[key])
                self.view[key].to_csv(key_file)

        return self.view[key]



    def get(self):
        return self.data

    def get_historical_klines(self, last_n=4096, force_fetch=False):

        if os.path.isfile(self.working_file) and force_fetch == False:
            self.data = pd.read_csv(self.working_file, index_col=[0]).astype(float)
            self.data.index = self.data.index.astype('datetime64[ns]')
            print('loaded existing {} data'.format(self.symbol))
        else:
            print('no existing {} CSV, loading fresh from Binance'.format(self.symbol))
            if (last_n > 30000):
                print('this might take a while......')

            callback_time = dt.datetime.now() - timedelta(minutes=5 * last_n) - timedelta(days=1)
            print(callback_time.strftime('%m %b, %Y'))
            klines = self.client.get_historical_klines(self.symbol, self.interval, start_str=callback_time.strftime('%m %b, %Y'))
            data = pd.DataFrame(klines)
            data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                            'taker_base_vol', 'taker_quote_vol', 'ignore']
            # change the timestamp
            data = data.astype(float)
            self.data = data.mask(np.isclose(data, 0)).ffill(downcast='infer')
            data['symbol'] = self.symbol
            data['close_time'] = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.close_time]
            data = data.set_index(
                pd.MultiIndex.from_frame(data[['close_time', 'symbol']]))

            # Forward fill missing values isclose because floating point innacuracy :C

            self.data.to_csv(self.working_file)

    def get_klines(self, n=1000):
        klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.n)
        data = pd.DataFrame(klines)
        print('reaching for {0} {1} {2} klines'.format(n, self.symbol, self.interval))

        data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                        'taker_base_vol', 'taker_quote_vol', 'ignore','symbol']
        data = data.astype(float)
        data = data.mask(np.isclose(data, 0)).ffill(downcast='infer')
        data['symbol'] = self.symbol
        # change the timestamp
        data.index = [[dt.datetime.fromtimestamp(x / 1000.0) for x in data.close_time], data[['symbol']]]
        # Forward fill missing values isclose because floating point innacuracy :C

        self.data = data


def multi_load(symbols, interval, client):
    data = {}
    for symbol in symbols:
        data[symbol] = Data(symbol, interval, client)
    return data


def bake_indexi(base_data, baked_view):
    data = base_data.copy()
    temp = pd.DataFrame()
    for symbol in data:

        data[symbol].data['symbol'] = data[symbol].symbol
        temp.append(data[symbol].data.rename(
            columns=[str(i)+symbol for i in data[symbol].data.columns]
        ))
        data[symbol].data = data[symbol].data.set_index(
            pd.MultiIndex.from_frame(data[symbol].data[['open_time','symbol']]))
    return  temp

    return data
##this will mangle the original object idk why
def stacked_frame(base_data, baked_view = 'CPA'):
    data = bake_indexi(base_data,baked_view)
    temp = pd.DataFrame()
    for symbol in data:
        temp = pd.concat([temp,data[symbol].data], sort=False)
    return temp.sort_index()




##Data operations
def trim(data, length = 50,  points = True):
    if points:
        data.data = data.data[len(data.data)*(length/100):]
    else:
        data.data = data.data[length:]
    data.view = {}
    data.to_csv(data.working_file)

def produce_trainingsets(data):
        x_params = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'poy', 'pod', 'pow']
        y_params = ['h5', 'h15', 'h30', 'h60', 'h120', 'h240', 'l5', 'l15', 'l30', 'l60', 'l120', 'l240']
        d_len = int(len(data[x_params[0]]) * 0.8)

        x_train = data[x_params].iloc[:d_len].astype(float)
        x_test = data[x_params].iloc[d_len:].astype(float)
        x_live = data[x_params].astype(float)

        y_train = data[y_params].iloc[:d_len].astype(float)
        y_test = data[y_params].iloc[d_len:].astype(float)

        x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))
        x_live = x_live.values.reshape((x_live.shape[0], 1, x_live.shape[1]))

        return x_train, x_test, y_train, y_test, x_live


def calculate_cycles(data):
    year = []
    day = []
    week = []
    for moment in data.index:
        year.append(moment.day_of_year / 366)
        week.append(moment.day_of_week / 6)
        day.append((moment.second + moment.minute * 60 + moment.hour * 3600) / 86399)
    data['poy'] = year
    data['pow'] = week
    data['pod'] = day
    return data


def calculate_peaks(data):
    # Higest&lowest for next 5,15,30,60,120,240 datapoints
    h5, h15, h30, h60, h120, h240 = ([] for i in range(6))
    l5, l15, l30, l60, l120, l240 = ([] for i in range(6))

    # for x in range(200)[5:10+5]:
    # print(data[['close']][x:199])

    for i in range(len(data[['close']])):
        # Calculating if peak or dip:
        # #shit-implementation ß)
        h5.append(data['close'][i] == max(data['close'][i:i + 5]))
        h15.append(data['close'][i] == max(data['close'][i:i + 15]))
        h30.append(data['close'][i] == max(data['close'][i:i + 30]))
        h60.append(data['close'][i] == max(data['close'][i:i + 60]))
        h120.append(data['close'][i] == max(data['close'][i:i + 120]))
        h240.append(data['close'][i] == max(data['close'][i:i + 240]))

        l5.append(data['close'][i] == min(data['close'][i:i + 5]))
        l15.append(data['close'][i] == min(data['close'][i:i + 15]))
        l30.append(data['close'][i] == min(data['close'][i:i + 30]))
        l60.append(data['close'][i] == min(data['close'][i:i + 60]))
        l120.append(data['close'][i] == min(data['close'][i:i + 120]))
        l240.append(data['close'][i] == min(data['close'][i:i + 240]))

    len(h240) == len(data[['close']])

    data['h5'] = h5
    data['h15'] = h15
    data['h30'] = h30
    data['h60'] = h60
    data['h120'] = h120
    data['h240'] = h240

    data['l5'] = l5
    data['l15'] = l15
    data['l30'] = l30
    data['l60'] = l60
    data['l120'] = l120
    data['l240'] = l240
    return data


def calculate_ascent(data):

    x_ascend = ['open', 'high', 'low', 'close', 'volume', 'num_trades']
    # data['volume'], data['num_trades'] = data['volume'].replace(0,1), data['num_trades'].replace(0,1)
    # data['volume'], data['num_trades'] = data['volume'].fillna(1), data['num_trades'].fillna(1)
    data[x_ascend] = data[x_ascend].astype(float)
    data[x_ascend] = data[x_ascend].pct_change()
    data = data.iloc[1:]

    return data