# coding: utf-8 -*-
# author: 梁开孟
# date：2021/10/7 0007 01:12

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from stock_price_trend_predict.load_data import *

class StockPricePredict(object):
    def __init__(self, stock_number):
        data = dataset(stock_number)
        win_close_prices, win_close_labels = self.window_dataset(data)
        x_train, x_test, y_train, y_test = self.split_data(win_close_prices, win_close_labels)
        model = self.set_model(x_train)
        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=1,
                  batch_size=32)

        train_pre = model.predict(x_train).reshape(1, -1)[0]
        test_pre = model.predict(x_test).reshape(1, -1)[0]
        pred = np.array(train_pre.tolist() + test_pre.tolist())
        y = np.array(y_train.tolist() + y_test.tolist())
        plt.figure(figsize=(15, 8))
        plt.plot(y, label='origin')
        plt.plot(pred[1:], label='predict')
        plt.legend()
        plt.savefig('picture/{}.jpg'.format(stock_number+" "+ time.strftime("%Y-%m-%d", time.localtime())),
                    dpi=400,
                    bbox_inches='tight')
        self.t = model.history

    def window_dataset(self, df, step=51):
        win_close_prices = []
        win_close_labels = []
        Count = len(df)
        close_prices = df.close.tolist()
        for i, j in zip(range(0, Count-step), range(step, Count)):
            win_close_prices.append(close_prices[i:j])
            win_close_labels.append(close_prices[j])
        return np.array(win_close_prices), np.array(win_close_labels)

    def split_data(self, win_close_prices, win_close_labels):
        split_dot = round(len(win_close_prices)*0.9)
        x_train, x_test = win_close_prices[:split_dot], win_close_prices[split_dot:]
        y_train, y_test = win_close_labels[:split_dot], win_close_labels[split_dot:]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_train, x_test, y_train, y_test

    def set_model(self, feature):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(feature.shape[1], 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

if __name__ == "__main__":
    """
        688711 -- 宏微科技
        688621 -- 阳光诺和
        688626 -- 翔宇医疗
    """
    stock_number = "600653"
    mclass = StockPricePredict(stock_number)
    print(mclass.t.history)