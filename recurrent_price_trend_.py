# coding: utf-8 -*-
# author: 梁开孟
# date：2021/10/8 0008 18:51
"""
    任务：预测股票价格走势
"""

import time
import tushare as ts
import numpy as np
import tensorflow as tf

today = time.strftime("%Y-%m-%d", time.localtime())

class TrainModel(object):
    def __init__(self, stock_number):
        data = self.download_realtime_data(stock_number)
        step = self.select_step_window(data)

        x, y = self.window_dataset(data, int(step))
        x, y = np.reshape(x[:-3], (x[:-3].shape[0], x[:-3].shape[1], 1)), y[:-3]
        model = self.set_model(int(step))
        model.fit(x, y, epochs=300)

        x0, y0 = np.reshape(x[-3:], (x[-3:].shape[0], x[-3:].shape[1], 1)), y[-3:]
        x0 = np.reshape(x0, (x0.shape[0], x0.shape[1], 1))
        pred = model.predict(x0)
        print(y0, pred)

    def select_step_window(self, df):
        steps = range(5, round(len(df) * 0.8))
        result = {}
        for step in steps:
            x, y = self.window_dataset(df, step)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            model = self.set_model(step)
            model.fit(x, y, epochs=300)
            result[str(step)] = model.evaluate(x, y)
        return min(result, key=result.get)

    def download_realtime_data(self,
                               stock_number,
                               start="1997-01-01",
                               end = today):
        return ts.get_k_data(stock_number, start=start, end=end)

    def window_dataset(self, df, step=9):
        win_close_prices = []
        win_close_labels = []
        Count = len(df)
        close_prices = df.close.tolist()
        for i, j in zip(range(0, Count-step), range(step, Count)):
            win_close_prices.append(close_prices[i:j])
            win_close_labels.append(close_prices[j])
        return np.array(win_close_prices), np.array(win_close_labels)

    def set_model(self, feature):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(feature, 1)),
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
    stock_number = "605162"
    TrainModel(stock_number)