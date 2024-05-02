import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

# file = "../backtests/test.csv"
file = "../backtests/momentum_macd.csv"

df = pd.read_csv(file, names=["timestamp", "MACD", "MACD_signal"])
# clean up the chaotic beginning
first_day = df[:10000]  # look at the first day
first_day = first_day.drop(first_day[first_day.timestamp < 10000].index)
first_day = first_day.drop(first_day[first_day.timestamp < 50000].index)
first_day = first_day.drop(first_day[first_day.timestamp > 60000].index)
print(first_day)

first_day.plot(x="timestamp", y=["MACD", "MACD_signal"], title="first day")

def find_crossing(dataframe):
    macd_flag = -1
    buy_orders = []
    sell_orders = []

    for index, row in dataframe.iterrows():
        timestamp = row["timestamp"]
        macd, macd_signal = row["MACD"], row["MACD_signal"]
        if macd > macd_signal:
            sell_orders.append(np.nan)
            if macd_flag != 1:  # bullish momentum
                buy_orders.append(macd)
                macd_flag = 1
            else:
                buy_orders.append(np.nan)

        elif macd < macd_signal:
            buy_orders.append(np.nan)
            if macd_flag != 0:  # bearish momentum
                sell_orders.append(macd)
                macd_flag = 0
            else:
                sell_orders.append(np.nan)

        else:
            buy_orders.append(np.nan)
            sell_orders.append(np.nan)

    return buy_orders, sell_orders

buys, sells = find_crossing(first_day)

# These scatter plots have bugged timestamps
plt.scatter(first_day["timestamp"], buys, color='green', label='Buy', marker='^', alpha=1)
plt.scatter(first_day["timestamp"], sells, color='red', label='Sell', marker='v', alpha=1)

plt.show()

"""

second_day = df[10000:20000]  # look at the second day
second_day = second_day.drop(second_day[second_day.timestamp < 10000].index)
second_day = second_day[:200]
print(second_day)

second_day.plot(x="timestamp", y=["MACD", "MACD_signal"], title="second day")
plt.show()

third_day = df[20000:30000]  # look at the second day
third_day = third_day.drop(third_day[third_day.timestamp < 10000].index)
third_day = third_day[:200]
print(third_day)

third_day.plot(x="timestamp", y=["MACD", "MACD_signal"], title="third day")
plt.show()
"""


"""
# caveman statistics.
timestamps = []
macd_values = []
macd_signals = []
with open(file, mode="r") as f:
    for line in f:
        timestamp, macd, macd_signal = line.split(',')
        timestamps.append(timestamp)
        macd_values.append(macd)
        macd_signals.append(macd_signal)
"""
