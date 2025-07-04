import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

@jit
def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    up_zig = True  # Last extreme is a bottom. Next is a top.
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig:  # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma:
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:  # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma:
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms


def get_extremes(ohlc: pd.DataFrame, sigma: float, close_key='close', high_key='high', low_key='low') -> pd.DataFrame:
    tops, bottoms = directional_change(ohlc[close_key].to_numpy(), ohlc[high_key].to_numpy(), ohlc[low_key].to_numpy(), sigma)
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    return extremes

def main(symbol: str='SPY', change_decimal: float=0.05):
    # data = pd.read_csv('.././data/BTCUSDT3600.csv')
    data = pd.read_csv(f'C:/Users/ccroc/Dev/NextGen-Traders/GoldenPocketScanner/output/data_cache/cached_price_data_{symbol}_daily_adjusted.csv.zip', index_col=0, parse_dates=True)
    if 'close' not in data.columns:
        # Handle different OHLC column naming conventions
        if '4. close' in data.columns:
            data = data.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            })
        elif 'Close' in data.columns:
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close'
            })
    # data['date'] = data['date'].astype('datetime64[s]')
    # data = data.set_index('date')

    close_numpy = data['close'].to_numpy()
    tops, bottoms = directional_change(close_numpy, data['high'].to_numpy(), data['low'].to_numpy(), change_decimal)
    plot_highs_and_lows(close_numpy, tops, bottoms)


def plot_highs_and_lows(close_numpy, tops, bottoms):
    pd.Series(close_numpy).plot()
    for top in tops:
        plt.plot(top[1], top[2], marker='v', color='red', markersize=4)
    for bottom in bottoms:
        plt.plot(bottom[1], bottom[2], marker='^', color='green', markersize=4)
    plt.show()


if __name__ == '__main__':
    main(symbol='ERD.TRT', change_decimal=0.1856)
