import pandas as pd
import datetime as dt
from util import get_data
from indicators import MACD, BollingerBands, RSI
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

class ManualStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100_000)->pd.DataFrame:
        adjusted_close_prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
        # Build up the dataframe
        adjusted_close_prices.index.name = 'Date'
        output_df = pd.DataFrame(index=adjusted_close_prices.index, columns=[symbol, 'RSI', 'BollingerBands', 'MACD_buy', 'MACD_sell'])
        output_df[symbol] = adjusted_close_prices[symbol]
        output_df['RSI'] = RSI(output_df[symbol])
        output_df['BollingerBands'] = BollingerBands(output_df,symbol)['Signal']
        output_df['MACD_buy'] = MACD(output_df, symbol)['Buy Signal']
        output_df['MACD_sell'] = MACD(output_df, symbol)['Sell Signal']

        flag = 0  # 0: no position, 1: long, -1: short
        index = 0
        trades = pd.DataFrame(columns=['Date', 'Symbol', 'Shares'])

        for i in range(0, len(output_df) - 1):
            if flag == 0:  # No position
                if output_df['RSI'].iloc[i] < 30 or output_df['BollingerBands'].iloc[i] < 0.2 or output_df['MACD_buy'].iloc[i] == True:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [1000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = 1  # Enter long position
                    index += 1
                elif output_df['RSI'].iloc[i] > 70 or output_df['BollingerBands'].iloc[i] > 0.8 or output_df['MACD_sell'].iloc[i]  == True:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [-1000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = -1  # Enter short position
                    index += 1
                else:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [0]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    index += 1

            elif flag == -1:  # Short position
                if output_df['RSI'].iloc[i] < 25 or output_df['BollingerBands'].iloc[i] < 0.15:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [2000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = 1  # Exit short and go long
                    index += 1
                elif output_df['RSI'].iloc[i] < 30 or output_df['BollingerBands'].iloc[i] < 0.2 or output_df['MACD_buy'].iloc[i]  == True:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [1000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = 0  # Exit short and go neutral
                    index += 1
                else:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol],'Shares': [0]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    index += 1

            elif flag == 1:  # Long position
                if output_df['RSI'].iloc[i] > 75 or output_df['BollingerBands'].iloc[i] > 0.85:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [-2000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = -1  # Exit long and go short
                    index += 1
                elif output_df['RSI'].iloc[i] > 70 or output_df['BollingerBands'].iloc[i] > 0.8 or output_df['MACD_sell'].iloc[i]  == True:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [-1000]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    flag = 0  # Exit long and go neutral
                    index += 1
                else:
                    trade = pd.DataFrame({'Date': [output_df.index[i]], 'Symbol': [symbol], 'Shares': [0]})
                    trades = pd.concat([trades, trade], ignore_index=True)
                    index += 1

        # Closing any open positions at the last day
        if flag == 1:  # If we are long, sell on the last day
            trade = pd.DataFrame({'Date': [output_df.index[-1]], 'Symbol': [symbol], 'Shares': [-1000]})
            trades = pd.concat([trades, trade], ignore_index=True)
        elif flag == -1:  # If we are short, buy on the last day
            trade = pd.DataFrame({'Date': [output_df.index[-1]], 'Symbol': [symbol], 'Shares': [1000]})
            trades = pd.concat([trades, trade], ignore_index=True)
        else:
            trade = pd.DataFrame({'Date': [output_df.index[-1]], 'Symbol': [symbol], 'Shares': [0]})
            trades = pd.concat([trades, trade], ignore_index=True)

        # Trades -> df has columns 'Date' as index, 'Symbol', 'Shares', where Shares will be [2000, 1000, 0, -1000, -2000], where negative means shorting
        trades.set_index('Date', inplace=True) 
        # Drop Symbol column
        trades.drop('Symbol', axis=1, inplace=True)
        return trades
    
    def plot(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100_000, name='ManualStrategy.png'):
        stock_price = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
        # Manual Strategy 
        trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        pr_trades = pd.concat([stock_price, trades], axis=1).dropna()
        portvals_ms = compute_portvals(pr_trades, symbol, start_val=sv, commission_fee=0.0, market_impact=0.0)
        normed_manual = portvals_ms / portvals_ms.iloc[0]
        # Benchmark
        benchmark = pd.DataFrame(data=0, index=portvals_ms.index, columns=['Shares'])
        benchmark.iloc[0] = 1000
        pr_trades_benckmark = pd.concat([stock_price, benchmark], axis=1).dropna()
        benchmark = compute_portvals(pr_trades_benckmark, symbol, start_val=sv, commission_fee=0.0, market_impact=0.0)
        normed_benchmark = benchmark / benchmark.iloc[0]
        # Plot
        df = pd.concat([normed_manual, normed_benchmark], axis=1)
        df.columns = ['Manual Strategy', 'Benchmark']
        df.plot(title='Manual Strategy', xlabel='Date', ylabel='Normalized Portfolio Value', grid=True, color=['red', 'purple'])
        for i in range(0, len(trades)):
            if trades['Shares'].iloc[i] >= 1000:
                plt.axvline(x=trades.index[i], color='blue', linestyle='--')
            elif trades['Shares'].iloc[i] <= -1000:
                plt.axvline(x=trades.index[i], color='black', linestyle='--')
        plt.savefig(name)

if __name__ == "__main__":
    ms = ManualStrategy()
    ms.plot(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100_000, name='images/ManualStrategy-in-sample.png')
    ms.plot(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100_000, name='images/ManualStrategy-out-sample.png')


