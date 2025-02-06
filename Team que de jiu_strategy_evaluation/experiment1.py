# The code that implements this experiment and generates the relevant charts and data should be submitted as experiment1.py. 

import pandas as pd
import datetime as dt
import os
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
from util import plot_data, get_data
import matplotlib.pyplot as plt

def report_metrics(port_vals: pd.Series, strategy: str, file_path: str, verbose=False) -> None:
    daily_rets = port_vals.copy()
    daily_rets[1:] = (port_vals[1:] / port_vals[:-1].values) - 1
    daily_rets = daily_rets.iloc[1:]

    # Calculate cumulative return, average daily return, and standard deviation of daily return
    cr = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    # Save the metrics to the specified file
    os.makedirs('reports', exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Cumulative Return of {strategy}: {cr:.6f}\n")
        f.write(f"Standard Deviation of {strategy}: {sddr:.6f}\n")
        f.write(f"Average Daily Return of {strategy}: {adr:.6f}\n\n")
    if verbose:
        print(f"Cumulative Return of {strategy}: {cr:.6f}")
        print(f"Standard Deviation of {strategy}: {sddr:.6f}")
        print(f"Average Daily Return of {strategy}: {adr:.6f}\n")

def experiment1(symbol = 'JPM', impact=0.0, commission_fee=0.0, sv=100_000, verbose=False):
    report_file = 'reports/experiment1_report.txt'
    # In-sample period
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    stock_price = get_data([symbol], pd.date_range(sd_in, ed_in), addSPY=False).dropna()
    # Build up the dataframe
    stock_price.index.name = 'Date'

    stock_price_out = get_data([symbol], pd.date_range(sd_out, ed_out), addSPY=False).dropna()
    stock_price_out.index.name = 'Date'
    
    # Manual Strategy
    ## in-sample
    ms = ManualStrategy()
    trades_ms = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    pr_trades_ms = pd.concat([stock_price, trades_ms], axis=1).dropna()
    portvals_ms = compute_portvals(pr_trades_ms, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(portvals_ms, 'Manual Strategy In-sample', report_file, verbose)
    normed_manual = portvals_ms / portvals_ms.iloc[0]

    ## out-sample
    trades_ms_out = ms.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    pr_trades_ms_out = pd.concat([stock_price_out, trades_ms_out], axis=1).dropna()
    portvals_ms_out = compute_portvals(pr_trades_ms_out, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(portvals_ms_out, 'Manual Strategy Out-sample', report_file, verbose)
    normed_manual_out = portvals_ms_out / portvals_ms_out.iloc[0]


    # Strategy Learner
    ## in-sample
    learner = StrategyLearner(verbose = False, impact = 0.0, commission=0.0)
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    trades_sl = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    pr_trades_sl = pd.concat([stock_price, trades_sl], axis=1).dropna()
    portvals_sl = compute_portvals(pr_trades_sl, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(portvals_sl, 'Strategy Learner In-sample', report_file, verbose)
    normed_sl = portvals_sl / portvals_sl.iloc[0]
    ## out-sample
    trades_sl_out = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    pr_trades_sl_out = pd.concat([stock_price_out, trades_sl_out], axis=1).dropna()
    portvals_sl_out = compute_portvals(pr_trades_sl_out, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(portvals_sl_out, 'Strategy Learner Out-sample', report_file, verbose)
    normed_sl_out = portvals_sl_out / portvals_sl_out.iloc[0]

    # Benchmark
    ## in-sample
    benchmark = pd.DataFrame(data=0, index=portvals_ms.index, columns=['Shares'])
    benchmark.iloc[0] = 1000
    pr_trades_benckmark = pd.concat([stock_price, benchmark], axis=1).dropna()
    benchmark = compute_portvals(pr_trades_benckmark, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(benchmark, 'Benchmark In-sample', report_file, verbose)
    normed_benchmark = benchmark / benchmark.iloc[0]
    ## out-sample
    benchmark_out = pd.DataFrame(data=0, index=portvals_ms_out.index, columns=['Shares'])
    benchmark_out.iloc[0] = 1000
    pr_trades_benckmark_out = pd.concat([stock_price_out, benchmark_out], axis=1).dropna()
    benchmark_out = compute_portvals(pr_trades_benckmark_out, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(benchmark_out, 'Benchmark Out-sample', report_file, verbose)
    normed_benchmark_out = benchmark_out / benchmark_out.iloc[0]


    # Plot the results
    ## in-sample
    df = pd.concat([normed_manual, normed_sl, normed_benchmark], axis=1)
    df.columns = ['Manual Strategy', 'Strategy Learner', 'Benchmark']
    df.plot(title='Experiment 1: In-sample', xlabel='Date', ylabel='Normalized Portfolio Value', grid=True, color=['black', 'blue', 'green'])
    plt.savefig('images/experiment1-insample.png')
    ## out-sample
    df_out = pd.concat([normed_manual_out, normed_sl_out, normed_benchmark_out], axis=1)
    df_out.columns = ['Manual Strategy', 'Strategy Learner', 'Benchmark']
    df_out.plot(title='Experiment 1: Out-sample', xlabel='Date', ylabel='Normalized Portfolio Value', grid=True, color=['black', 'blue', 'green'])
    plt.savefig('images/experiment1-outsample.png')


if __name__ == "__main__":
    experiment1(symbol = 'JPM', impact=0.0, commission_fee=0.0, sv=100_000, verbose=True)