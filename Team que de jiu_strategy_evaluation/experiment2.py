import datetime as dt
import pandas as pd
import os
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner
from experiment1 import report_metrics
from util import get_data
from marketsimcode import compute_portvals

def run(symbol = 'JPM', impact=0.0, commission_fee=0.0, sv=100_000):
    os.makedirs('reports', exist_ok=True)
    report_file = 'reports/experiment2_report.txt'
    # In-sample period
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    stock_price = get_data([symbol], pd.date_range(sd_in, ed_in), addSPY=False).dropna()
    # Build up the dataframe
    stock_price.index.name = 'Date'
    learner = StrategyLearner()
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    trades_sl = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    pr_trades_sl = pd.concat([stock_price, trades_sl], axis=1).dropna()
    portvals_sl = compute_portvals(pr_trades_sl, symbol, start_val=sv, commission_fee=commission_fee, market_impact=impact)
    report_metrics(portvals_sl, f'Strategy Learner In-sample with impact={impact}', report_file)
    normed_sl = portvals_sl / portvals_sl.iloc[0]
    return normed_sl

def experiment2():
    impacts = [0.0005, 0.005, 0.05, 0.1]
    # Create a figure and axis to plot all impact results on the same plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through different impact values and generate plots
    for impact in impacts:
        normed_sl = run(impact=impact)  # Call experiment2 with correct impact
        normed_sl.plot(ax=ax, label=f'Impact = {impact}')  # Plot on the same axis

    # Customize the plot
    ax.set_title('Strategy Learner In-sample with Various Impacts')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Value')
    ax.grid(True)
    ax.legend(title='Impact Levels')  # Ensure the legend is displayed

    # Save and show the plot
    plt.savefig('images/experiment2.png')
    # plt.show()


if __name__ == '__main__':
    experiment2()
    
