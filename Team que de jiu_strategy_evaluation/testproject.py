from experiment1 import experiment1
from experiment2 import experiment2
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import datetime as dt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Experiment 1
    experiment1()
    # Experiment 2
    experiment2()
    # Manual Strategy
    ms = ManualStrategy(verbose = False, impact = 0.0, commission=0.0)
    ms.plot(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100_000, name='images/ManualStrategy-in-sample.png')
    ms.plot(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100_000, name='images/ManualStrategy-out-sample.png')
    # Strategy Learner
    learner = StrategyLearner()
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011, 12, 31), sv=100000)
    trades = learner.testPolicy(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011, 12, 31), sv=100000)
    trades.plot(title="Strategy Learner", xlabel="Date", ylabel="Shares")
    plt.savefig('images/StrategyLearner_Trades.png')
