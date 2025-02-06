import datetime as dt
import BagLearner as bl
import RTLearner as rt
import numpy as np   
from indicators import *
import matplotlib.pyplot as plt  		  	   		 	   		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data 	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
class StrategyLearner(object):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # constructor  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		 	   		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		 	   		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)

  		  	   		 	   		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		 	   		  		  		    	 		 		   		 		  
    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):  		  	   		 	   		  		  		    	 		 		   		 		    		  	   		 	   		  		  		    	 		 		   		 		  
        adjusted_close_prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
        # Build up the dataframe
        adjusted_close_prices.index.name = 'Date'
        input_df = pd.DataFrame(index=adjusted_close_prices.index)
        input_df[symbol] = adjusted_close_prices[symbol]
        input_df['RSI'] = RSI(input_df[symbol])
        input_df['BollingerBands'] = BollingerBands(input_df,symbol)['Signal']
        input_df['MACD'] = MACD(input_df, symbol)['MACD']
        input_df['MACD_Signal_Line'] = MACD(input_df, symbol)['Signal Line']
        input_df['MACD_Histogram'] = MACD(input_df, symbol)['Histogram']
        input_df_copy = input_df.copy()
        input_df.fillna(0, inplace=True)
        trainX = input_df[:-5].values # remove last 5 days for prediction
        trainY = []
        price = input_df[symbol].values
        # We predict the price after 5 days
        for i in range(len(input_df_copy)-5):
            ratio = (price[i+5]-price[i])/price[i]
            if ratio > (0.02 + self.impact): # if price increase more than 2%, we buy
                trainY.append(1)
            elif ratio < (-0.02 - self.impact): # if price decrease more than 2%, we sell
                trainY.append(-1)
            else:
                trainY.append(0) # hold
        trainY=np.array(trainY)
        self.learner.add_evidence(trainX, trainY)
	  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		 	   		  		  		    	 		 		   		 		  
    def testPolicy(self,symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1),sv=10000):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        adjusted_close_prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
        # Build up the dataframe
        adjusted_close_prices.index.name = 'Date'
        input_df = pd.DataFrame(index=adjusted_close_prices.index)
        input_df[symbol] = adjusted_close_prices[symbol]
        input_df['RSI'] = RSI(input_df[symbol])
        input_df['BollingerBands'] = BollingerBands(input_df,symbol)['Signal']
        input_df['MACD'] = MACD(input_df, symbol)['MACD']
        input_df['MACD_Signal_Line'] = MACD(input_df, symbol)['Signal Line']
        input_df['MACD_Histogram'] = MACD(input_df, symbol)['Histogram']
        input_df.fillna(0, inplace=True)
        testX = input_df.values
        # Predicted Values
        testY = self.learner.query(testX) 

        trades = pd.DataFrame(index=adjusted_close_prices.index, columns=['Shares']).fillna(0)
        position = 0
        """
        The Strategy here is when we have no stock on hand, we can buy or sell 1000 shares.
        if we have stock on hand, we can sell 2000 shares or buy another 1000 shares or back to 0 shares
        if we have short position, we can buy 2000 shares or sell another 1000 shares or back to 0 shares
        on the last day, we need to close the position.
        """
        for i in range(0,len(trades)-1):
            if position==0:
                if testY[i]>0:
                    trades.values[i] = 1000
                    position = 1
                elif testY[i]<0:
                    trades.values[i] = -1000
                    position = -1
                # To make the table consistance in length, we included if no trading happen, we put it as 0. all else: statement below applies to this case
                else:
                    trades.values[i] = 0
                    position = 0

            elif position==1:
                if testY[i]<0:
                    trades.values[i]=-2000
                    position=-1
                elif testY[i]==0:
                    trades.values[i]=-1000
                    position = 0
                else:
                    trades.values[i]=0
                    position=1

            else:
                if testY[i]>0:
                    trades.values[i]=2000
                    position=1
                elif testY[i]==0:
                    trades.values[i]=1000
                    position=0
                else:
                    trades.values[i]=0
                    position=-1

        if position==-1:
            trades.values[len(trades)-1]=1000
        elif position==1:
            trades.values[len(trades)-1]=-1000
        else:
            trades.values[len(trades)-1]=0

        trades.name = "Shares"

        return trades  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    learner = StrategyLearner()
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011, 12, 31), sv=100000)
    trades = learner.testPolicy(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011, 12, 31), sv=100000)
    trades.plot(title="Strategy Learner", xlabel="Date", ylabel="Shares")
    plt.savefig('images/StrategyLearner_Trades.png')
    

