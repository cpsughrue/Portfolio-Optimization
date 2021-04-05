from data_function import stock_return
import numpy as np

class Stock:

    def __init__(self, ticker_symbol, AlphaVantage_Key):
        ''' 
        :type ticker_symbol: str
        :type returns: np.array(dtype = float, ndim = 1)
        :type AlphaVantage_Key: str
        '''
        self.ticker = ticker_symbol
        self.returns = stock_return(self.ticker, AlphaVantage_Key)
        self.expected_return = np.average(self.returns)
        self.variance = np.var(self.returns)
