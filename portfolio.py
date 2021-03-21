from stock import Stock
from data_function import stock_return

import pandas as pd
import numpy as np
import math
import cvxpy as cp


class Portfolio:

    def __init__(self, AlphaVantage_Key):
        '''
        :type AlphaVantage_Key: str
        :type curr_pf: List[Stock]
        :type list_pf: List[str]
        '''
        self.AlphaVantage_Key = AlphaVantage_Key
        self.curr_pf = []
        self.list_pf = []

    def view_portfolio(self):
        '''
        :rtype List[str]
        '''
        return self.list_pf

    def add_stocks(self, tickers):
        '''
        :type tickers: List[str]
        :return assets: List[Stock]
        '''
        assets = [None] * len(tickers)
        assets_str = [None] * len(tickers)

        for index, stock in enumerate(tickers):

            stock_obj = Stock(stock, self.AlphaVantage_Key)

            assets[index] = stock_obj
            assets_str[index] = stock

        self.curr_pf = self.curr_pf + assets
        self.list_pf = self.list_pf + assets_str
        
        return assets

    def __calc_matrix_of_returns(self, portfolio):
        '''
        :type portfolio: List[Stock]
        :rtype ndarray(dtype = float, ndim = 2)
        '''
        num_row = len(portfolio)
        max_column = 0

        for stock in portfolio:
            if stock.returns.size > max_column:
                max_column = stock.returns.size

        matrix_of_returns = np.full((num_row, max_column), np.nan)
        for index, stock in enumerate(portfolio):
            matrix_of_returns[index, :stock.returns.size] = stock.returns
        
        return matrix_of_returns


    def __portfolio_returns(self, matrix_of_returns):
        '''
        :type matrix_of_returns: ndarray(dtype = float, ndim = 2)
        :rtype ndarray(dtype = float, ndim = 1)
        '''
        return np.nanmean(matrix_of_returns, axis = 1)


    def __portfolio_covariance(self, matrix_of_returns):
        '''
        :type matrix_of_returns: ndarray(dtype = float, ndim = 2)
        :rtype ndarray(dtype = float, ndim = 2), ndarray(dtype = float, ndim = 1)
        '''
        portfolio_df = pd.DataFrame(data = matrix_of_returns.T)
        return portfolio_df.cov().to_numpy()


    def max_sharpe_opt(self, risk_free = ['BIL'], allow_shorts = False):
        '''
        :type risk_free_proxy: str
        :type allow_shorts: Boolean
        :treturn ndarray(dtype = float, ndim = 1)
        '''
        num_assets = len(self.list_pf)
        matrix_of_returns = self.__calc_matrix_of_returns(self.curr_pf)
        
        expected_returns = self.__portfolio_returns(matrix_of_returns)
        cov_matrix = self.__portfolio_covariance(matrix_of_returns)
        
        risk_free_rate = np.average(stock_return(risk_free, self.AlphaVantage_Key))
        
        w = cp.Variable(num_assets)
        k = cp.Variable()

        function = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(function)

        new_constraints = []
        if allow_shorts == False:
            lower_bound = np.array([0] * num_assets)
            new_constraints = [w >= lower_bound]

        constraints = [(expected_returns - risk_free_rate).T @ w == 1,
                        cp.sum(w) == k,
                        k >= 0] + new_constraints

        problem = cp.Problem(objective, constraints)
        problem.solve()

        weights = (w.value / k.value).round(4)
        return weights

if __name__ == '__main__':
    #initialize portfolio
    AlphaVantage_Key = 'NV8EOH1CVVZRJ8DU'
    my_portfolio = Portfolio(AlphaVantage_Key)

    #add three stocks to portfolio
    stocks = ['AAPL', 'GOOGL', 'AMZN']
    maybe = my_portfolio.add_stocks(stocks)

    #calculate appropriate weights to maximize the sharpe ratio of the portfolio
    weights = my_portfolio.max_sharpe_opt()
    portfolio = my_portfolio.view_portfolio()
    d = {portfolio[i] : weights[i] for i in range(len(portfolio))}
    print(d)
