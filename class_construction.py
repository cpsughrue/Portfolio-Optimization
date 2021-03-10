from functions import *

class Portfolio:

    class Stock:

        def __init__(self, ticker_symbol, AlphaVantage_Key):
            ''' 
            :type ticker: str
            :type returns: np.array(dtype = float, ndim = 1)
            :type AlphaVantage_Key: str
            '''
            self.ticker = ticker_symbol
            self.returns = stock_return(self.ticker, AlphaVantage_Key)
            self.expected_return = np.average(self.returns)
            self.variance = np.var(self.returns)

    def __init__(self, AlphaVantage_Key):
        '''
        :type AlphaVantage_Key: str
        :type curr_pf: List[str]
        :type pf_returns: np.array(dtype = float, ndim = 2)
        '''
        self.AlphaVantage_Key = AlphaVantage_Key
        self.curr_pf = []
        self.list_pf = []

    def view_portfolio(self):
        '''
        :rtype List[str]
        '''
        return self.list_pf

    #working on implementation that allows user to add list of ticker_symbols at once
    def add_stock(self, ticker_symbol):
        '''
        :type ticker_symbol: str
        :return asset: instance of Stock class
        '''
        asset = self.Stock(ticker_symbol, self.AlphaVantage_Key)
        self.list_pf.append(ticker_symbol)
        self.curr_pf.append(asset)
        return asset

    #in progress
    def remove_stock(self, ticker_symbol):
        '''
        :type ticker_symbol: str
        '''
        pass

    def max_sharpe_opt(self, risk_free = 'BIL', allow_shorts = False):
        '''
        :type risk_free_proxy: str
        :type allow_shorts: Boolean
        :treturn ndarray(dtype = float, ndim = 1)
        '''

        matrix_of_returns = calc_matrix_of_returns(self.curr_pf)
        cov_matrix, expected_returns = statistics(matrix_of_returns, self.list_pf)
        risk_free_rate = np.average(stock_return(risk_free, self.AlphaVantage_Key))
        num_assets = len(self.list_pf)

        w = cp.Variable(num_assets)
        k = cp.Variable()

        function = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(function)

        new_constraints = []
        if allow_shorts == False:
            lower_bound = np.array([0] * num_assets)
            new_constraints = [w >= lower_bound]

        constraints = [(expected_returns - risk_free_rate).T * w == 1,
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
    AAPL = my_portfolio.add_stock('AAPL')
    GOOGL = my_portfolio.add_stock('GOOGL')
    AMZN = my_portfolio.add_stock('AMZN')

    #calculate appropriate weights to maximize the sharpe ratio of the portfolio
    weights = my_portfolio.max_sharpe_opt()
    portfolio = my_portfolio.view_portfolio()
    d = {portfolio[i] : weights[i] for i in range(len(portfolio))}
    print(d)