import requests
import json
import pandas as pd
import numpy as np
import math
import cvxpy as cp

#returns json data
def gather_data(ticker_symbol, AlphaVantage_Key):
    '''
    :type stock: str
    :rtype json file
    '''
    AlphaVantage_URL = "https://www.alphavantage.co/query"
    data = {"function": "TIME_SERIES_MONTHLY_ADJUSTED",
            "symbol": ticker_symbol,
            "datatype": "json",
            "apikey": AlphaVantage_Key}
    return requests.get(AlphaVantage_URL, data)

def clean_data(json_file):
    '''
    :type json_file: json file
    :rtype ndrray(dtype = float, ndim = 2)
    '''
    #converts data in json file to pandas DataFrame
    data = json_file.json()
    data_df = pd.DataFrame.from_dict(data['Monthly Adjusted Time Series'], orient = 'index')
    #converts column of pandas DataFrame to np.array(dtype = float, ndim = 1)
    adjusted_close = data_df.loc[:,'5. adjusted close'].to_numpy().astype(np.float)
    #calculated percent change between adjusted closing price and reshapes np.array
    pct_change = (adjusted_close[:-1] / adjusted_close[1:]) - 1
    return np.reshape(pct_change, (1, len(pct_change)))

def stock_return(ticker_symbol, AlphaVantage_Key):
    '''
    :type ticker_symbol: str
    :type AlphaVantage_Key: str
    :rtype ndrray(dtype = float, ndim = 2)
    '''
    json_file = gather_data(ticker_symbol, AlphaVantage_Key)
    stock_returns = clean_data(json_file)
    return stock_returns

def calc_matrix_of_returns(portfolio):
    '''
    :type portfolio: List[Stock]
    :rtype ndarray(dtype = float, ndim = 2)
    '''
    num_row = len(portfolio)
    num_column = 0
    #determine number of columns needed for matrix initialization
    for stock in portfolio:
        if stock.returns.shape[1] > num_column:
            num_column = stock.returns.shape[1]
    #initialize and populate matrix of returns
    matrix_of_returns = np.full((num_row, num_column), np.nan)
    for index, stock in enumerate(portfolio):
        matrix_of_returns[index, :stock.returns.shape[1]] = stock.returns
    return matrix_of_returns

def statistics(matrix_of_returns, list_of_stocks):
    '''
    :type matrix_of_returns: ndarray(dtype = float, ndim = 2)
    :type list_of_stocks: List[str]
    :rtype ndarray(dtype = float, ndim = 2), ndarray(dtype = float, ndim = 1)
    '''
    #convert 2D numpy array into pandas dataframe to handle missing data when constructing covariance matrix
    #each stock is represented by a row in numpy array
    #numpy array must be transposed in order to represent data in tabular format
    portfolio_df = pd.DataFrame(data = matrix_of_returns.T, columns = list_of_stocks)
    #calculate covariance matrix and expected return for each stock
    covariance_matrix = portfolio_df.cov().to_numpy()
    expected_returns = np.nanmean(matrix_of_returns, axis = 1)
    return covariance_matrix, expected_returns

#no longer in use but still functional
def append_matrix_of_returns(curr_matrix, addition):
    '''
    :type curr_matrix: ndrray(dtype = float, ndim = 2)
    :type addition: ndrray(dtype = float, ndim = 2)
    :type ndarray(dtype = float, ndim = 2)
    '''

    if curr_matrix.size == 1:
    
        num_column = addition.shape[1] - curr_matrix.shape[1]
        new_column = np.full((curr_matrix.shape[0], num_column), np.nan)
        curr_matrix = np.concatenate((curr_matrix, new_column), axis = 1)
        
        curr_matrix[-1, :addition.shape[1]] = addition
        return curr_matrix

    else:
    
        if addition.shape[1] < curr_matrix.shape[1]:
            
            new_row = np.full((1, curr_matrix.shape[1]), np.nan)
            curr_matrix = np.concatenate((curr_matrix, new_row), axis = 0)
            
            curr_matrix[-1, :addition.shape[1]] = addition
            return curr_matrix
        
        else:
            
            num_column = addition.shape[1] - curr_matrix.shape[1]
            new_column = np.full((curr_matrix.shape[0], num_column), np.nan)
            curr_matrix = np.concatenate((curr_matrix, new_column), axis = 1)
            
            new_row = np.full((1, curr_matrix.shape[1]), np.nan)
            curr_matrix = np.concatenate((curr_matrix, new_row), axis = 0)
            
            curr_matrix[-1, :addition.shape[1]] = addition
            return curr_matrix

    return curr_matrix