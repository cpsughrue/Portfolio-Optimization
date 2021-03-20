import requests
import json
import pandas as pd
import numpy as np

def get_data(ticker_symbol, AlphaVantage_Key):
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
    :rtype ndrray(dtype = float, ndim = 1)
    '''
    data = json_file.json()
    data_df = pd.DataFrame.from_dict(data['Monthly Adjusted Time Series'], orient = 'index')

    targetColumn = '5. adjusted close'
    adjusted_close = data_df.loc[:, targetColumn].to_numpy().astype(np.float)

    pct_change = (adjusted_close[:-1] / adjusted_close[1:]) - 1
    return pct_change

def stock_return(ticker_symbol, AlphaVantage_Key):
    '''
    :type ticker_symbol: str
    :type AlphaVantage_Key: str
    :rtype ndrray(dtype = float, ndim = 1)
    '''
    json_file = get_data(ticker_symbol, AlphaVantage_Key)
    adjusted_returns = clean_data(json_file)
    return adjusted_returns