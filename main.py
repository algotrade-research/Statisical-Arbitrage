import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import os
from statsmodels.tsa.api import AutoReg
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import psycopg2
import time
from datetime import timedelta,datetime
from helper import *
from forming_combination.data_handler import DataHandler
from forming_combination.combination_formation import Combination_Formations
from signal_generation.signal_generator import process_results_df
from backtesting.port_mana import PortfolioManager
from utils.helper import plot_asset_balance


stocks=['ACB', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'KDH', 'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'PNJ', 'POW', 'SAB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']
futures='VN30F1M'
start_date='2021-06-01'
end_date='2022-07-31'
etf_list = ['FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']


# Initialize DataHandler with ETF list and etf_included flag
data_handler = DataHandler(
    futures=futures,
    stocks=stocks, 
    start_date=start_date,
    end_date=end_date,
    etf_list=etf_list,
    etf_included=False  # Set to False to exclude ETFs entirely
)

# Run the strategy
strategy = Combination_Formations(data_handler)
strategy.run_strategy()
results_df_1, stock_price = strategy.get_results()

results_df, positions_df, trading_log = process_results_df(results_df_1,stock_price,stocks,tier=1)

pm = PortfolioManager()
asset_df, detail_df, trading_log_df = pm.run_backtest(positions_df, stock_price)

plot_asset_balance(asset_df)