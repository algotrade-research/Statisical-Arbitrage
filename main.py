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
from forming_combination.data_handler import DataHandler
from forming_combination.combination_formation import Combination_Formations
from signal_generation.signal_generator import process_results_df
from backtesting.port_mana import PortfolioManager
from utils.helper import plot_asset_balance, generate_periods_df,run_backtest_for_periods
from utils.calculate_metrics import calculate_metrics
from data.get_data import *
pd.set_option('future.no_silent_downcasting', True)

# Input
estimation_window = 60
min_trading_days=45
max_clusters=10
top_stocks=8
correlation_threshold=0.6
residual_threshold=0.3
improvement_threshold=0.03
ou_window=estimation_window
tier=5
first_allocation=0.4    
adding_allocation=0.2


vn30_stocks=get_vn30('2021-06-01', '2025-01-10')
vn30_stocks.index = pd.to_datetime(vn30_stocks.index)

# Step 1: Generate the periods DataFrame with ETFs
etfs_list = [ 'FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']
start_date='2021-06-01'
end_date='2025-01-01'
periods_df = generate_periods_df(vn30_stocks, 
                                 start_date, 
                                 end_date, 
                                 window=80)

# Step 2: Run the backtest using the periods DataFrame
combined_returns_df, combined_detail_df,average_fee_ratio = run_backtest_for_periods(
    periods_df=periods_df,
    futures='VN30F1M',
    etf_list=etfs_list,
    etf_included=False,
    estimation_window=estimation_window,
    min_trading_days=min_trading_days,
    max_clusters=max_clusters,
    top_stocks=top_stocks,
    correlation_threshold=correlation_threshold,
    residual_threshold=residual_threshold,
    improvement_threshold=improvement_threshold,
    tier=tier,
    first_allocation=first_allocation,
    adding_allocation=adding_allocation,
    use_existing_data=True
)
train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]
print("TRAIN SET")
calculate_metrics(train_set,average_fee_ratio, risk_free_rate=0.05)
print("TEST SET")
calculate_metrics(test_set,average_fee_ratio, risk_free_rate=0.05)