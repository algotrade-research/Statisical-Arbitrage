from forming_combination.data_handler import DataHandler
from data.get_data import get_vn30, get_etf_price, get_futures_price
from utils.helper import generate_periods_df
import pandas as pd
import os
import numpy as np

# Fetch VN30 stock data
vn30_stocks = get_vn30("2021-06-01", "2025-01-01")
vn30_stocks.index = pd.to_datetime(vn30_stocks.index)

# Fetch VN30 Price data
index_price = get_etf_price("VN30", "2021-06-01", "2025-01-01")
index_price.index = pd.to_datetime(index_price.index)

# Get the directory where this script (load_data.py) is located (Project 2/)
module_dir = os.path.dirname(os.path.abspath(__file__))

# Define the optimization_data folder as a sibling directory
folder = os.path.join(module_dir, "data")
os.makedirs(folder, exist_ok=True)  # Create folder if it doesn’t exist

# Define the save path
save_path_1 = os.path.join(folder, "vn30_stocks.csv")
save_path_2 = os.path.join(folder, "index_price.csv")

# Save the file
vn30_stocks.to_csv(save_path_1, index=True)
index_price.to_csv(save_path_2, index=True)

print(f"File saved to {os.path.abspath(save_path_1)}")
print(f"File saved to {os.path.abspath(save_path_2)}")

start_date = "2021-06-01"
end_date = "2025-01-01"
periods_df = generate_periods_df(vn30_stocks, start_date, end_date, window=80)

for period_idx, period in periods_df.iterrows():
        active_stocks = period["stocks_list"]
        start_date = period["start_date"]
        end_date = period["end_date"]
        data_handler = DataHandler(
            futures='VN30F1M',
            stocks=active_stocks,
            start_date=start_date,
            end_date=end_date,
            use_existing_data=True
        )
        data_handler.load_data()
        print(f"Data loaded for period {period_idx}: {start_date} to {end_date}")