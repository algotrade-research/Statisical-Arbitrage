import pandas as pd
from utils.helper import (
    generate_periods_df,
    run_backtest_for_periods,
)
from utils.calculate_metrics import (
    calculate_metrics,
    calculate_monthly_returns,
    pivot_monthly_returns_to_table,
)
from data.get_data import *
pd.set_option("future.no_silent_downcasting", False)
import numpy as np
np.random.seed(42)
import json
import os
import sys

# Default parameters (embedded in the code)
DEFAULT_PARAMS = {
    "estimation_window": 50,
    "min_trading_days": 25,  
    "max_clusters": 10,
    "top_stocks": 3,
    "tier": 1,
    "first_allocation": 0.4,
    "adding_allocation": 0.2,
    "correlation_threshold": 0.6
}

# Function to load parameters based on mode
def load_parameters(mode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = f"{mode}.json"
    param_path = os.path.join(base_dir, "parameters", param_file)
    
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from {param_file}")
    except FileNotFoundError:
        print(f"Warning: {param_path} not found. Using default parameters.")
        params = DEFAULT_PARAMS
    return params

# Check command-line argument
if len(sys.argv) < 2:
    print("Error: Please specify a mode ('initial' or 'optimization').")
    print("Example: python main.py initial")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ["initial", "optimization"]:
    print("Error: Mode must be 'initial' or 'optimization'.")
    sys.exit(1)

# Load parameters based on the mode
train_params = load_parameters(mode)

# Access parameters
estimation_window = train_params["estimation_window"]
min_trading_days = train_params["min_trading_days"]
max_clusters = train_params["max_clusters"]
top_stocks = train_params["top_stocks"]
tier = train_params["tier"]
first_allocation = train_params["first_allocation"]
adding_allocation = train_params["adding_allocation"]
correlation_threshold = train_params["correlation_threshold"]
ou_window = estimation_window


data_folder = "data"
csv_path = os.path.join(data_folder, "vn30_stocks.csv")

# Check if CSV exists
if os.path.exists(csv_path):
    # Load from CSV
    vn30_stocks = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    vn30_stocks.index = pd.to_datetime(vn30_stocks.index)
else:
    # Fetch data if CSV doesn't exist
    vn30_stocks = get_vn30("2021-06-01", "2025-01-10")
    vn30_stocks.index = pd.to_datetime(vn30_stocks.index)

# Step 1: Generate the periods DataFrame with ETFs
etfs_list = ["FUEVFVND", "FUESSVFL", "E1VFVN30", "FUEVN100"]
start_date = "2021-06-01"
end_date = "2025-01-01"
periods_df = generate_periods_df(vn30_stocks, start_date, end_date, window=80)

# Step 2: Run the backtest using the periods DataFrame
combined_returns_df, combined_detail_df, average_fee_ratio = run_backtest_for_periods(
    periods_df=periods_df,
    futures="VN30F1M",
    etf_list=etfs_list,
    etf_included=False,
    estimation_window=estimation_window,
    min_trading_days=min_trading_days,
    max_clusters=max_clusters,
    top_stocks=top_stocks,
    correlation_threshold=correlation_threshold,
    tier=tier,
    first_allocation=first_allocation,
    adding_allocation=adding_allocation,
    use_existing_data=True,
)
train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]
print("TRAIN SET")
calculate_metrics(train_set, average_fee_ratio, risk_free_rate=0.05)
print("TEST SET")
calculate_metrics(test_set, average_fee_ratio, risk_free_rate=0.05)
print("OVERALL")
calculate_metrics(
    combined_returns_df, average_fee_ratio, risk_free_rate=0.05, plotting=True
)
monthly_returns = calculate_monthly_returns(combined_returns_df)
print(pivot_monthly_returns_to_table(monthly_returns))
