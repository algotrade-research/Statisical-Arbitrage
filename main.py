import sys
import json
import os
import pandas as pd
import numpy as np
from utils.helper import generate_periods_df, run_backtest_for_periods
from utils.calculate_metrics import (
    calculate_metrics,
    calculate_monthly_returns,
    pivot_monthly_returns_to_table,
)
from data.get_data import get_vn30

# Set pandas and numpy options
pd.set_option("future.no_silent_downcasting", False)
np.random.seed(42)

# Default parameters (fallback if JSON not found)
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


def load_parameters(mode):
    """Load parameters from a JSON file based on the mode, or use defaults."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Use 'in_sample' parameters for 'out_sample' and 'overall' if no specific file exists
    param_mode = mode if mode in ["in_sample", "optimization"] else "optimization"
    param_file = f"{param_mode}.json"
    param_path = os.path.join(base_dir, "parameters", param_file)
    
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from {param_file}")
        return params
    except FileNotFoundError:
        print(f"Warning: {param_path} not found. Using default parameters.")
        return DEFAULT_PARAMS


def parse_arguments():
    """Parse and validate command-line arguments."""
    if len(sys.argv) < 3:
        print("Error: Please specify a mode ('in_sample', 'optimization', 'out_sample', 'overall') and data usage ('use_data' or 'not_use_data').")
        print("Example: python main.py optimization use_data")
        sys.exit(1)

    mode = sys.argv[1].lower()
    data_usage = sys.argv[2].lower()

    valid_modes = ["in_sample", "optimization", "out_sample", "overall"]
    if mode not in valid_modes:
        print(f"Error: Mode must be one of {valid_modes}.")
        sys.exit(1)
    if data_usage not in ["use_data", "not_use_data"]:
        print("Error: Data usage must be 'use_data' or 'not_use_data'.")
        sys.exit(1)

    return mode, data_usage == "use_data"


def load_vn30_stocks(use_existing_data, csv_path):
    """Load VN30 stock data from CSV if available and requested, otherwise fetch fresh."""
    if use_existing_data and os.path.exists(csv_path):
        vn30_stocks = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        vn30_stocks.index = pd.to_datetime(vn30_stocks.index)
        print(f"Loaded vn30_stocks from {csv_path}")
    else:
        vn30_stocks = get_vn30("2021-06-01", "2025-01-10")
        vn30_stocks.index = pd.to_datetime(vn30_stocks.index)
        print("Fetched fresh vn30_stocks data")
    return vn30_stocks


def run_analysis(vn30_stocks, params, use_existing_data, mode,monthly=False):
    """Run the backtest and calculate metrics based on the specified mode."""
    # Extract parameters
    estimation_window = params["estimation_window"]
    min_trading_days = params["min_trading_days"]
    max_clusters = params["max_clusters"]
    top_stocks = params["top_stocks"]
    tier = params["tier"]
    first_allocation = params["first_allocation"]
    adding_allocation = params["adding_allocation"]
    correlation_threshold = params["correlation_threshold"]

    # Step 1: Generate periods DataFrame
    etfs_list = ["FUEVFVND", "FUESSVFL", "E1VFVN30", "FUEVN100"]
    start_date = "2021-06-01"
    end_date = "2025-01-01"
    periods_df = generate_periods_df(vn30_stocks, start_date, end_date, window=80)

    # Step 2: Run backtest
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
        use_existing_data=use_existing_data,
    )

    # Step 3: Split into train and test sets
    train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
    test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]

    # Step 4: Calculate and plot metrics based on mode
    if mode in ["in_sample", "optimization"]:
        print("TRAIN SET")
        calculate_metrics(train_set, average_fee_ratio, risk_free_rate=0.05, plotting=True,use_existing_data=use_existing_data)
    elif mode == "out_sample":
        print("TEST SET")
        calculate_metrics(test_set, average_fee_ratio, risk_free_rate=0.05, plotting=True, use_existing_data=use_existing_data)
    elif mode == "overall":
        print("OVERALL")
        calculate_metrics(combined_returns_df, average_fee_ratio, risk_free_rate=0.05, plotting=True, use_existing_data=use_existing_data)
    
    # Display monthly returns table (optional for all modes)
    if monthly==True:
        monthly_returns = calculate_monthly_returns(combined_returns_df)
        print(pivot_monthly_returns_to_table(monthly_returns))


def main():
    """Main function to orchestrate the script execution."""
    # Parse command-line arguments
    mode, use_existing_data = parse_arguments()

    # Load parameters
    params = load_parameters(mode)

    # Load data
    data_folder = "data"
    csv_path = os.path.join(data_folder, "vn30_stocks.csv")
    vn30_stocks = load_vn30_stocks(use_existing_data, csv_path)

    # Run analysis
    run_analysis(vn30_stocks, params, use_existing_data, mode)


if __name__ == "__main__":
    main()