import pandas as pd
from datetime import timedelta
from data.get_data import get_vn30
from forming_combination.data_handler import DataHandler
from forming_combination.combination_formation import Combination_Formations
from backtesting.port_mana import PortfolioManager
from signal_generation.signal_generator import process_results_df
import matplotlib.pyplot as plt


def generate_periods_df(vn30_stocks, start_date, end_date, window=60):
    """Generate a DataFrame of trading periods based on VN30 stock data, with the first start_date as specified.

    Args:
        vn30_stocks (pd.DataFrame): DataFrame with 'Date' index and 'Stock' column from get_vn30.
        start_date (str): Desired start date for the first period in 'YYYY-MM-DD' format.
        end_date (str): End date for the last period in 'YYYY-MM-DD' format.
        window (int): Number of days to subtract from rebalancing date for start_date. Defaults to 60.

    Returns:
        pd.DataFrame: DataFrame with columns ['stocks_list', 'start_date', 'end_date'],
                      where the first start_date matches the input start_date.

    Raises:
        ValueError: If vn30_stocks is empty.
    """
    if vn30_stocks.empty:
        raise ValueError("No data found in vn30_stocks.")

    # Group stocks by date to get the list of stocks for each rebalancing date
    grouped = vn30_stocks.groupby("Date")["Stock"].apply(list)
    date_stock_pairs = [(date, stocks) for date, stocks in grouped.items()]

    # Generate periods
    periods = []
    for i, (current_date, current_stocks) in enumerate(date_stock_pairs):
        # Calculate start_date
        start_date_calc = current_date - timedelta(days=window)

        # Override the first start_date to match the input start_date
        if i == 0:
            period_start_date = pd.to_datetime(start_date)
        else:
            period_start_date = start_date_calc

        # Determine end_date
        if i < len(date_stock_pairs) - 1:
            next_date = date_stock_pairs[i + 1][0]
            end_date_calc = next_date - timedelta(days=1)
        else:
            end_date_calc = pd.to_datetime(
                end_date
            )  # Use the provided end_date for the last period

        # Format dates as strings
        start_date_str = period_start_date.strftime("%Y-%m-%d")
        end_date_str = end_date_calc.strftime("%Y-%m-%d")

        periods.append(
            {
                "stocks_list": current_stocks,  # No ETFs added
                "start_date": start_date_str,
                "end_date": end_date_str,
            }
        )

    # Convert to DataFrame
    periods_df = pd.DataFrame(periods)
    return periods_df


def run_backtest_for_periods(
    periods_df,
    futures="VN30F1M",
    etf_list=None,
    etf_included=False,
    estimation_window=60,
    min_trading_days=45,
    max_clusters=10,
    top_stocks=5,
    correlation_threshold=0.6,
    residual_threshold=0.3,
    improvement_threshold=0.03,
    tier=1,
    first_allocation=0.4,
    adding_allocation=0.15,
    use_existing_data=False,
):
    """Run a backtest across all periods specified in periods_df.

    Args:
        periods_df (pd.DataFrame): DataFrame with ['stocks_list', 'start_date', 'end_date'] columns.
        futures (str): Futures symbol for backtesting. Defaults to 'VN30F1M'.
        etf_list (list): List of ETF ticker symbols to pass to DataHandler. Defaults to None.
        etf_included (bool): Whether to include ETFs in DataHandler. Defaults to False.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
            - combined_returns_df: Time series of daily returns.
            - combined_detail_df: Aggregated backtest details.
    """
    all_returns_dfs = []
    all_detail_dfs = []
    total_fee_ratio = 0.0
    # Run backtest for each period
    for period_idx, period in periods_df.iterrows():
        active_stocks = period["stocks_list"]
        start_date = period["start_date"]
        end_date = period["end_date"]

        # print(
        #     f"Running backtest for period {period_idx + 1}: {start_date} to {end_date} "
        # )

        # Initialize and run strategy with etf_list and etf_included
        data_handler = DataHandler(
            futures=futures,
            stocks=active_stocks,
            start_date=start_date,
            end_date=end_date,
            etf_list=etf_list,
            etf_included=etf_included,
            estimation_window=estimation_window,
            max_clusters=max_clusters,
            use_existing_data=use_existing_data,
        )
        strategy = Combination_Formations(
            data_handler,
            top_stocks=top_stocks,
            min_trading_days=min_trading_days,
            correlation_threshold=correlation_threshold,
            residual_threshold=residual_threshold,
            improvement_threshold=improvement_threshold,
        )
        strategy.run_strategy()
        results_df_1, stock_price = strategy.get_results()

        # Process results
        results_df, positions_df, trading_log = process_results_df(
            results_df_1,
            stock_price,
            stocks=active_stocks,
            ou_window=estimation_window,
            tier=tier,
            first_allocation=first_allocation,
            adding_allocation=adding_allocation,
        )

        # Run backtest
        pm = PortfolioManager(estimation_window=estimation_window)
        asset_df, detail_df, trading_log_df, fee_ratio = pm.run_backtest(
            positions_df, stock_price
        )

        # Compute and store returns
        returns_df = asset_df["balance"].pct_change().fillna(0).to_frame(name="returns")
        all_returns_dfs.append(returns_df)
        all_detail_dfs.append(detail_df)
        total_fee_ratio += fee_ratio
    average_fee_ratio = total_fee_ratio / len(periods_df)
    # Aggregate results
    combined_returns_df = pd.concat(all_returns_dfs).sort_index()
    combined_returns_df = combined_returns_df[
        ~combined_returns_df.index.duplicated(keep="first")
    ]

    combined_detail_df = pd.concat(all_detail_dfs).sort_index()
    combined_detail_df = combined_detail_df[
        ~combined_detail_df.index.duplicated(keep="first")
    ]

    return combined_returns_df, combined_detail_df, average_fee_ratio


def plot_asset_balance(asset_df: pd.DataFrame):
    """
    Plots the 'balance' column of the given asset DataFrame using matplotlib.

    Args:
        asset_df (pd.DataFrame): DataFrame containing a 'balance' column and a 'Date' index.
    """
    if "balance" not in asset_df.columns:
        print("Error: 'balance' column not found in the DataFrame.")
        return

    if not isinstance(asset_df.index, pd.DatetimeIndex):
        print(
            "Error: DataFrame index is not a DatetimeIndex. Ensure 'Date' is the index."
        )
        return

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    plt.plot(asset_df.index, asset_df["balance"], label="Balance")
    plt.title("Asset Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)  # Add grid lines for better readability
    plt.xticks(rotation=45)  # rotate x axis labels
    plt.tight_layout()  # avoid labels being cut off
    plt.show()
