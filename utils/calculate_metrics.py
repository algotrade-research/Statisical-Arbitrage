# utils/compute_metrics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data.get_data import get_etf_price
from tabulate import tabulate
import os

def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a series of returns.

    Args:
        returns_series (pd.Series): Series of returns.

    Returns:
        pd.Series: Cumulative returns.
    """
    return (1 + returns_series).cumprod()

def calculate_hpr(cumulative_returns: pd.Series) -> float:
    """Calculate the Holding Period Return (HPR).

    Args:
        cumulative_returns (pd.Series): Series of cumulative returns.

    Returns:
        float: HPR.
    """
    return cumulative_returns.iloc[-1] - 1

def calculate_annualized_return(hpr: float, time_length: float) -> float:
    """Calculate the annualized return.

    Args:
        hpr (float): Holding Period Return.
        time_length (float): Time length in years.

    Returns:
        float: Annualized return.
    """
    return (1 + hpr) ** (1 / time_length) - 1

def calculate_excess_hpr(strategy_hpr: float, benchmark_hpr: float) -> float:
    """Calculate the excess Holding Period Return (HPR).

    Args:
        strategy_hpr (float): HPR of the strategy.
        benchmark_hpr (float): HPR of the benchmark.

    Returns:
        float: Excess HPR.
    """
    return strategy_hpr - benchmark_hpr

def calculate_annual_excess_return(strategy_annual_return: float, benchmark_annual_return: float) -> float:
    """Calculate the annualized excess return.

    Args:
        strategy_annual_return (float): Annualized return of the strategy.
        benchmark_annual_return (float): Annualized return of the benchmark.

    Returns:
        float: Annualized excess return.
    """
    return strategy_annual_return - benchmark_annual_return

def calculate_volatility(returns_series: pd.Series, trading_days: int) -> float:
    """Calculate the annualized volatility.

    Args:
        returns_series (pd.Series): Series of returns.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Annualized volatility.
    """
    return returns_series.std() * np.sqrt(trading_days)

def calculate_drawdowns(cumulative_returns: pd.Series) -> pd.Series:
    """Calculate the drawdown series.

    Args:
        cumulative_returns (pd.Series): Series of cumulative returns.

    Returns:
        pd.Series: Drawdown series.
    """
    running_max = np.maximum.accumulate(cumulative_returns.dropna())
    running_max[running_max < 1] = 1
    drawdowns = (cumulative_returns / running_max) - 1
    return drawdowns

def calculate_max_drawdown(drawdowns: pd.Series) -> float:
    """Calculate the maximum drawdown.

    Args:
        drawdowns (pd.Series): Series of drawdowns.

    Returns:
        float: Maximum drawdown (positive value).
    """
    return -drawdowns.min()

def calculate_longest_drawdown(cumulative_returns: pd.Series) -> int:
    """Calculate the longest drawdown period in days.

    Args:
        cumulative_returns (pd.Series): Series of cumulative returns with datetime index.

    Returns:
        int: Longest drawdown period in days.
    """
    drawdowns = calculate_drawdowns(cumulative_returns)
    in_drawdown = drawdowns < 0
    drawdown_periods = []
    start = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start is None:
            start = i
        elif not is_dd and start is not None:
            drawdown_periods.append(i - start)
            start = None
    if start is not None:
        drawdown_periods.append(len(in_drawdown) - start)
    return max(drawdown_periods) if drawdown_periods else 0

def calculate_sharpe_ratio(annual_return: float, volatility: float, risk_free_rate: float) -> float:
    """Calculate the Sharpe Ratio.

    Args:
        annual_return (float): Annualized return.
        volatility (float): Annualized volatility.
        risk_free_rate (float): Annual risk-free rate.

    Returns:
        float: Sharpe Ratio.
    """
    return (annual_return - risk_free_rate) / volatility if volatility != 0 else np.nan

def calculate_downside_deviation(returns_series: pd.Series, trading_days: int) -> float:
    """Calculate the annualized downside deviation.

    Args:
        returns_series (pd.Series): Series of returns.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Annualized downside deviation.
    """
    downward = returns_series[returns_series < 0]
    return downward.std() * np.sqrt(trading_days) if not downward.empty else 0

def calculate_sortino_ratio(annual_return: float, downside_deviation: float) -> float:
    """Calculate the Sortino Ratio.

    Args:
        annual_return (float): Annualized return.
        downside_deviation (float): Annualized downside deviation.

    Returns:
        float: Sortino Ratio.
    """
    return annual_return / downside_deviation if downside_deviation != 0 else np.nan

def calculate_var(annual_return: float, volatility: float, confidence_level: float = 0.01, n_simulations: int = 100000) -> float:
    """Calculate the Value at Risk (VaR) using normal distribution simulations.

    Args:
        annual_return (float): Annualized return.
        volatility (float): Annualized volatility.
        confidence_level (float): Confidence level for VaR. Defaults to 0.01.
        n_simulations (int): Number of simulations. Defaults to 100,000.

    Returns:
        float: VaR value (positive).
    """
    simulations = np.random.normal(annual_return, volatility, size=n_simulations)
    sorted_simulations = np.sort(simulations)
    var_index = int(n_simulations * confidence_level)
    return -sorted_simulations[var_index]

def calculate_cvar(annual_return: float, volatility: float, confidence_level: float = 0.01, n_simulations: int = 100000) -> float:
    """Calculate the Conditional Value at Risk (cVaR) using normal distribution simulations.

    Args:
        annual_return (float): Annualized return.
        volatility (float): Annualized volatility.
        confidence_level (float): Confidence level for cVaR. Defaults to 0.01.
        n_simulations (int): Number of simulations. Defaults to 100,000.

    Returns:
        float: cVaR value (positive).
    """
    simulations = np.random.normal(annual_return, volatility, size=n_simulations)
    sorted_simulations = np.sort(simulations)
    var_index = int(n_simulations * confidence_level)
    return -sorted_simulations[:var_index].mean()

def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate the beta of the asset relative to the benchmark.

    Args:
        asset_returns (pd.Series): Series of asset returns.
        benchmark_returns (pd.Series): Series of benchmark returns.

    Returns:
        float: Beta value.
    """
    cov = np.cov(asset_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    return cov / var if var != 0 else np.nan

def calculate_alpha(annual_return: float, beta: float, benchmark_annual_return: float) -> float:
    """Calculate the alpha of the asset.

    Args:
        annual_return (float): Annualized return of the asset.
        beta (float): Beta of the asset.
        benchmark_annual_return (float): Annualized return of the benchmark.

    Returns:
        float: Alpha value.
    """
    return annual_return - beta * benchmark_annual_return if not np.isnan(beta) else np.nan

def calculate_tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series, trading_days: int) -> float:
    """Calculate the annualized tracking error.

    Args:
        asset_returns (pd.Series): Series of asset returns.
        benchmark_returns (pd.Series): Series of benchmark returns.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Annualized tracking error.
    """
    excess_returns = asset_returns - benchmark_returns
    return excess_returns.std() * np.sqrt(trading_days)

def calculate_information_ratio(annual_excess_return: float, tracking_error: float) -> float:
    """Calculate the Information Ratio.

    Args:
        annual_excess_return (float): Annualized excess return.
        tracking_error (float): Annualized tracking error.

    Returns:
        float: Information Ratio.
    """
    return annual_excess_return / tracking_error if tracking_error != 0 else np.nan

def plot_cumulative_returns(strategy_cum_rets: pd.Series, benchmark_cum_rets: pd.Series) -> None:
    """Plot cumulative returns for strategy and benchmark.

    Args:
        strategy_cum_rets (pd.Series): Cumulative returns of the strategy.
        benchmark_cum_rets (pd.Series): Cumulative returns of the benchmark.
    """
    log_scale = strategy_cum_rets.max() > 5 or benchmark_cum_rets.max() > 5
    plt.figure(figsize=(15, 8))
    plt.plot(strategy_cum_rets, color="#035593", linewidth=3, label="Strategy")
    plt.plot(benchmark_cum_rets, color="#068C72", linewidth=3, label="VN30")
    if log_scale:
        plt.yscale("log")
        y_ticks = [1, 2, 5, 10]
        plt.yticks(y_ticks, labels=[str(int(tick)) for tick in y_ticks])
    plt.title("CUMULATIVE RETURN", size=15)
    plt.ylabel("Cumulative return", size=15)
    plt.xticks(size=15, fontweight="bold")
    plt.yticks(size=15, fontweight="bold")
    plt.legend(fontsize=15)
    plt.show()
def calculate_turnover(total_fee: float) -> float:
    """Calculate the turnover ratio based on total fees.
    Because each sell/buy transaction costs 0.23% of the total amount, we can calculate the turnover ratio as follows:
    """
    return total_fee / (.23/100)/252
def plot_drawdown(drawdowns: pd.Series, title: str) -> None:
    """Plot the drawdown series.

    Args:
        drawdowns (pd.Series): Series of drawdowns.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(15, 8))
    plt.fill_between(drawdowns.index, drawdowns * 100, 0, color="#CE5151")
    plt.plot(drawdowns.index, drawdowns * 100, color="#930303", linewidth=1.5)
    plt.title(title, size=15)
    plt.ylabel("Drawdown %", size=15)
    plt.xticks(size=15, fontweight="bold")
    plt.yticks(size=15, fontweight="bold")
    plt.show()



def calculate_metrics(
    returns_df: pd.DataFrame,
    total_fee_ratio,
    risk_free_rate: float = 0.05,
    trading_day: int = 252,
    freq: str = "D",
    plotting: bool = False,
    use_benchmark: bool = True,
    use_existing_data: bool = True
) -> pd.DataFrame:
    """Calculate performance metrics, plot cumulative returns, and drawdown for portfolio returns vs a benchmark.

    Args:
        returns_df (pd.DataFrame): Portfolio returns with 'Date' index and 'returns' column.
        total_fee_ratio (float): Total fee ratio for turnover calculation.
        risk_free_rate (float): Annual risk-free rate. Defaults to 0.05.
        trading_day (int): Number of trading days per year for annualization. Defaults to 252.
        freq (str): Frequency of the data ('D' for daily). Defaults to 'D'.
        plotting (bool): Whether to plot cumulative returns and drawdown. Defaults to False.
        use_benchmark (bool): Whether to include benchmark comparison. Defaults to True.
        use_existing_data (bool): Whether to use existing benchmark data from file. Defaults to True.

    Returns:
        pd.DataFrame: Metrics for Strategy and optionally VN30, with percentages where applicable.

    Raises:
        ValueError: If returns_df is empty, required columns are missing, or benchmark data is invalid.
    """
    # Validate inputs
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    # Ensure datetime index
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)

    # Initialize benchmark variables
    benchmark_returns = None
    benchmark_df = None

    if use_benchmark:
        # Get the directory structure
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)  # Go up to project2
        data_folder = os.path.join(project_root, "data")
        os.makedirs(data_folder, exist_ok=True)
        csv_file = os.path.join(data_folder, "index_price.csv")

        if use_existing_data and os.path.exists(csv_file):
            # Load existing benchmark data
            benchmark_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if 'price' not in benchmark_df.columns:
                raise ValueError("vn30_prices.csv must contain a 'price' column.")
            benchmark_df.set_index('datetime', inplace=True)
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
        else:
            # Fetch benchmark data (assuming get_etf_price is available)
            start = returns_df.index[0].strftime('%Y-%m-%d')
            end = returns_df.index[-1].strftime('%Y-%m-%d')
            benchmark_df = get_etf_price('VN30', start, end)
            benchmark_df.set_index('datetime', inplace=True)
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            if 'price' not in benchmark_df.columns:
                raise ValueError("Benchmark DataFrame must contain a 'price' column.")
            # Optionally save to file for future use
            benchmark_df.to_csv(csv_file)

        # Align and calculate benchmark returns
        benchmark_df = benchmark_df.reindex(returns_df.index, method='ffill')
        benchmark_returns = benchmark_df['price'].pct_change().fillna(0)

    # Calculate time length
    time_length = (returns_df.index[-1] - returns_df.index[0]).days / 365.25

    # Cumulative returns
    strategy_cum_rets = calculate_cumulative_returns(returns_df['returns'])
    benchmark_cum_rets = calculate_cumulative_returns(benchmark_returns) if use_benchmark else None

    # HPR
    strategy_hpr = calculate_hpr(strategy_cum_rets)
    benchmark_hpr = calculate_hpr(benchmark_cum_rets) if use_benchmark else None
    excess_hpr = calculate_excess_hpr(strategy_hpr, benchmark_hpr) if use_benchmark else None

    # Annualized returns
    strategy_annual_return = calculate_annualized_return(strategy_hpr, time_length)
    benchmark_annual_return = calculate_annualized_return(benchmark_hpr, time_length) if use_benchmark else None
    annual_excess_return = calculate_annual_excess_return(strategy_annual_return, benchmark_annual_return) if use_benchmark else None

    # Volatility
    strategy_volatility = calculate_volatility(returns_df['returns'], trading_day)
    benchmark_volatility = calculate_volatility(benchmark_returns, trading_day) if use_benchmark else None

    # Drawdowns and related metrics
    strategy_drawdowns = calculate_drawdowns(strategy_cum_rets)
    strategy_max_drawdown = calculate_max_drawdown(strategy_drawdowns)
    strategy_longest_drawdown = calculate_longest_drawdown(strategy_cum_rets)
    if use_benchmark:
        benchmark_drawdowns = calculate_drawdowns(benchmark_cum_rets)
        benchmark_max_drawdown = calculate_max_drawdown(benchmark_drawdowns)
        benchmark_longest_drawdown = calculate_longest_drawdown(benchmark_cum_rets)
    else:
        benchmark_drawdowns = benchmark_max_drawdown = benchmark_longest_drawdown = None

    # Risk-adjusted metrics
    strategy_sharpe = calculate_sharpe_ratio(strategy_annual_return, strategy_volatility, risk_free_rate)
    strategy_downside_dev = calculate_downside_deviation(returns_df['returns'], trading_day)
    strategy_sortino = calculate_sortino_ratio(strategy_annual_return, strategy_downside_dev)
    if use_benchmark:
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_annual_return, benchmark_volatility, risk_free_rate)
        benchmark_downside_dev = calculate_downside_deviation(benchmark_returns, trading_day)
        benchmark_sortino = calculate_sortino_ratio(benchmark_annual_return, benchmark_downside_dev)
    else:
        benchmark_sharpe = benchmark_downside_dev = benchmark_sortino = None

    # Benchmark comparison metrics
    beta = calculate_beta(returns_df['returns'], benchmark_returns) if use_benchmark else None
    alpha = calculate_alpha(strategy_annual_return, beta, benchmark_annual_return) if use_benchmark else None
    tracking_error = calculate_tracking_error(returns_df['returns'], benchmark_returns, trading_day) if use_benchmark else None
    information_ratio = calculate_information_ratio(annual_excess_return, tracking_error) if use_benchmark else None

    # VaR and cVaR
    strategy_var = calculate_var(strategy_annual_return, strategy_volatility)
    strategy_cvar = calculate_cvar(strategy_annual_return, strategy_volatility)
    if use_benchmark:
        benchmark_var = calculate_var(benchmark_annual_return, benchmark_volatility)
        benchmark_cvar = calculate_cvar(benchmark_annual_return, benchmark_volatility)
    else:
        benchmark_var = benchmark_cvar = None

    # Turnover ratio
    turnover = calculate_turnover(total_fee_ratio)

    # Compile metrics
    metrics_data = {
        'HPR (%)': [f"{strategy_hpr * 100:.2f}%", f"{benchmark_hpr * 100:.2f}%" if use_benchmark else "-"],
        'Excess HPR (%)': [f"{excess_hpr * 100:.2f}%" if use_benchmark else "-", "-"],
        'Annual Return (%)': [f"{strategy_annual_return * 100:.2f}%", f"{benchmark_annual_return * 100:.2f}%" if use_benchmark else "-"],
        'Annual Excess Return (%)': [f"{annual_excess_return * 100:.2f}%" if use_benchmark else "-", "-"],
        'Volatility (%)': [f"{strategy_volatility * 100:.2f}%", f"{benchmark_volatility * 100:.2f}%" if use_benchmark else "-"],
        'Maximum Drawdown (%)': [f"{strategy_max_drawdown * 100:.2f}%", f"{benchmark_max_drawdown * 100:.2f}%" if use_benchmark else "-"],
        'Longest Drawdown (days)': [f"{strategy_longest_drawdown:.0f}", f"{benchmark_longest_drawdown:.0f}" if use_benchmark else "-"],
        'Sharpe Ratio': [f"{strategy_sharpe:.2f}", f"{benchmark_sharpe:.2f}" if use_benchmark else "-"],
        'Sortino Ratio': [f"{strategy_sortino:.2f}", f"{benchmark_sortino:.2f}" if use_benchmark else "-"],
        'Information Ratio': [f"{information_ratio:.2f}" if use_benchmark else "-", "-"],
        'Beta': [f"{beta:.2f}" if use_benchmark else "-", "-"],
        'Alpha (%)': [f"{alpha * 100:.2f}%" if use_benchmark else "-", "-"],
        'Turnover Ratio (%)': [f"{turnover * 100:.2f}%", "-"],
        'VaR (%)': [f"{strategy_var * 100:.2f}%", f"{benchmark_var * 100:.2f}%" if use_benchmark else "-"],
        'cVaR (%)': [f"{strategy_cvar * 100:.2f}%", f"{benchmark_cvar * 100:.2f}%" if use_benchmark else "-"],
        'VaR/cVaR': [f"{(strategy_cvar / strategy_var if strategy_var != 0 else np.nan):.2f}",
                     f"{(benchmark_cvar / benchmark_var if benchmark_var != 0 else np.nan):.2f}" if use_benchmark else "-"]
    }
    metrics_df = pd.DataFrame(metrics_data, index=['Strategy', 'VN30'] if use_benchmark else ['Strategy'])

    # Prepare data for vertical table
    table_data = []
    for metric in metrics_data.keys():
        if use_benchmark:
            table_data.append([metric, metrics_df.loc['Strategy', metric], metrics_df.loc['VN30', metric]])
        else:
            table_data.append([metric, metrics_df.loc['Strategy', metric]])

    # Print metrics in a vertical table format
    print("\nMetrics for Strategy" + (" and VN30 Benchmark" if use_benchmark else "") + "\n" + "="*40)
    headers = ['Metric', 'Strategy', 'VN30'] if use_benchmark else ['Metric', 'Strategy']
    print(tabulate(table_data, headers=headers, tablefmt='psql', stralign='left'))

    # Plotting
    if plotting:
        plot_cumulative_returns(strategy_cum_rets, benchmark_cum_rets)
        # plot_drawdown(strategy_drawdowns, "DRAWDOWN (Strategy)")

    return metrics_df

def calculate_monthly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns from a time series of daily returns.

    Args:
        returns_df (pd.DataFrame): Portfolio returns with 'Date' index and 'returns' column.

    Returns:
        pd.DataFrame: Monthly returns with 'Year', 'Month', and 'Monthly Return' columns.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df['Year'] = returns_df.index.year
    returns_df['Month'] = returns_df.index.month

    monthly_returns = []
    for (year, month), group in returns_df.groupby(['Year', 'Month']):
        cum_return = (1 + group['returns']).prod() - 1
        monthly_returns.append({'Year': year, 'Month': month, 'Monthly Return': cum_return})

    return pd.DataFrame(monthly_returns).sort_values(['Year', 'Month'])

def calculate_yearly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly returns from a time series of daily returns.

    Args:
        returns_df (pd.DataFrame): Portfolio returns with 'Date' index and 'returns' column.

    Returns:
        pd.DataFrame: Yearly returns with 'Year' and 'Yearly Return' columns.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df['Year'] = returns_df.index.year

    yearly_returns = []
    for year, group in returns_df.groupby('Year'):
        cum_return = (1 + group['returns']).prod() - 1
        yearly_returns.append({'Year': year, 'Yearly Return': cum_return})


def pivot_monthly_returns_to_table(monthly_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly returns into a table with years as columns and months as rows.

    Args:
        monthly_returns_df (pd.DataFrame): DataFrame with 'Year', 'Month', and 'Monthly Return' columns.

    Returns:
        pd.DataFrame: Pivoted table with months (JAN-DEC) as rows and years as columns.
    """
    if monthly_returns_df.empty:
        raise ValueError("monthly_returns_df is empty.")
    if not all(col in monthly_returns_df.columns for col in ['Year', 'Month', 'Monthly Return']):
        raise ValueError("monthly_returns_df must contain 'Year', 'Month', and 'Monthly Return' columns.")

    # Map month numbers to month names
    month_map = {
        1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
        7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
    }
    monthly_returns_df = monthly_returns_df.copy()
    monthly_returns_df['Month'] = monthly_returns_df['Month'].map(month_map)

    # Add a yearly row for totals
    yearly_returns = monthly_returns_df.groupby('Year')['Monthly Return'].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    yearly_returns['Month'] = 'YEARLY'
    yearly_returns = yearly_returns[['Year', 'Month', 'Monthly Return']]

    # Concatenate the yearly returns with the monthly returns
    monthly_returns_df = pd.concat([yearly_returns, monthly_returns_df], ignore_index=True)

    # Pivot the table
    pivoted_df = monthly_returns_df.pivot(index='Month', columns='Year', values='Monthly Return')

    # Ensure all months and the 'YEARLY' row are present in the correct order
    desired_index = ['YEARLY', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    pivoted_df = pivoted_df.reindex(desired_index)

    # Format the values as percentages with 3 decimal places
    pivoted_df = pivoted_df.map(lambda x: f"{x:.2%}" if pd.notnull(x) else x)

    return pivoted_df


def calculate_shapre_and_mdd(returns_df: pd.DataFrame, risk_free_rate: float = 0.05, trading_day: int = 252, freq: str = "D") -> pd.DataFrame:
    """Calculate performance metrics, plot cumulative returns, and drawdown for portfolio returns vs a benchmark.

    Note: Despite 'benchmark_df' in the docstring, benchmark data is fetched internally using get_etf_price('VN30', start, end).

    Args:
        returns_df (pd.DataFrame): Portfolio returns with 'Date' index and 'returns' column.
        risk_free_rate (float): Annual risk-free rate. Defaults to 0.05.
        trading_day (int): Number of trading days per year for annualization. Defaults to 252.
        freq (str): Frequency of the data ('D' for daily). Defaults to 'D'.

    Returns:
        pd.DataFrame: Metrics for both Strategy and VN30, with percentages where applicable.

    Raises:
        ValueError: If returns_df is empty or if required columns are missing.
    """
    # Validate inputs
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    # Ensure datetime index
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    # Fetch benchmark data
    start = returns_df.index[0].strftime('%Y-%m-%d')
    end = returns_df.index[-1].strftime('%Y-%m-%d')
    # Calculate time length
    time_length = (returns_df.index[-1] - returns_df.index[0]).days / 365.25
    # Cumulative returns
    strategy_cum_rets = calculate_cumulative_returns(returns_df['returns'])
    # HPR
    strategy_hpr = calculate_hpr(strategy_cum_rets)
    # Annualized returns
    strategy_annual_return = calculate_annualized_return(strategy_hpr, time_length)
    # Volatility
    strategy_volatility = calculate_volatility(returns_df['returns'], trading_day)
    # Drawdowns and related metrics
    strategy_drawdowns = calculate_drawdowns(strategy_cum_rets)
    strategy_max_drawdown = calculate_max_drawdown(strategy_drawdowns)
    # Risk-adjusted metrics
    strategy_sharpe = calculate_sharpe_ratio(strategy_annual_return, strategy_volatility, risk_free_rate)
    return strategy_annual_return,strategy_sharpe, strategy_max_drawdown
