import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from datetime import timedelta,datetime
from helper import *
from .ou_process import OuProcess
from .get_allocation_tier import * 


# # Place your modified compute_allocations and process_results_df here, followed by the rest of your code
# def generate_signals(residuals_pivot: pd.DataFrame, ou_window: int = 60) -> pd.DataFrame:
#     """Generates OU-based trading signals from residuals.

#     Args:
#         residuals_pivot (pd.DataFrame): Pivot table of residuals with dates as index and combination IDs as columns.
#         ou_window (int, optional): Window size for OU process fitting. Defaults to 60.

#     Returns:
#         pd.DataFrame: DataFrame with OU parameters (kappa, m, sigma, s_score) for each combination and date.
#     """
#     signal_gen = OuProcess(residuals_pivot, ou_window=ou_window)
#     signal_gen.apply_ou_fitting()
#     return signal_gen.ou_params

# def compute_allocations(ou_params: pd.DataFrame, residuals_pivot: pd.DataFrame, ou_window: int = 60, tier: int = 1) -> pd.DataFrame:
#     """Computes allocation percentages based on OU s-scores using the specified tier.

#     Args:
#         ou_params (pd.DataFrame): DataFrame with OU parameters (kappa, m, sigma, s_score).
#         residuals_pivot (pd.DataFrame): Pivot table of residuals.
#         ou_window (int, optional): Window size for OU process fitting. Defaults to 60.
#         tier (int, optional): Allocation tier to use (1-5). Defaults to 1.

#     Returns:
#         pd.DataFrame: DataFrame with allocation percentages for each combination and date.
#     """
#     allocation_func = allocation_functions.get(tier, get_allocation_tier_1)  # Default to tier 1 if tier is invalid
#     allocation_percentages = pd.DataFrame(index=ou_params.index, columns=residuals_pivot.columns, dtype=float).fillna(0.0)
#     trend_tracker = {comb_id: False for comb_id in residuals_pivot.columns}

#     for comb_id in allocation_percentages.columns:
#         s_scores = ou_params[(comb_id, 's_score')]
#         prev_allocation = 0.0
#         prev_s_score = np.nan
#         for i, date in enumerate(s_scores.index):
#             if i < ou_window:
#                 allocation = 0.0
#             else:
#                 s_score = s_scores[date]
#                 if pd.isna(s_score) or pd.isna(residuals_pivot.loc[date, comb_id]):
#                     allocation = 0.0
#                 else:
#                     is_decreasing = s_score < prev_s_score if not pd.isna(prev_s_score) else False
#                     trend_tracker[comb_id] = is_decreasing
#                     allocation = allocation_func(s_score, prev_allocation, prev_s_score, trend_tracker[comb_id])
#                     prev_s_score = s_score if not pd.isna(s_score) else prev_s_score
#             allocation_percentages.loc[date, comb_id] = allocation
#             prev_allocation = allocation
#     return allocation_percentages
# def calculate_positions(allocation_percentages: pd.DataFrame, results_df: pd.DataFrame, stock_price: pd.DataFrame, stocks: list) -> pd.DataFrame:
#     """Calculates trading positions based on allocation percentages.

#     Args:
#         allocation_percentages (pd.DataFrame): DataFrame with allocation percentages.
#         results_df (pd.DataFrame): DataFrame with combination results (betas, residuals, etc.).
#         stock_price (pd.DataFrame): DataFrame with stock and futures prices.
#         stocks (list): List of stock symbols.

#     Returns:
#         pd.DataFrame: DataFrame with positions for VN30F1M and each stock, including absolute values.
#     """
#     dates = results_df['Date'].sort_values().unique()
#     columns = ['Total_Port_Trading', 'VN30F1M_Position'] + [f'{stock}_Position' for stock in stocks] + ['Num_Active_Combinations', 'Active_Combination_IDs']
#     positions_df = pd.DataFrame(index=dates, columns=columns, dtype=float)
#     positions_df['Active_Combination_IDs'] = positions_df['Active_Combination_IDs'].astype(object)
#     positions_df = positions_df.fillna(0.0)

#     for date in dates:
#         if date not in allocation_percentages.index:
#             continue
#         active_combs = allocation_percentages.loc[date][allocation_percentages.loc[date] > 0]
#         num_active = len(active_combs)
#         active_ids = list(active_combs.index)
#         positions_df.loc[date, 'Num_Active_Combinations'] = num_active
#         positions_df.loc[date, 'Active_Combination_IDs'] = str(active_ids)

#         if num_active == 0:
#             total_allocation = 0.0
#         else:
#             base_allocation = min(0.4 + 0.1 * (num_active - 1), .98)
#             intended_allocations = active_combs * base_allocation
#             total_intended = intended_allocations.sum()
#             scale_factor = .98 / total_intended if total_intended > .98 else .98
#             scaled_allocations = intended_allocations * scale_factor
#             total_allocation = scaled_allocations.sum()

#         positions_df.loc[date, 'Total_Port_Trading'] = total_allocation
#         positions_df.loc[date, 'VN30F1M_Position'] = -total_allocation * 0.20  # 20% short

#         stock_allocation = total_allocation * 0.80  # 80% to stocks
#         for comb_id in active_combs.index:
#             comb_allocation = scaled_allocations[comb_id] * stock_allocation / total_allocation if total_allocation > 0 else 0
#             comb_row = results_df[(results_df['Date'] == date) & (results_df['Combination_ID'] == comb_id)]
#             if comb_row.empty or date not in stock_price.index:
#                 continue

#             comb_stocks = [s for s in stocks if f'Beta_{s}' in comb_row.columns and comb_row[f'Beta_{s}'].values[0] >= 0]
#             sum_beta_price = 0.0
#             valid_comb_stocks = []
#             for s in comb_stocks:
#                 beta = comb_row[f'Beta_{s}'].values[0]
#                 price = stock_price.loc[date, s] if s in stock_price.columns and not pd.isna(stock_price.loc[date, s]) else np.nan
#                 if not pd.isna(price):
#                     sum_beta_price += beta * price
#                     valid_comb_stocks.append(s)

#             if sum_beta_price > 0:
#                 for stock in valid_comb_stocks:
#                     beta = comb_row[f'Beta_{stock}'].values[0]
#                     current_price = stock_price.loc[date, stock]
#                     stock_proportion = (beta * current_price) / sum_beta_price
#                     stock_position = comb_allocation * stock_proportion
#                     positions_df.loc[date, f'{stock}_Position'] += stock_position

#     return positions_df
# def process_results_df(results_df: pd.DataFrame, stock_price: pd.DataFrame, stocks: list, ou_window: int = 60, tier: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Processes the results DataFrame to generate signals, allocations, positions, and trading logs.

#     Args:
#         results_df (pd.DataFrame): DataFrame with combination results.
#         stock_price (pd.DataFrame): DataFrame with stock and futures prices.
#         stocks (list): List of stock symbols.
#         ou_window (int, optional): Window size for OU process fitting. Defaults to 60.
#         tier (int, optional): Allocation tier to use (1-5). Defaults to 1.

#     Returns:
#         tuple: (results_df, positions_df, trading_log) where:
#             - results_df: Updated results with s-scores and allocations.
#             - positions_df: DataFrame with trading positions.
#             - trading_log: DataFrame with trading actions and deltas.
#     """
#     results_df = results_df.sort_values('Date')
#     residuals_pivot = results_df.pivot(index='Date', columns='Combination_ID', values='Residual')
#     ou_params = generate_signals(residuals_pivot, ou_window=ou_window)
#     allocation_percentages = compute_allocations(ou_params, residuals_pivot, ou_window=ou_window, tier=tier)
#     positions_df = calculate_positions(allocation_percentages, results_df, stock_price, stocks)

#     # Add s-scores and allocations to results_df
#     results_df['s_score'] = results_df.apply(
#         lambda row: ou_params.loc[row['Date'], (row['Combination_ID'], 's_score')]
#         if row['Date'] in ou_params.index else np.nan, axis=1
#     )
#     results_df['Allocation'] = results_df.apply(
#         lambda row: allocation_percentages.loc[row['Date'], row['Combination_ID']]
#         if row['Date'] in allocation_percentages.index else 0.0, axis=1
#     )

#     # Add absolute values to positions_df
#     for date in positions_df.index:
#         if date in stock_price.index and not pd.isna(stock_price.loc[date, 'VN30F1M']):
#             vn30_pos = positions_df.loc[date, 'VN30F1M_Position']
#             positions_df.loc[date, 'Abs_VN30F1M'] = abs(vn30_pos) * stock_price.loc[date, 'VN30F1M']
#             total_abs_stocks = 0.0
#             for stock in stocks:
#                 stock_pos = positions_df.loc[date, f'{stock}_Position']
#                 if stock_pos > 0:
#                     stock_price_val = stock_price.loc[date, stock]
#                     positions_df.loc[date, f'Abs_{stock}'] = stock_pos * stock_price_val
#                     total_abs_stocks += stock_pos * stock_price_val
#                 else:
#                     positions_df.loc[date, f'Abs_{stock}'] = 0.0
#             positions_df.loc[date, 'Abs_Stocks'] = total_abs_stocks
#         else:
#             positions_df.loc[date, 'Abs_VN30F1M'] = np.nan
#             positions_df.loc[date, 'Abs_Stocks'] = np.nan
#             for stock in stocks:
#                 positions_df.loc[date, f'Abs_{stock}'] = np.nan

#     # Generate trading log
#     trading_log = pd.DataFrame(index=positions_df.index)
#     trading_log['Total_Port_Trading'] = positions_df['Total_Port_Trading']
#     trading_log['Delta_VN30F1M'] = positions_df['VN30F1M_Position'].diff().fillna(0.0)
#     trading_log['Action_VN30F1M'] = np.where(
#         (trading_log['Delta_VN30F1M'] > 0) & (positions_df['VN30F1M_Position'].shift(1).fillna(0.0) < 0), 'buy to cover',
#         np.where(trading_log['Delta_VN30F1M'] < 0, 'sell short', 'hold')
#     )
#     for stock in stocks:
#         pos_col = f'{stock}_Position'
#         delta_col = f'Delta_{stock}'
#         action_col = f'Action_{stock}'
#         trading_log[delta_col] = positions_df[pos_col].diff().fillna(0.0)
#         trading_log[action_col] = np.where(
#             trading_log[delta_col] > 0, 'buy',
#             np.where(trading_log[delta_col] < 0, 'sell', 'hold')
#         )
#     trading_log['Num_Active_Combinations'] = positions_df['Num_Active_Combinations']
#     trading_log['Active_Combination_IDs'] = positions_df['Active_Combination_IDs']

#     # Round values
#     for df in [positions_df, trading_log]:
#         for col in df.columns:
#             if col.startswith(('Total_Port_Trading', 'VN30F1M_Position', 'Delta_VN30F1M', 'Abs_VN30F1M', 'Abs_Stocks')) or col.endswith('_Position') or col.startswith('Delta_') or col.startswith('Abs_'):
#                 df[col] = df[col].apply(lambda x: round(x, 4) if pd.notna(x) and abs(x) > 1e-10 else 0.0)

#     # Sort indices
#     results_df.sort_values(by=['Combination_ID', 'Date'], inplace=True)
#     positions_df.sort_index(inplace=True)
#     trading_log.sort_index(inplace=True)

#     return results_df, positions_df, trading_log
#     """Processes the results DataFrame to generate signals, allocations, positions, and trading logs.

#     Args:
#         results_df (pd.DataFrame): DataFrame with combination results.
#         stock_price (pd.DataFrame): DataFrame with stock and futures prices.
#         stocks (list): List of stock symbols.
#         ou_window (int, optional): Window size for OU process fitting. Defaults to 60.

#     Returns:
#         tuple: (results_df, positions_df, trading_log) where:
#             - results_df: Updated results with s-scores and allocations.
#             - positions_df: DataFrame with trading positions.
#             - trading_log: DataFrame with trading actions and deltas.
#     """
#     results_df = results_df.sort_values('Date')
#     residuals_pivot = results_df.pivot(index='Date', columns='Combination_ID', values='Residual')
#     ou_params = generate_signals(residuals_pivot, ou_window=ou_window)
#     allocation_percentages = compute_allocations(ou_params, residuals_pivot, ou_window=ou_window)
#     positions_df = calculate_positions(allocation_percentages, results_df, stock_price, stocks)

#     # Add s-scores and allocations to results_df
#     results_df['s_score'] = results_df.apply(
#         lambda row: ou_params.loc[row['Date'], (row['Combination_ID'], 's_score')]
#         if row['Date'] in ou_params.index else np.nan, axis=1
#     )
#     results_df['Allocation'] = results_df.apply(
#         lambda row: allocation_percentages.loc[row['Date'], row['Combination_ID']]
#         if row['Date'] in allocation_percentages.index else 0.0, axis=1
#     )

#     # Add absolute values to positions_df
#     for date in positions_df.index:
#         if date in stock_price.index and not pd.isna(stock_price.loc[date, 'VN30F1M']):
#             vn30_pos = positions_df.loc[date, 'VN30F1M_Position']
#             positions_df.loc[date, 'Abs_VN30F1M'] = abs(vn30_pos) * stock_price.loc[date, 'VN30F1M']
#             total_abs_stocks = 0.0
#             for stock in stocks:
#                 stock_pos = positions_df.loc[date, f'{stock}_Position']
#                 if stock_pos > 0:
#                     stock_price_val = stock_price.loc[date, stock]
#                     positions_df.loc[date, f'Abs_{stock}'] = stock_pos * stock_price_val
#                     total_abs_stocks += stock_pos * stock_price_val
#                 else:
#                     positions_df.loc[date, f'Abs_{stock}'] = 0.0
#             positions_df.loc[date, 'Abs_Stocks'] = total_abs_stocks
#         else:
#             positions_df.loc[date, 'Abs_VN30F1M'] = np.nan
#             positions_df.loc[date, 'Abs_Stocks'] = np.nan
#             for stock in stocks:
#                 positions_df.loc[date, f'Abs_{stock}'] = np.nan

#     # Generate trading log
#     trading_log = pd.DataFrame(index=positions_df.index)
#     trading_log['Total_Port_Trading'] = positions_df['Total_Port_Trading']
#     trading_log['Delta_VN30F1M'] = positions_df['VN30F1M_Position'].diff().fillna(0.0)
#     trading_log['Action_VN30F1M'] = np.where(
#         (trading_log['Delta_VN30F1M'] > 0) & (positions_df['VN30F1M_Position'].shift(1).fillna(0.0) < 0), 'buy to cover',
#         np.where(trading_log['Delta_VN30F1M'] < 0, 'sell short', 'hold')
#     )
#     for stock in stocks:
#         pos_col = f'{stock}_Position'
#         delta_col = f'Delta_{stock}'
#         action_col = f'Action_{stock}'
#         trading_log[delta_col] = positions_df[pos_col].diff().fillna(0.0)
#         trading_log[action_col] = np.where(
#             trading_log[delta_col] > 0, 'buy',
#             np.where(trading_log[delta_col] < 0, 'sell', 'hold')
#         )
#     trading_log['Num_Active_Combinations'] = positions_df['Num_Active_Combinations']
#     trading_log['Active_Combination_IDs'] = positions_df['Active_Combination_IDs']

#     # Round values
#     for df in [positions_df, trading_log]:
#         for col in df.columns:
#             if col.startswith(('Total_Port_Trading', 'VN30F1M_Position', 'Delta_VN30F1M', 'Abs_VN30F1M', 'Abs_Stocks')) or col.endswith('_Position') or col.startswith('Delta_') or col.startswith('Abs_'):
#                 df[col] = df[col].apply(lambda x: round(x, 4) if pd.notna(x) and abs(x) > 1e-10 else 0.0)

#     # Sort indices
#     results_df.sort_values(by=['Combination_ID', 'Date'], inplace=True)
#     positions_df.sort_index(inplace=True)
#     trading_log.sort_index(inplace=True)

#     return results_df, positions_df, trading_log

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from typing import Dict, Tuple, List
import logging


from .get_allocation_tier import allocation_functions, get_allocation_tier_1


def generate_signals(residuals_pivot: pd.DataFrame, ou_window: int = 60) -> pd.DataFrame:
    """Generates OU-based trading signals from residuals.

    Args:
        residuals_pivot (pd.DataFrame): Pivot table of residuals with dates as index and combination IDs as columns.
        ou_window (int, optional): Window size for OU process fitting. Defaults to 60.

    Returns:
        pd.DataFrame: DataFrame with OU parameters (kappa, m, sigma, s_score) for each combination and date.
    """
    logging.info(f"Generating signals with ou_window={ou_window}")
    signal_gen = OuProcess(residuals_pivot, ou_window=ou_window)
    signal_gen.apply_ou_fitting()
    return signal_gen.ou_params

def compute_allocations(ou_params: pd.DataFrame, residuals_pivot: pd.DataFrame, ou_window: int = 60, tier: int = 1) -> pd.DataFrame:
    """Computes allocation percentages based on OU s-scores using the specified tier.

    Args:
        ou_params (pd.DataFrame): DataFrame with OU parameters (kappa, m, sigma, s_score).
        residuals_pivot (pd.DataFrame): Pivot table of residuals.
        ou_window (int, optional): Window size for OU process fitting. Defaults to 60.
        tier (int, optional): Allocation tier to use (1-7). Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame with allocation percentages for each combination and date.
    """
    
    allocation_func = allocation_functions.get(tier, get_allocation_tier_1)
    allocation_percentages = pd.DataFrame(index=ou_params.index, columns=residuals_pivot.columns, dtype=float).fillna(0.0)
    trend_tracker = {comb_id: False for comb_id in residuals_pivot.columns}
    peak_s_scores = {comb_id: 0.0 for comb_id in residuals_pivot.columns}

    for comb_id in allocation_percentages.columns:
        s_scores = ou_params[(comb_id, 's_score')]
        prev_allocation = 0.0
        prev_s_score = np.nan
        for i, date in enumerate(s_scores.index):
            if i < ou_window:
                allocation = 0.0
            else:
                s_score = s_scores[date]
                sigma = ou_params.loc[date, (comb_id, 'sigma')] if tier == 6 else 1.0
                if pd.isna(s_score) or pd.isna(residuals_pivot.loc[date, comb_id]):
                    allocation = 0.0
                else:
                    is_decreasing = s_score < prev_s_score if not pd.isna(prev_s_score) else False
                    trend_tracker[comb_id] = is_decreasing
                    if prev_allocation > 0 and s_score > peak_s_scores[comb_id]:
                        peak_s_scores[comb_id] = s_score
                    if tier == 6:
                        allocation = allocation_func(s_score, prev_allocation, prev_s_score, is_decreasing, peak_s_scores[comb_id], sigma)
                    else:
                        allocation = allocation_func(s_score, prev_allocation, prev_s_score, is_decreasing, peak_s_scores[comb_id])
                    prev_s_score = s_score if not pd.isna(s_score) else prev_s_score
                    if allocation == 0.0:
                        peak_s_scores[comb_id] = 0.0  # Reset peak on exit
            allocation_percentages.loc[date, comb_id] = allocation
            prev_allocation = allocation
    logging.info(f"Computed allocations with tier={tier}")
    return allocation_percentages

def calculate_positions(allocation_percentages: pd.DataFrame, results_df: pd.DataFrame,
                        stock_price: pd.DataFrame, stocks: List[str]) -> pd.DataFrame:
    """Calculates trading positions based on allocation percentages.

    Args:
        allocation_percentages (pd.DataFrame): DataFrame with allocation percentages.
        results_df (pd.DataFrame): DataFrame with combination results (betas, residuals, etc.).
        stock_price (pd.DataFrame): DataFrame with stock and futures prices.
        stocks (List[str]): List of stock symbols.

    Returns:
        pd.DataFrame: DataFrame with positions for VN30F1M and each stock, including absolute values.
    """
    dates = results_df['Date'].sort_values().unique()
    columns = ['Total_Port_Trading', 'VN30F1M_Position'] + [f'{stock}_Position' for stock in stocks] + \
              ['Num_Active_Combinations', 'Active_Combination_IDs']
    positions_df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    positions_df['Active_Combination_IDs'] = positions_df['Active_Combination_IDs'].astype(object)
    positions_df = positions_df.fillna(0.0)

    for date in dates:
        if date not in allocation_percentages.index:
            continue
        active_combs = allocation_percentages.loc[date][allocation_percentages.loc[date] > 0]
        num_active = len(active_combs)
        active_ids = list(active_combs.index)
        positions_df.loc[date, 'Num_Active_Combinations'] = num_active
        positions_df.loc[date, 'Active_Combination_IDs'] = str(active_ids)

        if num_active == 0:
            total_allocation = 0.0
        else:
            base_allocation = min(0.4 + 0.1 * (num_active - 1), 0.98)
            intended_allocations = active_combs * base_allocation
            total_intended = intended_allocations.sum()
            scale_factor = 0.98 / total_intended if total_intended > 0.98 else 0.98
            scaled_allocations = intended_allocations * scale_factor
            total_allocation = scaled_allocations.sum()

        positions_df.loc[date, 'Total_Port_Trading'] = total_allocation
        positions_df.loc[date, 'VN30F1M_Position'] = -total_allocation * 0.20  # 20% short

        stock_allocation = total_allocation * 0.80  # 80% to stocks
        for comb_id in active_combs.index:
            comb_allocation = scaled_allocations[comb_id] * stock_allocation / total_allocation if total_allocation > 0 else 0
            comb_row = results_df[(results_df['Date'] == date) & (results_df['Combination_ID'] == comb_id)]
            if comb_row.empty or date not in stock_price.index:
                continue

            comb_stocks = [s for s in stocks if f'Beta_{s}' in comb_row.columns and comb_row[f'Beta_{s}'].values[0] >= 0]
            sum_beta_price = 0.0
            valid_comb_stocks = []
            for s in comb_stocks:
                beta = comb_row[f'Beta_{s}'].values[0]
                price = stock_price.loc[date, s] if s in stock_price.columns and not pd.isna(stock_price.loc[date, s]) else np.nan
                if not pd.isna(price):
                    sum_beta_price += beta * price
                    valid_comb_stocks.append(s)

            if sum_beta_price > 0:
                for stock in valid_comb_stocks:
                    beta = comb_row[f'Beta_{stock}'].values[0]
                    current_price = stock_price.loc[date, stock]
                    stock_proportion = (beta * current_price) / sum_beta_price
                    stock_position = comb_allocation * stock_proportion
                    positions_df.loc[date, f'{stock}_Position'] += stock_position

    logging.info(f"Calculated positions for {len(dates)} dates")
    return positions_df

def process_results_df(results_df: pd.DataFrame, stock_price: pd.DataFrame, stocks: List[str],
                       ou_window: int = 60, tier: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processes the results DataFrame to generate signals, allocations, positions, and trading logs.

    Args:
        results_df (pd.DataFrame): DataFrame with combination results.
        stock_price (pd.DataFrame): DataFrame with stock and futures prices.
        stocks (List[str]): List of stock symbols.
        ou_window (int, optional): Window size for OU process fitting. Defaults to 60.
        tier (int, optional): Allocation tier to use (1-7). Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (results_df, positions_df, trading_log).
    """
    logging.info(f"Processing results with ou_window={ou_window}, tier={tier}")
    results_df = results_df.sort_values('Date')
    residuals_pivot = results_df.pivot(index='Date', columns='Combination_ID', values='Residual')
    ou_params = generate_signals(residuals_pivot, ou_window=ou_window)
    allocation_percentages = compute_allocations(ou_params, residuals_pivot, ou_window=ou_window, tier=tier)
    positions_df = calculate_positions(allocation_percentages, results_df, stock_price, stocks)

    # Add s-scores and allocations to results_df
    results_df['s_score'] = results_df.apply(
        lambda row: ou_params.loc[row['Date'], (row['Combination_ID'], 's_score')]
        if row['Date'] in ou_params.index else np.nan, axis=1
    )
    results_df['Allocation'] = results_df.apply(
        lambda row: allocation_percentages.loc[row['Date'], row['Combination_ID']]
        if row['Date'] in allocation_percentages.index else 0.0, axis=1
    )

    # Add absolute values to positions_df
    for date in positions_df.index:
        if date in stock_price.index and not pd.isna(stock_price.loc[date, 'VN30F1M']):
            vn30_pos = positions_df.loc[date, 'VN30F1M_Position']
            positions_df.loc[date, 'Abs_VN30F1M'] = abs(vn30_pos) * stock_price.loc[date, 'VN30F1M']
            total_abs_stocks = 0.0
            for stock in stocks:
                stock_pos = positions_df.loc[date, f'{stock}_Position']
                if stock_pos > 0:
                    stock_price_val = stock_price.loc[date, stock]
                    positions_df.loc[date, f'Abs_{stock}'] = stock_pos * stock_price_val
                    total_abs_stocks += stock_pos * stock_price_val
                else:
                    positions_df.loc[date, f'Abs_{stock}'] = 0.0
            positions_df.loc[date, 'Abs_Stocks'] = total_abs_stocks
        else:
            positions_df.loc[date, 'Abs_VN30F1M'] = np.nan
            positions_df.loc[date, 'Abs_Stocks'] = np.nan
            for stock in stocks:
                positions_df.loc[date, f'Abs_{stock}'] = np.nan

    # Generate trading log
    trading_log = pd.DataFrame(index=positions_df.index)
    trading_log['Total_Port_Trading'] = positions_df['Total_Port_Trading']
    trading_log['Delta_VN30F1M'] = positions_df['VN30F1M_Position'].diff().fillna(0.0)
    trading_log['Action_VN30F1M'] = np.where(
        (trading_log['Delta_VN30F1M'] > 0) & (positions_df['VN30F1M_Position'].shift(1).fillna(0.0) < 0), 'buy to cover',
        np.where(trading_log['Delta_VN30F1M'] < 0, 'sell short', 'hold')
    )
    for stock in stocks:
        pos_col = f'{stock}_Position'
        delta_col = f'Delta_{stock}'
        action_col = f'Action_{stock}'
        trading_log[delta_col] = positions_df[pos_col].diff().fillna(0.0)
        trading_log[action_col] = np.where(
            trading_log[delta_col] > 0, 'buy',
            np.where(trading_log[delta_col] < 0, 'sell', 'hold')
        )
    trading_log['Num_Active_Combinations'] = positions_df['Num_Active_Combinations']
    trading_log['Active_Combination_IDs'] = positions_df['Active_Combination_IDs']

    # Round values
    for df in [positions_df, trading_log]:
        for col in df.columns:
            if col.startswith(('Total_Port_Trading', 'VN30F1M_Position', 'Delta_VN30F1M', 'Abs_VN30F1M', 'Abs_Stocks')) or \
               col.endswith('_Position') or col.startswith('Delta_') or col.startswith('Abs_'):
                df[col] = df[col].apply(lambda x: round(x, 4) if pd.notna(x) and abs(x) > 1e-10 else 0.0)

    # Sort indices
    results_df.sort_values(by=['Combination_ID', 'Date'], inplace=True)
    positions_df.sort_index(inplace=True)
    trading_log.sort_index(inplace=True)

    logging.info("Completed processing results")
    return results_df, positions_df, trading_log