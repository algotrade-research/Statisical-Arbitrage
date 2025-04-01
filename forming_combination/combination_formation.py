import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
from .data_handler import DataHandler



class Combination_Formations:    
    """A class to implement a statistical arbitrage strategy using cointegration.

    This class identifies cointegrated combinations of futures and stocks, validates them,
    and tracks active combinations over time to generate trading signals.

    Args:
        data_handler: An object handling data access (futures, stocks, and historical data).
        min_trading_days (int, optional): Minimum trading days before re-evaluating a combination. Defaults to 45.
        threshold (float, optional): Minimum beta threshold for stock inclusion. Defaults to 0.05.
        max_stocks (int, optional): Maximum number of stocks in a combination. Defaults to 10.
        confidence_level (int, optional): Confidence level for Johansen cointegration test. Defaults to 1.
        adf_significance (float, optional): Significance level for ADF test. Defaults to 0.05.
        correlation_threshold (float, optional): Threshold for residual correlation to avoid duplicates. Defaults to 0.6.
        dynamic_threshold (bool, optional): Whether to dynamically adjust correlation threshold. Defaults to True.
        residual_threshold (float, optional): Threshold for residual size relative to futures price. Defaults to 0.3.
        improvement_threshold (float, optional): Minimum improvement in trace statistic for adding a stock. Defaults to 0.03.

    Attributes:
        active_combinations (list): List of currently active cointegrated combinations.
        combination_id (int): Unique identifier for combinations.
        results (list): List of results for each day and combination.
        validation_cache (dict): Cache for validation results to avoid redundant computations.
    """
    def __init__(self, data_handler, min_trading_days=45, threshold=0.05,
                 max_stocks=10, confidence_level=1, adf_significance=0.05,
                 correlation_threshold=0.6, dynamic_threshold=True,
                 residual_threshold=0.3, improvement_threshold=0.03,top_stocks=5):
        self.data_handler = data_handler
        self.futures = data_handler.futures
        self.stocks = data_handler.stocks
        self.estimation_window = data_handler.estimation_window
        self.data = data_handler.data
        self.min_trading_days = min_trading_days
        self.threshold = threshold
        self.max_stocks = max_stocks
        self.confidence_level = confidence_level
        self.confidence_level_joh_final = min(2, confidence_level + 1)
        self.adf_significance = adf_significance
        self.adf_significance_trading = min(0.1, 2 * adf_significance)
        self.correlation_threshold = correlation_threshold
        self.dynamic_threshold = dynamic_threshold
        self.residual_threshold = residual_threshold
        self.improvement_threshold = improvement_threshold
        self.active_combinations = []
        self.combination_id = 0
        self.results = []
        self.validation_cache = {}
        self.top_stocks = top_stocks  # Number of top stocks to consider for initial selection

    def get_pairwise_candidates(self, window_data, stocks_pool):
        """Identifies stocks that are cointegrated with the futures using pairwise Johansen tests.

        Args:
            window_data (pd.DataFrame): Historical data for the estimation window.
            stocks_pool (list): List of stock symbols to test for cointegration.

        Returns:
            list: Sorted list of stock symbols that are cointegrated with the futures, ranked by trace statistic.
        """
        candidates = []
        for stock in stocks_pool:
            try:
                result = coint_johansen(window_data[[self.futures, stock]], det_order=1, k_ar_diff=1)
                if result.lr1[0] > result.cvt[0, self.confidence_level]:
                    candidates.append((stock, result.lr1[0]))
            except Exception as e:
                print(f"Pairwise test failed for {stock}: {e}")
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [stock for stock, _ in candidates]

    def build_combination_greedy(self, window_data, candidates):
        """Greedily builds a cointegrated combination of stocks with the futures, trying multiple starting points.

        Args:
            window_data (pd.DataFrame): Historical data for the estimation window.
            candidates (list): List of candidate stock symbols.

        Returns:
            list: List of selected stock symbols forming a cointegrated combination.
        """
        if not candidates:
            return []
        best_selected = []
        best_trace_stat = -np.inf
        for start_stock in candidates[:self.top_stocks]:  # Try top 3 starting points
            selected = [start_stock]
            current_trace_stat = coint_johansen(window_data[[self.futures, start_stock]], det_order=1, k_ar_diff=1).lr1[0]
            for stock in [s for s in candidates if s != start_stock]:
                if len(selected) >= self.max_stocks:
                    break
                test_subset = selected + [stock]
                try:
                    result = coint_johansen(window_data[[self.futures] + test_subset], det_order=1, k_ar_diff=1)
                    if result.lr1[0] <= result.cvt[0, self.confidence_level]:
                        continue
                    improvement = (result.lr1[0] - current_trace_stat) / current_trace_stat
                    if improvement < self.improvement_threshold:    
                        continue
                    evec = result.evec[:, 0]
                    betas = -evec[1:] / evec[0]
                    if not all(beta >= 0 for beta in betas):
                        continue
                    selected.append(stock)
                    current_trace_stat = result.lr1[0]
                except Exception as e:
                    print(f"Combination test failed: {e}")
            if current_trace_stat > best_trace_stat:
                best_trace_stat = current_trace_stat
                best_selected = selected[:]
        return best_selected

    def validate_combination(self, window_data, selected):
        """Validates a combination by checking cointegration, beta positivity, stationarity, and residual size.

        Args:
            window_data (pd.DataFrame): Historical data for the estimation window.
            selected (list): List of selected stock symbols.

        Returns:
            tuple: (combination_params, adf_pvalue) where combination_params is a dict with intercept and betas,
                   or (None, np.inf) if validation fails.
        """
        comb_key = frozenset(selected)
        if comb_key in self.validation_cache:
            return self.validation_cache[comb_key]
        try:
            result = coint_johansen(window_data[[self.futures] + list(selected)], det_order=1, k_ar_diff=1)
            if result.lr1[0] <= result.cvt[0, self.confidence_level_joh_final]:
                self.validation_cache[comb_key] = (None, np.inf)
                return None, np.inf
            evec = result.evec[:, 0]
            betas = -evec[1:] / evec[0]
            if not all(beta >= 0 for beta in betas):
                self.validation_cache[comb_key] = (None, np.inf)
                return None, np.inf
            
            synthetic_portfolio = sum(window_data[s] * b for s, b in zip(selected, betas))
            residuals = window_data[self.futures] - synthetic_portfolio
            intercept = -residuals.mean()
            adf_pvalue = adfuller(residuals)[1]
            if adf_pvalue >= self.adf_significance:
                self.validation_cache[comb_key] = (None, adf_pvalue)
                return None, adf_pvalue
            futures_avg = window_data[self.futures].mean()
            if np.percentile(np.abs(residuals), 95) > self.residual_threshold * futures_avg:
                self.validation_cache[comb_key] = (None, adf_pvalue)
                return None, adf_pvalue
            selected_betas = {s: b for s, b in zip(selected, betas) if abs(b) > self.threshold}
            combination_params = {'intercept': intercept, 'betas': selected_betas}
            self.validation_cache[comb_key] = (combination_params, adf_pvalue)
            return combination_params, adf_pvalue
        except Exception as e:
            print(f"Validation failed for {selected}: {e}")
            self.validation_cache[comb_key] = (None, np.inf)
            return None, np.inf

    def is_similar(self, new_residuals, existing_residuals):
        """Checks if two sets of residuals are similar based on correlation.

        Args:
            new_residuals (pd.Series): Residuals of a new combination.
            existing_residuals (pd.Series): Residuals of an existing combination.

        Returns:
            bool: True if residuals are similar (correlation above threshold), False otherwise.
        """
        if len(new_residuals) != len(existing_residuals):
            return False
        corr, _ = pearsonr(new_residuals, existing_residuals)
        return corr > self.correlation_threshold

    def adjust_correlation_threshold(self):
        """Dynamically adjusts the correlation threshold based on the number of active combinations.

        If there are fewer than 10 active combinations, increases the threshold; otherwise, decreases it.
        """
        if self.dynamic_threshold:
            if len(self.active_combinations) < 10:
                self.correlation_threshold = min(0.8, self.correlation_threshold + 0.05)
            else:
                self.correlation_threshold = max(0.5, self.correlation_threshold - 0.05)

    def run_strategy(self):
        """Runs the statistical arbitrage strategy over the entire dataset.

        Identifies cointegrated combinations, validates them, and tracks residuals over time.
        Updates active combinations and logs results.
        """
        for day in range(self.estimation_window, len(self.data)):
            estimation_data = self.data.iloc[day - self.estimation_window:day]
            current_day = self.data.index[day]
            futures_current_price = self.data.iloc[day][self.futures]
            self.adjust_correlation_threshold()
            clusters = self.data_handler.cluster_stocks(estimation_data, current_day, futures_current_price)

            for cluster in clusters:
                candidates = self.get_pairwise_candidates(estimation_data, cluster)
                selected = self.build_combination_greedy(estimation_data, candidates)
                if selected:
                    params, new_adf_pvalue = self.validate_combination(estimation_data, selected)
                    if params:
                        self.add_combination_if_not_similar(params, new_adf_pvalue, estimation_data, current_day)

            top_candidates = []
            for cluster in clusters:
                cluster_candidates = self.get_pairwise_candidates(estimation_data, cluster)[:3]
                top_candidates.extend(cluster_candidates)
            top_candidates = list(set(top_candidates))

            if top_candidates:
                cross_selected = self.build_combination_greedy(estimation_data, top_candidates)
                if cross_selected:
                    cross_params, cross_adf_pvalue = self.validate_combination(estimation_data, cross_selected)
                    if cross_params:
                        self.add_combination_if_not_similar(cross_params, cross_adf_pvalue, estimation_data, current_day)

            all_candidates = self.get_pairwise_candidates(estimation_data, self.stocks)
            cross_selected = self.build_combination_greedy(estimation_data, all_candidates)
            if cross_selected:
                cross_params, cross_adf_pvalue = self.validate_combination(estimation_data, cross_selected)
                if cross_params:
                    self.add_combination_if_not_similar(cross_params, cross_adf_pvalue, estimation_data, current_day)

            for comb in self.active_combinations[:]:
                if day < comb['start_day']:
                    continue
                comb['trading_days'] += 1
                current_prices = self.data.iloc[day]
                synthetic_portfolio = sum(current_prices[s] * b for s, b in comb['params']['betas'].items())
                residual = current_prices[self.futures] - (comb['params']['intercept'] + synthetic_portfolio)
                comb['all_residuals'].append(residual)
                if comb['trading_days'] >= self.min_trading_days:
                    recent_residuals = pd.Series(comb['all_residuals'][-self.estimation_window:])
                    if adfuller(recent_residuals)[1] >= self.adf_significance_trading:
                        self.active_combinations.remove(comb)
                        continue
                row = {
                    'Date': current_day,
                    'Combination_ID': comb['id'],
                    'Residual': residual,
                    'Total_Combinations': len(self.active_combinations),
                    'Num_Stocks': len(comb['params']['betas']),
                    'Is_Estimation': False,
                    'Intercept': comb['params']['intercept'],
                    **{f'Beta_{s}': b for s, b in comb['params']['betas'].items()}
                }
                self.results.append(row)

    def add_combination_if_not_similar(self, params, new_adf_pvalue, estimation_data, current_day):
        """Adds a new combination if its residuals are not similar to existing ones.

        Args:
            params (dict): Parameters of the new combination (intercept and betas).
            new_adf_pvalue (float): ADF p-value of the new combination's residuals.
            estimation_data (pd.DataFrame): Historical data for the estimation window.
            current_day (pd.Timestamp): Current date in the backtest.
        """
        synthetic_portfolio = sum(estimation_data[s] * b for s, b in params['betas'].items())
        residuals = estimation_data[self.futures] - (params['intercept'] + synthetic_portfolio)
        similar_found = False
        to_remove = []
        for comb in self.active_combinations:
            existing_residuals = pd.Series(comb['all_residuals'][-self.estimation_window:])
            if self.is_similar(residuals, existing_residuals):
                if comb['trading_days'] >= self.min_trading_days:
                    existing_adf_pvalue = adfuller(existing_residuals)[1]
                    if new_adf_pvalue < 0.5 * existing_adf_pvalue:
                        to_remove.append(comb)
                else:
                    similar_found = True
        for comb in to_remove:
            self.active_combinations.remove(comb)
        if not similar_found:
            self.combination_id += 1
            self.active_combinations.append({
                'id': self.combination_id,
                'params': params,
                'start_day': self.data.index.get_loc(current_day),
                'all_residuals': residuals.tolist(),
                'trading_days': 0
            })
            for i, res in enumerate(residuals):
                row = {
                    'Date': estimation_data.index[i],
                    'Combination_ID': self.combination_id,
                    'Residual': res,
                    'Total_Combinations': len(self.active_combinations),
                    'Num_Stocks': len(params['betas']),
                    'Is_Estimation': True,
                    'Intercept': params['intercept'],
                    **{f'Beta_{s}': b for s, b in params['betas'].items()}
                }
                self.results.append(row)
            # print(f"\n=== New Combination {self.combination_id} at {current_day.date()} ===")
            # print(f"VN30F1M = {params['intercept']:.3f} + " + " + ".join([f"{b:.3f}*{s}" for s, b in params['betas'].items()]))

    def get_results(self):
        """Returns the results of the strategy as a DataFrame and the stock price data.

        Returns:
            tuple: (results_df, stock_price) where results_df is a DataFrame of results and stock_price is the price data.
        """
        results_df = pd.DataFrame(self.results)
        # Add all stocks to results_df with 0 betas for non-combination stocks
        for stock in self.stocks:
            beta_col = f'Beta_{stock}'
            if beta_col not in results_df.columns:
                results_df[beta_col] = 0.0
        results_df = results_df.sort_values(by=['Combination_ID', 'Date'])
        stock_price = self.data
        return results_df, stock_price

