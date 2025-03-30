import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
import os
from statsmodels.tsa.api import AutoReg



class OuProcess:
    """
    Generates trading signals by fitting an Ornstein-Uhlenbeck (OU) process to residuals.

    Args:
        residuals (pd.DataFrame): Pivot table of residuals with dates as index and combination IDs as columns.
        ou_window (int, optional): Window size for fitting the OU process. Defaults to 60.
        fallback_days (int, optional): Number of days to use previous OU parameters if fitting fails. Defaults to 5.

    Attributes:
        ou_params (pd.DataFrame): DataFrame storing OU parameters (kappa, m, sigma, s_score) for each combination.
        last_valid_params (dict): Stores the last valid OU parameters for each combination.
        ou_cache (dict): Cache for OU fitting results to avoid redundant computations.
    """
    def __init__(self, residuals: pd.DataFrame, ou_window: int = 60, fallback_days: int = 5):
        self.residuals = residuals
        self.ou_window = ou_window
        self.fallback_days = fallback_days
        self.ou_params = None
        self.last_valid_params = {col: None for col in residuals.columns}
        self.ou_cache = {}

    def fit_ou_process(self, series: pd.Series, date: pd.Timestamp) -> Dict[str, float]:
        """Fits an Ornstein-Uhlenbeck process to a series of residuals and computes the s-score.

        Args:
            series (pd.Series): Residual series for a combination.
            date (pd.Timestamp): Current date for caching purposes.

        Returns:
            dict: OU parameters {'kappa': float, 'm': float, 'sigma': float, 's_score': float}.
                  Returns NaN values if fitting fails.
        """
        cache_key = (series.name, date)
        if cache_key in self.ou_cache:
            return self.ou_cache[cache_key]
        if len(series) < self.ou_window:
            return {'kappa': np.nan, 'm': np.nan, 'sigma': np.nan, 's_score': np.nan}
        series_window = series[-self.ou_window:].dropna().to_numpy()
        if len(series_window) < self.ou_window:
            return {'kappa': np.nan, 'm': np.nan, 'sigma': np.nan, 's_score': np.nan}
        try:
            model = AutoReg(series_window, lags=1).fit()
            a, b = model.params
            p_value_b = model.pvalues[1]
            if p_value_b >= 0.10 or b <= 0 or b >= 1:
                return {'kappa': np.nan, 'm': np.nan, 'sigma': np.nan, 's_score': np.nan}
            kappa = -np.log(b) * np.sqrt(252)
            m = a / (1 - b)
            sigma = np.sqrt(model.sigma2 * 2 * kappa / (1 - b**2))
            latest = series.iloc[-1]
            sigma_eq = sigma / np.sqrt(2 * kappa) if kappa > 0 else np.inf
            s_score = (latest - m) / sigma_eq if sigma_eq != 0 else 0
            params = {'kappa': kappa, 'm': m, 'sigma': sigma, 's_score': s_score}
            self.ou_cache[cache_key] = params
            return params
        except (ValueError, np.linalg.LinAlgError):
            return {'kappa': np.nan, 'm': np.nan, 'sigma': np.nan, 's_score': np.nan}

    def apply_ou_fitting(self):
        """Applies OU process fitting to all residual series over time.

        Updates the ou_params DataFrame with kappa, m, sigma, and s_score for each combination and date.
        """
        columns = pd.MultiIndex.from_product([self.residuals.columns, ['kappa', 'm', 'sigma', 's_score']])
        self.ou_params = pd.DataFrame(index=self.residuals.index, columns=columns)
        for t in range(self.ou_window, len(self.residuals)):
            date = self.residuals.index[t]
            for stock in self.residuals.columns:
                series = self.residuals[stock].iloc[:t + 1]
                params = self.fit_ou_process(series, date)
                if not np.isnan(params['kappa']):
                    self.last_valid_params[stock] = {'params': params, 'date': date}
                elif self.last_valid_params[stock] and (date - self.last_valid_params[stock]['date']).days <= self.fallback_days:
                    last_params = self.last_valid_params[stock]['params']
                    latest = series.iloc[-1]
                    m, kappa, sigma = last_params['m'], last_params['kappa'], last_params['sigma']
                    sigma_eq = sigma / np.sqrt(2 * kappa) if kappa > 0 else np.inf
                    params['s_score'] = (latest - m) / sigma_eq if sigma_eq != 0 else 0
                for param, value in params.items():
                    self.ou_params.loc[date, (stock, param)] = value


