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



class PortfolioManager:
    """Manages a portfolio for backtesting a trading strategy with futures and stocks.

    Handles position adjustments and profit calculations for VN30F1M futures and stocks, treating futures like stocks
    with unrealized profits contributing to portfolio value.

    Attributes:
        initial_balance (float): Initial cash balance.
        vn30_fee_per_point (float): Fee per point for VN30F1M futures.
        stock_fee_rate (float): Fee rate for stock transactions.
        contract_size (int): Contract size for futures and stock lots.
        min_cash_fraction (float): Minimum cash fraction of initial balance.
        estimation_window (int): Number of days to ignore for pair searching.
    """

    def __init__(self, initial_balance: float = 1000000000, vn30_fee_per_point: float = 0.23,
                 stock_fee_rate: float = 0.0023, contract_size: int = 100,
                 min_cash_fraction: float = 0.02, estimation_window: int = 60):
        """Initializes the PortfolioManager with given parameters.

        Args:
            initial_balance (float, optional): Initial cash balance. Defaults to 1,000,000,000.
            vn30_fee_per_point (float, optional): Fee per point for VN30F1M. Defaults to 0.23.
            stock_fee_rate (float, optional): Fee rate for stocks. Defaults to 0.0023.
            contract_size (int, optional): Contract size for trades. Defaults to 100.
            min_cash_fraction (float, optional): Minimum cash fraction. Defaults to 0.02.
            estimation_window (int, optional): Days to skip for reporting. Defaults to 60.
        """
        self.initial_balance = initial_balance
        self.vn30_fee_per_point = vn30_fee_per_point
        self.stock_fee_rate = stock_fee_rate
        self.contract_size = contract_size
        self.min_cash_fraction = min_cash_fraction
        self.estimation_window = estimation_window
        self.reset()

    def reset(self):
        """Resets the portfolio to its initial state."""
        self.cash = self.initial_balance
        self.positions = {'VN30F1M': 0}  # Contracts for VN30F1M (positive = long, negative = short)
        self.stock_positions = {}  # Shares for stocks
        self.entry_prices = {'VN30F1M': 0}  # Average entry price for VN30F1M
        self.stock_entry_prices = {}  # Average entry prices for stocks
        self.prev_vn30_price = None
        self.prev_stock_prices = {}
        self.prev_balance = self.initial_balance
        self.cumulative_profit_stocks = 0.0
        self.cumulative_profit_futures = 0.0
        self.asset_df = []
        self.detail_df = []
        self.trading_log = []

    def _calculate_fee(self, is_futures: bool, quantity: int, price: float) -> float:
        """Calculates transaction fees for futures or stocks.

        Args:
            is_futures (bool): True if the asset is VN30F1M futures, False if stock.
            quantity (int): Number of contracts or shares traded.
            price (float): Price per unit (contract or share).

        Returns:
            float: Transaction fee.
        """
        if is_futures:
            return abs(quantity) * self.vn30_fee_per_point * self.contract_size
        return abs(quantity) * price * self.stock_fee_rate

    def get_portfolio_value(self, vn30f1m_price: float, stock_prices: Dict[str, float]) -> float:
        """Calculates the total portfolio value including unrealized profits.

        Args:
            vn30f1m_price (float): Current price of VN30F1M futures.
            stock_prices (Dict[str, float]): Current prices of stocks.

        Returns:
            float: Total portfolio value (cash + stocks value + futures value).
        """
        stocks_value = sum(self.stock_positions.get(stock, 0) * stock_prices.get(stock, 0)
                           for stock in self.stock_positions if not np.isnan(stock_prices.get(stock, 0)))
        futures_value = (vn30f1m_price - self.entry_prices['VN30F1M']) * self.positions['VN30F1M'] * self.contract_size \
                        if self.positions['VN30F1M'] != 0 and not np.isnan(vn30f1m_price) else 0
        return self.cash + stocks_value + futures_value

    def _adjust_position(self, asset: str, price: float, target_units: int, is_futures: bool) -> Tuple[float, float, int]:
        """Adjusts the position of an asset (VN30F1M or stock) to the target number of units.

        Args:
            asset (str): Asset symbol ('VN30F1M' or stock ticker).
            price (float): Current price of the asset.
            target_units (int): Target number of contracts (VN30F1M) or shares (stocks).
            is_futures (bool): True if adjusting VN30F1M, False if stock.

        Returns:
            Tuple[float, float, int]: (fee, realized_pnl, units_traded).
        """
        current_units = self.positions[asset] if is_futures else self.stock_positions.get(asset, 0)
        delta_units = target_units - current_units
        if delta_units == 0:
            return 0, 0, 0
        fee = self._calculate_fee(is_futures, delta_units, price)
        min_cash = self.min_cash_fraction * self.initial_balance
        entry_dict = self.entry_prices if is_futures else self.stock_entry_prices

        if delta_units > 0:  # Buy
            cost = delta_units * price
            total_cost = cost + fee
            if self.cash - total_cost >= min_cash:
                self.cash -= total_cost
                if asset not in entry_dict:
                    entry_dict[asset] = price
                    if is_futures:
                        self.positions[asset] = delta_units
                    else:
                        self.stock_positions[asset] = delta_units
                else:
                    total_units = current_units + delta_units
                    avg_entry = (entry_dict[asset] * current_units + price * delta_units) / total_units
                    entry_dict[asset] = avg_entry
                    if is_futures:
                        self.positions[asset] = total_units
                    else:
                        self.stock_positions[asset] = total_units
                return fee, 0, delta_units
        elif delta_units < 0:  # Sell
            units_to_sell = min(abs(delta_units), abs(current_units))
            if units_to_sell > 0:
                proceeds = units_to_sell * price
                realized_pnl = (price - entry_dict[asset]) * units_to_sell * (self.contract_size if is_futures else 1)
                self.cash += proceeds - fee
                if is_futures:
                    self.positions[asset] -= units_to_sell
                else:
                    self.stock_positions[asset] -= units_to_sell
                if (is_futures and self.positions[asset] == 0) or (not is_futures and self.stock_positions[asset] == 0):
                    del entry_dict[asset]
                return fee, realized_pnl, -units_to_sell
        return 0, 0, 0

    def _calculate_daily_profits(self, prev_vn30_price: float, vn30_price: float,
                                 prev_stock_prices: Dict[str, float], stock_prices: Dict[str, float]) -> Tuple[float, float]:
        """Calculates daily unrealized profits for reporting.

        Args:
            prev_vn30_price (float): Previous day's VN30F1M price.
            vn30_price (float): Current day's VN30F1M price.
            prev_stock_prices (Dict[str, float]): Previous day's stock prices.
            stock_prices (Dict[str, float]): Current day's stock prices.

        Returns:
            Tuple[float, float]: (profit_stocks_today, profit_futures_today).
        """
        profit_futures = ((vn30_price - prev_vn30_price) * self.positions['VN30F1M'] * self.contract_size
                          if prev_vn30_price is not None and not np.isnan(vn30_price) else 0)
        profit_stocks = sum((stock_prices[stock] - prev_stock_prices[stock]) * self.stock_positions[stock]
                            for stock in self.stock_positions
                            if prev_stock_prices.get(stock) is not None and not np.isnan(stock_prices.get(stock, 0)))
        return profit_stocks, profit_futures

    def _record_state(self, date: pd.Timestamp, vn30_price: float, stock_prices: Dict[str, float], daily_fee: float,
                      profit_stocks_today: float, profit_futures_today: float):
        """Records the portfolio state for reporting.

        Args:
            date (pd.Timestamp): Current date.
            vn30_price (float): Current VN30F1M price.
            stock_prices (Dict[str, float]): Current stock prices.
            daily_fee (float): Total fees for the day.
            profit_stocks_today (float): Daily unrealized profit from stocks.
            profit_futures_today (float): Daily unrealized profit from futures.
        """
        portfolio_value = self.get_portfolio_value(vn30_price, stock_prices)
        vn30_notional = abs(self.positions['VN30F1M']) * vn30_price * self.contract_size if not np.isnan(vn30_price) else 0
        abs_stocks = sum(self.stock_positions.get(stock, 0) * stock_prices.get(stock, 0)
                         for stock in stock_prices if not np.isnan(stock_prices.get(stock, 0)))
        positions = {'VN30F1M': self.positions['VN30F1M'], **self.stock_positions}

        self.asset_df.append({
            'Date': date, 'balance': portfolio_value, 'vn30f1m_notional': vn30_notional,
            'abs_stocks': abs_stocks, 'cash': self.cash, 'positions': positions.copy()
        })

        vn30_pct_change = ((vn30_price - self.prev_vn30_price) / self.prev_vn30_price
                           if self.prev_vn30_price is not None and self.prev_vn30_price != 0 and not np.isnan(vn30_price) else 0)
        stock_pct_change = (sum(((stock_prices[stock] - self.prev_stock_prices[stock]) / self.prev_stock_prices[stock]) *
                                (self.stock_positions.get(stock, 0) * self.prev_stock_prices[stock] / abs_stocks)
                                for stock in self.stock_positions
                                if self.prev_stock_prices.get(stock) is not None and self.prev_stock_prices[stock] != 0
                                and not np.isnan(stock_prices.get(stock, 0)) and abs_stocks > 0) if abs_stocks > 0 else 0)

        weights = {
            'vn30f1m_weight': (self.positions['VN30F1M'] * vn30_price * self.contract_size / portfolio_value)
                              if portfolio_value > 0 and not np.isnan(vn30_price) else 0,
            **{f'{stock}_weight': (self.stock_positions.get(stock, 0) * stock_prices.get(stock, 0) / portfolio_value)
               if portfolio_value > 0 and not np.isnan(stock_prices.get(stock, 0)) else 0 for stock in stock_prices}
        }

        self.detail_df.append({
            'Date': date, 'balance': portfolio_value, 'vn30f1m_notional': vn30_notional,
            'abs_stocks': abs_stocks, 'cash': self.cash, 'fee': daily_fee,
            'vn30f1m_pct_change': vn30_pct_change, 'stock_pct_change': stock_pct_change,
            'profit_stocks_today': profit_stocks_today, 'profit_futures_today': profit_futures_today,
            'cumulative_profit_stocks': self.cumulative_profit_stocks,
            'cumulative_profit_futures': self.cumulative_profit_futures,
            **weights
        })

    def run_backtest(self, position_df: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs a backtest of the portfolio using position and price data.

        Args:
            position_df (pd.DataFrame): DataFrame with target positions.
            prices (pd.DataFrame): DataFrame with asset prices.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (asset_df, detail_df, trading_log).
        """
        position_df.index = pd.to_datetime(position_df.index)
        prices.index = pd.to_datetime(prices.index)
        common_dates = position_df.index.intersection(prices.index)
        position_df = position_df.loc[common_dates]
        prices = prices.loc[common_dates]
        stocks = [col.replace('_Position', '') for col in position_df.columns if col.endswith('_Position') and col != 'VN30F1M_Position']

        for day, date in enumerate(common_dates):
            vn30_price = prices.loc[date, 'VN30F1M'] if 'VN30F1M' in prices.columns else np.nan
            stock_prices = {stock: prices.loc[date, stock] if stock in prices.columns else np.nan for stock in stocks}

            # Calculate daily profits
            profit_stocks_today, profit_futures_today = self._calculate_daily_profits(
                self.prev_vn30_price, vn30_price, self.prev_stock_prices, stock_prices
            )
            self.cumulative_profit_stocks += profit_stocks_today
            self.cumulative_profit_futures += profit_futures_today

            # Portfolio value before trading
            portfolio_value = self.get_portfolio_value(vn30_price, stock_prices)

            # Adjust futures position
            daily_fee = 0
            contracts_traded_vn30 = 0
            if not np.isnan(vn30_price) and vn30_price > 0:
                target_proportion = position_df.loc[date, 'VN30F1M_Position']
                target_contracts = int((portfolio_value * target_proportion / vn30_price) / self.contract_size) * self.contract_size
                fee_futures, _, contracts_traded_vn30 = self._adjust_position('VN30F1M', vn30_price, target_contracts, True)
                daily_fee += fee_futures

            # Adjust stock positions
            shares_traded = {}
            for stock in stocks:
                price = stock_prices.get(stock)
                if price is not None and not np.isnan(price) and price > 0:
                    target_proportion = position_df.loc[date, f'{stock}_Position']
                    target_shares = int((portfolio_value * target_proportion / price) / self.contract_size) * self.contract_size
                    fee_stock, _, traded = self._adjust_position(stock, price, target_shares, False)
                    daily_fee += fee_stock
                    shares_traded[stock] = traded
                else:
                    shares_traded[stock] = 0

            # Record state after estimation window
            if day >= self.estimation_window:
                self._record_state(date, vn30_price, stock_prices, daily_fee, profit_stocks_today, profit_futures_today)

            # Update previous values
            self.prev_vn30_price = vn30_price
            self.prev_stock_prices = stock_prices.copy()
            self.prev_balance = portfolio_value
            self.trading_log.append({'Date': date, 'VN30F1M_Traded': contracts_traded_vn30,
                                     **{f'{stock}_Traded': shares_traded[stock] for stock in stocks}})

        return (pd.DataFrame(self.asset_df).set_index('Date'),
                pd.DataFrame(self.detail_df).set_index('Date'),
                pd.DataFrame(self.trading_log).set_index('Date'))