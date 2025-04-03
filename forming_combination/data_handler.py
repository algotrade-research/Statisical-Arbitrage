import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from data.get_data import *
import os

class DataHandler:
    """Handles stock data loading and clustering for statistical arbitrage.

    Args:
        futures (str): Futures ticker symbol (e.g., 'VN30F1M').
        stocks (list): List of stock ticker symbols (excluding ETFs).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        estimation_window (int, optional): Window size for residual computation. Defaults to 60.
        cluster_update_interval (int, optional): Days between cluster updates. Defaults to 3.
        futures_change_threshold (float, optional): Threshold for futures price change. Defaults to 0.05.
        max_clusters (int, optional): Maximum number of clusters. Defaults to 10.
        etf_list (list, optional): List of ETF ticker symbols. Defaults to None.
        etf_included (bool, optional): Whether to include ETFs in clustering. Defaults to True.

    Attributes:
        futures (str): Futures ticker symbol.
        stocks (list): List of stock ticker symbols (excluding ETFs).
        start_date (str): Start date.
        end_date (str): End date.
        estimation_window (int): Window size for residuals.
        cluster_update_interval (int): Days between cluster updates.
        futures_change_threshold (float): Threshold for futures price change.
        max_clusters (int): Maximum number of clusters.
        etf_list (list): List of ETF ticker symbols.
        etf_included (bool): Whether ETFs are included in clustering.
        data (pd.DataFrame): Loaded price data.
        last_clusters (list): Last computed clusters.
        last_cluster_day (datetime): Date of last cluster update.
        last_futures_price (float): Last futures price.
    """

    def __init__(self, futures, stocks, start_date, end_date, 
                 estimation_window=60, cluster_update_interval=3, 
                 futures_change_threshold=0.05, max_clusters=10,
                 etf_list=None, etf_included=True,use_existing_data=False):
        self.futures = futures
        # Ensure stocks list does not include ETFs
        self.etf_list = etf_list if etf_list is not None else []
        self.stocks = [s for s in stocks if s not in self.etf_list]  # Filter out ETFs from stocks
        self.start_date = start_date
        self.end_date = end_date
        self.estimation_window = estimation_window
        self.cluster_update_interval = cluster_update_interval
        self.futures_change_threshold = futures_change_threshold
        self.max_clusters = max_clusters
        self.etf_included = etf_included
        self.use_existing_data = use_existing_data
        self.data = self.load_data()
        self.last_clusters = None
        self.last_cluster_day = None
        self.last_futures_price = None
        

    def load_data(self):
        """Load price data for futures, stocks, and optionally ETFs.

        Args:
            use_existing (bool, optional): If True, load from existing CSV if available; 
                                          otherwise fetch new data. Defaults to True.

        Returns:
            pd.DataFrame: Cleaned price data with no missing values.
        """
        # Get the directory where this script/module is located (combination_formation)
        module_dir = os.path.dirname(os.path.abspath(__file__))

        # Move up one level to the project2 directory, then into optimization_data
        project_root = os.path.dirname(module_dir)  # Go up to project2
        folder = os.path.join(project_root, "data")
        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn’t exist
        csv_file = os.path.join(folder, f"stock_price_{self.start_date}_to_{self.end_date}.csv")

        # List of symbols to load
        symbols_to_load = [self.futures] + self.stocks
        if self.etf_included:
            symbols_to_load.extend(self.etf_list)

        if self.use_existing_data and os.path.exists(csv_file):
            # Load data from existing CSV
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            # Ensure all expected symbols are in the loaded data
            missing_symbols = set(symbols_to_load) - set(data.columns)
            if not missing_symbols:
                return data.dropna()
            else:
                print(f"Missing symbols {missing_symbols} in CSV; fetching new data.")

        # Fetch new data if CSV doesn’t exist or is incomplete
        data = get_stock_data(symbols_to_load, self.start_date, self.end_date)
        data_cleaned = data.dropna()
        
        # Save to CSV for future use
        data_cleaned.to_csv(csv_file)
        return data_cleaned

    def compute_residuals(self, window_data):
        """Compute residuals from OLS regression of stocks (and ETFs if included) against futures.

        Args:
            window_data (pd.DataFrame): Price data for the estimation window.

        Returns:
            pd.DataFrame: Residuals for each stock/ETF.
        """
        residuals = pd.DataFrame(index=window_data.index)
        # Compute residuals for stocks and ETFs (if included)
        symbols = self.stocks
        if self.etf_included:
            symbols = symbols + self.etf_list
        for symbol in symbols:
            if symbol in window_data.columns:
                X = sm.add_constant(window_data[self.futures])
                y = window_data[symbol]
                model = sm.OLS(y, X).fit()
                residuals[symbol] = model.resid
        return residuals.dropna()

    def cluster_stocks(self, window_data, current_day, futures_current_price):
        """Cluster stocks based on residuals using KMeans, with optional ETF handling.

        Args:
            window_data (pd.DataFrame): Price data for the estimation window.
            current_day (datetime): Current date for clustering.
            futures_current_price (float): Current futures price.

        Returns:
            list: List of stock clusters.
        """
        if (self.last_clusters is not None and self.last_cluster_day is not None):
            days_since_last_cluster = (current_day - self.last_cluster_day).days
            futures_change = (abs(futures_current_price - self.last_futures_price) / 
                              self.last_futures_price if self.last_futures_price else 0)
            if (days_since_last_cluster < self.cluster_update_interval and 
                futures_change < self.futures_change_threshold):
                return self.last_clusters
        
        residuals = self.compute_residuals(window_data)
        if residuals.empty or len(residuals.columns) < 2:
            # If residuals are empty or too few symbols, return all symbols as one cluster
            symbols = self.stocks
            if self.etf_included:
                symbols = symbols + self.etf_list
            self.last_clusters = [symbols]
        else:
            # Separate ETFs and stocks based on etf_included flag
            if self.etf_included:
                # Include ETFs, but cluster them separately
                etf_list = self.etf_list
                stock_list = self.stocks  # Already filtered in __init__
            else:
                # Exclude ETFs entirely
                etf_list = []
                stock_list = self.stocks
            
            clusters = []
            # Cluster non-ETFs
            if stock_list:
                X = residuals[stock_list].T
                best_k = min(2, len(stock_list))  # Ensure at least 2 if possible
                best_score = -1
                for k in range(2, min(self.max_clusters + 1, len(stock_list))):
                    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
                    if kmeans.n_clusters > 1:
                        score = silhouette_score(X, kmeans.labels_)
                        if score > best_score:
                            best_score = score
                            best_k = k
                kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X)
                stock_clusters = {i: [] for i in range(best_k)}
                for stock, label in zip(stock_list, kmeans.labels_):
                    stock_clusters[label].append(stock)
                clusters.extend([c for c in stock_clusters.values() if c])
            # Add ETFs as a separate cluster if included
            if self.etf_included and etf_list:
                clusters.append(etf_list)
            
            self.last_clusters = clusters
        
        self.last_cluster_day = current_day
        self.last_futures_price = futures_current_price
        return self.last_clusters
