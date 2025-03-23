import pandas as pd
import psycopg2
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm

# Database connection parameters
DB_PARAMS = {
    "host": "api.algotrade.vn",
    "port": 5432,
    "database": "algotradeDB",
    "user": "intern_read_only",
    "password": "ZmDaLzFf8pg5"
}

# Establish global connection (consider passing as parameter in production)
CONNECTION = psycopg2.connect(**DB_PARAMS)


def execute_query(query, from_date, to_date):
    """Execute a database query with date range parameters.

    Args:
        query (str): SQL query with %s placeholders for dates.
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        list: Query results as a list of tuples, or None if an error occurs.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (from_date, to_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        return result
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_stock_price(symbol, start_date, end_date):
    """Fetch adjusted close prices for a stock from the database.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'ACB').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or None if query fails.
    """
    query = """
    SELECT * FROM quote.adjclose
    WHERE tickersymbol = %s
    AND datetime BETWEEN %s AND %s
    ORDER BY datetime
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (symbol, start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_etf_price(symbol, start_date, end_date):
    """Fetch close prices for an ETF from the database.

    Args:
        symbol (str): ETF ticker symbol (e.g., 'FUEVFVND').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or None if query fails.
    """
    query = """
    SELECT * FROM quote.close
    WHERE tickersymbol = %s
    AND datetime BETWEEN %s AND %s
    ORDER BY datetime
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (symbol, start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_futures_price(start_date, end_date):
    """Fetch futures price data for VN30F1M from the database.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or empty DataFrame if query fails.
    """
    query = """
    SELECT c.datetime, c.tickersymbol, c.price
    FROM quote.close c
    JOIN quote.futurecontractcode fc 
        ON c.datetime = fc.datetime 
        AND fc.tickersymbol = c.tickersymbol
    WHERE fc.futurecode = 'VN30F1M'
        AND c.datetime BETWEEN %s AND %s
    ORDER BY c.datetime
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error executing query: {e}")
        CONNECTION.rollback()
        cursor.close()
        return pd.DataFrame(columns=["datetime", "tickersymbol", "price"])


def get_stock_data(symbols, start_date, end_date):
    """Fetch stock, ETF, and futures price data for multiple symbols.

    Args:
        symbols (list): List of ticker symbols (e.g., ['VN30F1M', 'ACB']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and symbol columns containing prices.
    """
    stock_data = pd.DataFrame()
    for symbol in symbols:
        if symbol == 'VN30F1M':
            close_price = get_futures_price(start_date, end_date)
        elif symbol in ['FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']:
            close_price = get_etf_price(symbol, start_date, end_date)
        else:
            close_price = get_stock_price(symbol, start_date, end_date)
        
        if close_price is not None:
            close_price = close_price[['datetime', 'price']]
            close_price.set_index('datetime', inplace=True)
            close_price.rename(columns={'price': symbol}, inplace=True)
            stock_data = pd.concat([stock_data, close_price], axis=1).dropna()
        time.sleep(0.5)  # Rate limiting
    
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data = stock_data.set_index('Date').sort_index()
    return stock_data


class DataHandler:
    """Handles stock data loading and clustering for statistical arbitrage.

    Args:
        futures (str): Futures ticker symbol (e.g., 'VN30F1M').
        stocks (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        estimation_window (int): Window size for residual computation (default: 60).
        cluster_update_interval (int): Days between cluster updates (default: 5).
        futures_change_threshold (float): Threshold for futures price change (default: 0.05).
        max_clusters (int): Maximum number of clusters (default: 10).

    Attributes:
        futures (str): Futures ticker symbol.
        stocks (list): List of stock ticker symbols.
        start_date (str): Start date.
        end_date (str): End date.
        estimation_window (int): Window size for residuals.
        cluster_update_interval (int): Days between cluster updates.
        futures_change_threshold (float): Threshold for futures price change.
        max_clusters (int): Maximum number of clusters.
        data (pd.DataFrame): Loaded price data.
        last_clusters (list): Last computed clusters.
        last_cluster_day (datetime): Date of last cluster update.
        last_futures_price (float): Last futures price.
    """

    def __init__(self, futures, stocks, start_date, end_date, 
                 estimation_window=60, cluster_update_interval=5, 
                 futures_change_threshold=0.05, max_clusters=10):
        self.futures = futures
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.estimation_window = estimation_window
        self.cluster_update_interval = cluster_update_interval
        self.futures_change_threshold = futures_change_threshold
        self.max_clusters = max_clusters
        self.data = self.load_data()
        self.last_clusters = None
        self.last_cluster_day = None
        self.last_futures_price = None

    def load_data(self):
        """Load price data for futures and stocks.

        Returns:
            pd.DataFrame: Cleaned price data with no missing values.
        """
        data = get_stock_data([self.futures] + self.stocks, self.start_date, self.end_date)
        return data.dropna()

    def compute_residuals(self, window_data):
        """Compute residuals from OLS regression of stocks against futures.

        Args:
            window_data (pd.DataFrame): Price data for the estimation window.

        Returns:
            pd.DataFrame: Residuals for each stock.
        """
        residuals = pd.DataFrame(index=window_data.index)
        for stock in self.stocks:
            if stock in window_data.columns:
                X = sm.add_constant(window_data[self.futures])
                y = window_data[stock]
                model = sm.OLS(y, X).fit()
                residuals[stock] = model.resid
        return residuals.dropna()

    def cluster_stocks(self, window_data, current_day, futures_current_price):
        """Cluster stocks based on residuals using KMeans.

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
            self.last_clusters = [self.stocks]
        else:
            X = residuals.T
            best_k = 2
            best_score = -1
            for k in range(2, min(self.max_clusters + 1, len(self.stocks))):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
                if kmeans.n_clusters > 1:
                    score = silhouette_score(X, kmeans.labels_)
                    if score > best_score:
                        best_score = score
                        best_k = k
            kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X)
            clusters = {i: [] for i in range(best_k)}
            for stock, label in zip(self.stocks, kmeans.labels_):
                clusters[label].append(stock)
            self.last_clusters = [cluster for cluster in clusters.values() if cluster]
        
        self.last_cluster_day = current_day
        self.last_futures_price = futures_current_price
        return self.last_clusters


def get_vn30(from_date, to_date):
    """Fetch VN30 data between specified dates from the database.

    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and 'Stock' column, or None if query fails.
    """
    query = """
    SELECT * FROM quote.vn30 v 
    WHERE v.datetime >= %s AND v.datetime <= %s
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (from_date, to_date))
        result = pd.DataFrame(cursor.fetchall())
        cursor.close()
        CONNECTION.commit()
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None
    
    result = result.rename(columns={result.columns[0]: 'Date', result.columns[1]: 'Stock'})
    result_df = result.set_index('Date')
    result_df.index = pd.to_datetime(result_df.index)
    return result_df
