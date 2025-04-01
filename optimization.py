import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from utils.helper import plot_asset_balance, generate_periods_df,run_backtest_for_periods
from utils.calculate_metrics import calculate_shapre_and_mdd
from data.get_data import *
from optuna import *
from optuna.pruners import MedianPruner
# Set up logging to see Optuna's internal messages
optuna.logging.set_verbosity(optuna.logging.INFO)
pd.set_option('future.no_silent_downcasting', True)
import warnings

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory (project2)
os.chdir(script_dir)

# Suppress the specific Optuna warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna.distributions"
)
# Set random seeds for reproducibility
SEED = 42  # You can choose any integer
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)  # For hash-based operations
vn30_stocks = pd.read_csv('optimization_data/vn30_stocks.csv', index_col=0, parse_dates=True)
# Load data once at the start
etf_list = ['FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']
start_date = '2021-06-01'
end_date = '2025-01-01'
periods_df = generate_periods_df(
        vn30_stocks,
        start_date,
        end_date,
        window=80,
    )
def objective(trial):
    # Suggest parameters to tune
    estimation_window = trial.suggest_int("estimation_window", 40, 80, step=10)
    min_trading_days_fraction = trial.suggest_float(
        "min_trading_days", 1 / 3, 2 / 3, step=1 / 6
    )
    min_trading_days = int(min_trading_days_fraction * estimation_window)  # Convert fraction to days
    max_clusters = 10
    top_stocks = trial.suggest_int("top_stocks", 3, 9, step=3)
    correlation_threshold = trial.suggest_float(
        "correlation_threshold", 0.3, 0.6, step=0.3
    )
    residual_threshold = 0.3
    ou_window = estimation_window  
    tier = trial.suggest_categorical("tier", [1, 2,3,4])

    first_allocation = trial.suggest_float("first_allocation", 0.3, 0.6, step=0.1)
    adding_allocation = trial.suggest_float("adding_allocation", 0.1, 0.3, step=0.1)
    # Run backtest with suggested parameters
    combined_returns_df, _, _ = run_backtest_for_periods(
        periods_df=periods_df,
        futures="VN30F1M",
        etf_list=etf_list,
        etf_included=False,
        estimation_window=estimation_window,
        min_trading_days=min_trading_days_fraction * estimation_window,  # Convert fraction to days
        max_clusters=max_clusters,
        top_stocks=top_stocks,
        correlation_threshold=correlation_threshold,
        residual_threshold=residual_threshold,
        tier=tier,
        first_allocation=first_allocation,
        adding_allocation=adding_allocation,
        use_existing_data=True
    )

    # Split into train and test sets
    train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
    test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]

    # Calculate metrics for train set (optimization targets)
    sharpe_train,max_drawdown_train = calculate_shapre_and_mdd(train_set, risk_free_rate=0.05)

    # Calculate metrics for test set (for reporting, not optimization)
    sharpe_test, max_drawdown_test = calculate_shapre_and_mdd(test_set, risk_free_rate=0.05)

    # Store test set metrics for later analysis
    trial.set_user_attr("sharpe_test", sharpe_test)
    trial.set_user_attr("max_drawdown_test", max_drawdown_test)

    # Return tuple for multi-objective optimization (maximize Sharpe, minimize drawdown)
    return sharpe_train, max_drawdown_train

# Create an Optuna study with multi-objective optimization
study = optuna.create_study(
    directions=["maximize", "minimize"],  # Maximize Sharpe, minimize drawdown
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="vn30_arbitrage_tuning",
    load_if_exists=True,
    pruner=MedianPruner()
)

# Optimize with a moderate number of trials
study.optimize(objective, n_trials=50)

# Print the best results
best_trial = study.best_trials[0]  # Get the best trial (Pareto optimal)
best_params = best_trial.params
best_sharpe_train = best_trial.values[0]
best_drawdown_train = best_trial.values[1]
best_sharpe_test = best_trial.user_attrs["sharpe_test"]
best_drawdown_test = best_trial.user_attrs["max_drawdown_test"]

print(f"Best Train Sharpe Ratio: {best_sharpe_train}")
print(f"Best Train Max Drawdown: {best_drawdown_train}")
print(f"Test Sharpe Ratio: {best_sharpe_test}")
print(f"Test Max Drawdown: {best_drawdown_test}")
print(f"Best Parameters: {best_params}")
# Save all trials to CSV with custom column names matching variable names
trials_df = study.trials_dataframe()

# Rename columns to match your variable names
column_mapping = {
    'values_0': 'best_sharpe_train',
    'values_1': 'best_drawdown_train',
    'user_attrs_sharpe_test': 'best_sharpe_test',
    'user_attrs_max_drawdown_test': 'best_drawdown_test',
}

# Rename the columns in the DataFrame
trials_df.rename(columns=column_mapping, inplace=True)

# Keep only relevant columns (optional, remove if you want all columns)
relevant_columns = ['number', 'best_sharpe_train', 'best_drawdown_train', 
                   'best_sharpe_test', 'best_drawdown_test'] + [f'params_{p}' for p in best_params.keys()]
trials_df = trials_df[relevant_columns]

# Save to CSV
trials_df.to_csv("optuna_trials.csv", index=False)
# Visualize parameter importances (for Sharpe only, as it's multi-objective)
optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0]).show()

# Extract Sharpe Ratios and trial numbers
sharpe_ratios = [trial.values[0] for trial in study.trials if trial.values is not None]
trial_numbers = list(range(len(sharpe_ratios)))

# Compute the best Sharpe Ratio seen so far for each trial
best_sharpe_so_far = []
current_best = float("-inf")  # Since we are maximizing Sharpe Ratio
for sharpe in sharpe_ratios:
    current_best = max(current_best, sharpe)
    best_sharpe_so_far.append(current_best)

# Create the plot
plt.figure(figsize=(10, 6))
# Scatter plot of Sharpe Ratios for each trial
plt.scatter(trial_numbers, sharpe_ratios, color="blue", alpha=0.5, label="Sharpe Ratio per Trial")
# Plot the best Sharpe Ratio seen so far as a red line
plt.plot(trial_numbers, best_sharpe_so_far, color="red", label="Best Sharpe Ratio")
plt.xlabel("Trial")
plt.ylabel("Sharpe Ratio (Objective)")
plt.title("Sharpe Ratio Over Trials")
plt.legend()
plt.grid(True)
plt.show()