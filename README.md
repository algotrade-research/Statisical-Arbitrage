# Statistical Arbitrage

## Abstract
Financial markets are inherently noisy, providing opportunities for algorithmic strategies to exploit pricing inefficiencies. This report develops a statistical arbitrage strategy inspired by Avellaneda and Lee, adapted for the Vietnamese stock market, where short selling of individual stocks is prohibited. I propose longing a basket of stocks and shorting the VN30F1M futures contract to capture mean-reverting relationships. I formed combinations by employing clustering techniques, the Johansen cointegration test, and generate signals by using the s-score of the Ornstein-Uhlenbeck process. After the backtesting, I found that the key things depends on how the trading signal is designed and additional efforts (time and computational power) should be implemented for the best performance of the model

## Introduction
### Hypothesis
In the Vietnamese stock market, dominated by retail investors, stocks exhibit exaggerated price movements due to overreactions to news or sentiment. These deviations from fundamental values result in wider spreads between stock baskets and the VN30F1M futures, which revert to their historical mean, enabling profitable trades via algorithmic mean-reversion strategies. 

## Related Work

## Data

## Installation

- **Requirement:** `pip`, `virtualenv`
- **Create and source new virtual environment in the current working directory with command**

```
python3 -m virtualenv venv
source venv/bin/activate
```
- **Install the dependencies by:**
```
pip install -r requirements.txt
```
## Implementation
- **To backtest and see the result**
```
python main.py
```
## In-sample Backtesting
## In-Sample Backtesting Results

#### Initial Parameters (08/2022–12/2023)

| **Metric**            | **Strategy (Initial)** | **VN30**  |
|-----------------------|------------------------|-----------|
| HPR                   | -18.38%               | -20.80%   |
| Excess HPR            | 2.42%                 | n/a       |
| Annual Return         | -8.36%                | -9.54%    |
| Annual Excess Return  | 1.18%                 | n/a       |
| Maximum Drawdown      | 26.99%                | 42.46%    |
| Longest Drawdown      | 477                   | 477       |
| Turnover Ratio        | 8.25%                 | n/a       |
| Sharpe Ratio          | -1.31                 | -0.65     |
| Sortino Ratio         | -0.86                 | -0.54     |
| Information Ratio     | 0.06                  | n/a       |



## Out-of-sample Backtesting
#### Optimal Parameters (01/2024–12/2024)

| **Metric**            | **Strategy (Optimal)** | **VN30**  |
|-----------------------|------------------------|-----------|
| HPR                   | -2.24%                | 18.83%    |
| Excess HPR            | -21.07%               | n/a       |
| Annual Return         | -2.25%                | 18.9%     |
| Annual Excess Return  | -21.15%               | n/a       |
| Maximum Drawdown      | 8.58%                 | 8.38%     |
| Longest Drawdown      | 123                   | 71        |
| Turnover Ratio        | 9.09%                 | n/a       |
| Sharpe Ratio          | -1.19                 | 0.97      |
| Sortino Ratio         | -0.32                 | 1.67      |
| Information Ratio     | -1.75                 | n/a       |
## Optimization
## Conclusion

## Reference

  
