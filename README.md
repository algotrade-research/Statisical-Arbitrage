# Statistical Arbitrage

## Abstract
Financial markets are inherently noisy, providing opportunities for algorithmic strategies to exploit pricing inefficiencies. This report develops a statistical arbitrage strategy inspired by Avellaneda and Lee, adapted for the Vietnamese stock market, where short selling of individual stocks is prohibited. I propose longing a basket of stocks and shorting the VN30F1M futures contract to capture mean-reverting relationships. I formed combinations by employing clustering techniques, the Johansen cointegration test, and generate signals by using the s-score of the Ornstein-Uhlenbeck process. After the backtesting, I found that the key things depends on how the trading signal is designed and additional efforts (time and computational power) should be implemented for the best performance of the model

## Introduction
### Hypothesis
In the Vietnamese stock market, dominated by retail investors, stocks exhibit exaggerated price movements due to overreactions to news or sentiment. These deviations from fundamental values result in wider spreads between stock baskets and the VN30F1M futures, which revert to their historical mean, enabling profitable trades via algorithmic mean-reversion strategies. 
### Key Idea
Statistical arbitrage is a popular algorithmic trading strategy that delivers market-neutral returns, independent of market trends. It attracts investors with its diversification benefits and high-reward, low-risk potential—similar to earning high interest from a bank but with greater upside.

This strategy uses statistical methods to exploit pricing inefficiencies, often via mean-reverting portfolios. A classic example, pairs trading, involves trading two correlated securities (long one, short the other) when their price spread diverges, expecting it to revert. The model is:

$$ \frac{dP_t}{P_t} = \alpha \, dt + \beta \frac{dQ_t}{Q_t} + dX_t $$

Here, $P_t$ and $Q_t$ are stock prices, $\alpha$ is a drift (often small), $\beta$ is the hedge ratio, and $X_t$ is a mean-reverting residual guiding trades.

In Vietnam, short-selling stocks is banned, making pairs trading unfeasible. Instead, this strategy longs a basket of stocks and shorts the VN30F1M futures contract to stay market-neutral. The goal is to find:

$$ \text{VN30F1M} = \text{intercept} + \sum_{i} \beta_i \cdot \text{stock}_i + \text{residual} $$

where the residual is stationary. 
## Related Work

Statistical arbitrage is well-documented in finance. Avellaneda and Lee (2010) modeled pairs trading with cointegration and the Ornstein-Uhlenbeck process, where the spread \( X_t \) follows:

$$ dX_t = \kappa (\mu - X_t) \, dt + \sigma \, dW_t $$

Here, \( \kappa \) is the reversion speed, \( \mu \) is the mean, and \( \sigma \) is volatility, with trades based on the s-score. O-U parameters are estimated via an AR(1) model:

$$ X_{n+1} = a + b X_n + \theta_{n+1} $$

Stanford students (Lu, Parulekar, Xu, 2018) proposed clustering and the Johansen Test for cointegration—unlike the Engel-Granger test, it handles multiple cointegration relationships—enhancing stock-future cointegration analysis.
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

  
