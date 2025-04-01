# Statistical Arbitrage

## Abstract
Financial markets are inherently noisy, presenting opportunities for algorithmic strategies to exploit pricing inefficiencies for profit. This paper adapts the classical statistical arbitrage framework of Avellaneda and Lee [1] to the Vietnamese stock market, where short-selling stocks is prohibited. Instead of traditional pairs trading with two stocks, I propose a strategy that longs a portfolio of stocks and shorts the VN30F1M futures contract. I employ clustering and the Johansen cointegration test to identify cointegrated combinations, model residuals using the Ornstein-Uhlenbeck (OU) process, and generate trading signals based on s-scores to capture mean-reversion opportunities. My approach maintains the core principles of statistical arbitrage while addressing local market constraints.
## Installation

- **Requirement:** `pip`, `virtualenv`
- **Create and source new virtual environment in the current working directory with command**

```bash
python3 -m virtualenv venv
source venv/bin/activate
