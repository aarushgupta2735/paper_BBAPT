# BBAPT Paper Implementation

This repository contains the implementation of the "BBAPT" paper, which integrates Deep Reinforcement Learning (DRL) with advanced time-series forecasting (TimesNet) and behavioral finance concepts for cryptocurrency portfolio management. 

The project applies elements from Prospect Theory (such as risk aversion and overconfidence) to modify the actions taken by RL agents based on anticipated market conditions, aiming to create more robust and human-like trading strategies.

## Features

- **Deep Reinforcement Learning**: Train portfolio trading agents using advanced algorithms like A2C, PPO, and DDPG via `stable_baselines3`.
- **TimesNet Integration**: Incorporates `TimesNet` from `neuralforecast` for state-of-the-art time-series feature extraction and return forecasting.
- **Behavioral Mapping**: Dynamically adjusts agent actions using behavioral finance concepts (Overconfidence and Risk Aversion) based on TimesNet forecasts.
- **Custom Gym Environment**: Features a custom `PortfolioEnv` simulating realistic cryptocurrency trading with configurable transaction costs and short-selling rules.
- **Grid Search & Tuning**: Includes dedicated utilities for sweeping and tuning complex behavioral hyperparameters.
- **Traditional Baselines**: Compares RL approaches against standard portfolio optimization methods (Maximum Sharpe Ratio, Minimum Tail Risk, Equal Weighted, Minimum Variance) using `skfolio`.
- **Experiment Tracking**: Comprehensive integration with Weights & Biases (W&B) for tracking metrics, logging runs, and saving model checkpoints.

## Project Structure

```
BBAPT/
├── main.py                  # Main script to train the RL agent and run evaluation
├── baseline.py              # Script to run traditional portfolio optimization baselines
├── run_grid_search_main.py  # Script to perform grid search on behavioral hyperparameters
├── config/
│   └── config.py            # Centralized hyperparameter and configuration definitions
├── src/
│   ├── a2c.py               # Custom network architectures/policies for RL
│   ├── behavioral_map.py    # Logic for applying behavioral biases to actions
│   ├── data_prep.py         # Data loading, preprocessing, and TimesNet data preparation
│   ├── gridsearch.py        # Core logic for executing hyperparameter grid searches
│   └── portfolio_env.py     # Custom Gymnasium environment for portfolio trading
└── checkpoints/             # Directory where RL models are saved
```

## Setup & Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies (assuming you are setting up a virtual environment):

```bash
pip install pandas gymnasium stable_baselines3 neuralforecast skfolio wandb
```

### Weights & Biases (W&B) Setup

This project heavily relies on W&B for logging. Before running the scripts, ensure you are logged in to your W&B account:

```bash
wandb login
```

## Usage

### 1. Training the RL Model
To train the main RL model (A2C by default) combined with TimesNet forecasting and behavioral mapping, run:

```bash
python main.py
```
This script will:
- Load and prepare the cryptocurrency data.
- Initialize and prepare the `TimesNet` forecasting model.
- Train the selected RL algorithm.
- Run inference on the combined train and test sets, applying behavioral mapping to the agent's actions.
- Log performance metrics (Sharpe Ratio, Max Drawdown, Final Balance, etc.) to W&B.

### 2. Running Traditional Baselines
To compare the RL agent against traditional quantitative finance baselines, run:

```bash
python baseline.py
```
This script evaluates Maximum Sharpe Ratio, Minimum Tail Risk, Minimum Variance, and a Naive Equal-Weighted portfolio, logging the results to W&B.

### 3. Hyperparameter Grid Search
To tune the behavioral mapping parameters (Overconfidence and Risk Aversion bounds/coefficients), use the grid search script:

```bash
python run_grid_search_main.py
```
This will explore various configurations defined in the script, evaluate them on a validation set, log the sweep results to W&B, and save the optimal parameters to `best_hyperparameters.csv`.

## Configuration

All major hyperparameters, including starting capital, transaction costs, RL algorithm selection, and technical indicators, can be modified directly in `config/config.py`. By default, the environment is configured to trade a universe of 20 major cryptocurrencies.
