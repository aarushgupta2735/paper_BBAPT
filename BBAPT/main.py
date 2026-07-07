import pandas as pd
import yfinance as yf
import torch
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss 
from neuralforecast.models import TimesNet  
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from stable_baselines3 import A2C, PPO  
import gymnasium as gym
from gymnasium.utils.env_checker import check_env


from BBAPT.config.config import appConfig
from BBAPT.src.data_prep import data_for_timesnet, data_prep    
from BBAPT.src.portfolio_env import PortfolioEnv

#setup config
config = appConfig(
    initial_balance = 100000,
    allow_short_selling = False,
    technical_indicator_list = ['MACD','RSI_14','CCI_20','SMA_20','EMA_20'],    
    transaction_cost = 0.001,
    ticker_list = ["BTC-USD", "ETH-USD", "LTC-USD", "LINK-USD", "BCH-USD", "UNI-USD", "XLM-USD", "FIL-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "SHIB-USD", "TON-USD", "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "ETC-USD"],
    train_starting_date = "2020-09-22",
    train_ending_date = "2022-06-07",
    test_starting_date = "2022-06-08",
    test_ending_date = "2023-01-03",
    
    #PARAMETERS FOR BEHAVIOURAL MAPPING 
    THRESHOLD_PARAMETER = 0.015,    
    #overconfidence     
    oc_upper_threshold = 0.01,
    oc_lower_threshold = -0.01,
    oc_k_loss = 0.1,
    oc_k_gain = 0.1,
    oc_n = 0.725,                 
    #risk averse
    ra_upper_threshold = 0.01,            
    ra_lower_threshold = -0.01,  
    ra_k_loss = 0.1,
    ra_k_gain = 0.1,
    ra_n = 1.22,
) 

#data_preparation
train_df = data_prep(config.train_starting_date, config.train_ending_date, config.ticker_list)
test_df = data_prep(config.test_starting_date, config.test_ending_date, config.ticker_list)


#---TimesNet prediction----
model = TimesNet(h=1,
                input_size=24, #Context Window
                hidden_size = 16,
                conv_hidden_size = 32,
                loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                scaler_type='standard',
                learning_rate=1e-3,
                max_steps=100,
                val_check_steps=50,
                early_stop_patience_steps=2)
nf = NeuralForecast(
    models=[model],
    freq='B'
)


list_forecasts = []
for ticker in config.ticker_list:
    nf.fit(df=data_for_timesnet(train_df.loc[train_df['unique_id'] == ticker]),val_size=1)
    forecasts = nf.predict()
    list_forecasts.append(forecasts)

forecast = pd.concat(list_forecasts,ignore_index=True)
forecast['avg_return'] = forecast.groupby('ds')['TimesNet'].transform('mean')
#--------------------------

#Portfolio Environment
gym.register(
    id='PortfolioEnv-v1',
    entry_point=PortfolioEnv,
    kwargs={'data': train_df, 'config': config}
)
env = gym.make('PortfolioEnv-v1') 
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

# 1. Training Loop
print("--- Training RL Model ---")
model_rl = A2C('MlpPolicy', env, verbose=1)
model_rl.learn(total_timesteps=10000)
print("Training completed!")

# 2. Inference Loop
print("--- Running Inference ---")
gym.register(
    id='PortfolioEnv-test-v1',
    entry_point=PortfolioEnv,
    kwargs={'data': test_df, 'config': config}
)
test_env = gym.make('PortfolioEnv-test-v1')

obs, info = test_env.reset()
done = False
while not done:
    action, _states = model_rl.predict(obs)
    obs, reward, done, truncated, info = test_env.step(action)

print(f"Inference completed! Final portfolio balance: {info['balance']:.2f}")