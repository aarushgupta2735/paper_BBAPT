from BBAPT.src.config import appConfig
from BBAPT.src.gridsearch import run_grid_search

def findparams():
    config = appConfig(
        initial_balance=100000,
        allow_short_selling=False,
        technical_indicator_list=["MACD", "RSI_14", "CCI_20", "SMA_20", "EMA_20"],
        transaction_cost=0.001,
        ticker_list=[
            "BTC-USD",
            "ETH-USD",
            "LTC-USD",
            "LINK-USD",
            "BCH-USD",
            "UNI-USD",
            "XLM-USD",
            "FIL-USD",
            "BNB-USD",
            "SOL-USD",
            "XRP-USD",
            "ADA-USD",
            "SHIB-USD",
            "TON-USD",
            "DOGE-USD",
            "AVAX-USD",
            "TRX-USD",
            "DOT-USD",
            "MATIC-USD",
            "ETC-USD",
        ],
        train_starting_date="2020-09-22",
        train_ending_date="2022-06-07",
        test_starting_date="2022-06-08",
        test_ending_date="2023-01-03",
        THRESHOLD_PARAMETER=0.015,
        #oc_upper_threshold=0.01,
        #oc_lower_threshold=-0.01,
        #oc_k_loss=0.1,
        #oc_k_gain=0.1,
        #oc_n=0.725,
        #ra_upper_threshold=0.01,
        #ra_lower_threshold=-0.01,
        #ra_k_loss=0.1,
        #ra_k_gain=0.1,
        #ra_n=1.22,
    )

    oc_params = {
        'oc_upper_threshold': [0.01, 0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12],
        'oc_lower_threshold': [-0.01, -0.02, -0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.09,-0.10,-0.12],
        'oc_k_loss': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'oc_k_gain': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'oc_n': [1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.40],
    }

    ra_params = {
        'ra_upper_threshold': [0.0,0.01, 0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
        'ra_lower_threshold': [0.0,-0.01, -0.02, -0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.09,-0.1],
        'ra_k_loss': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'ra_k_gain': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'ra_n': [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    }
    
        
    run_grid_search(config, model_rl, forecast_fn, "ra", ra_params, val_fraction=0.20)
    run_grid_search(config, model_rl, forecast_fn, "oc", oc_params, val_fraction=0.20)
    
    print("\nBest parameters found:\n", 
        "oc_upper_threshold=", config.oc_upper_threshold, \
        "oc_lower_threshold=", config.oc_lower_threshold, \
        "oc_k_loss=", config.oc_k_loss, \
        "oc_k_gain=", config.oc_k_gain, \
        "oc_n=", config.oc_n, \
        "ra_upper_threshold=", config.ra_upper_threshold, \
        "ra_lower_threshold=", config.ra_lower_threshold, \
        "ra_k_loss=", config.ra_k_loss, \
        "ra_k_gain=", config.ra_k_gain, \
        "ra_n=", config.ra_n)   

if __name__ == "__main__":
    findparams()