from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from BBAPT.config.config import appConfig

def maximum_sharpe_ratio_model(train_df,test_df):
    model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
    )
    model.fit(train_df)
    action = model.predict(test_df)
    return {"weights":action,"portfolio_sharpe_ratio":action.sharpe_ratio}

def minimum_tail_risk_model(train_df,test_df):
    model = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_CVAR,
    risk_measure=RiskMeasure.CVAR,
    )
    model.fit(train_df)
    action = model.predict(test_df)
    return {"weights":action,"portfolio_sharpe_ratio":action.sharpe_ratio}

def equally_weighted_portfolio(config):
    return {"weights":np.ones(config.n_stocks)/config.n_stocks,"portfolio_sharpe_ratio":np.nan}
    
def minimum_variance_portfolio(train_df,test_df):
    model = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.VARIANCE,
    )
    model.fit(train_df)
    action = model.predict(test_df)
    return {"weights":action,"portfolio_sharpe_ratio":action.sharpe_ratio}

def main():
    config = appConfig(
        initial_balance=100000,
        allow_short_selling=False,
        technical_indicator_list=["MACD", "RSI_14", "CCI_20", "SMA_20", "EMA_20"],
        transaction_cost=0.001,
        ticker_list=["BTC-USD", "ETH-USD", "LTC-USD", "LINK-USD", "BCH-USD", "UNI-USD", "XLM-USD", "FIL-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "SHIB-USD", "TON-USD", "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "ETC-USD"],

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
    train_df = data_prep(
        config.train_starting_date,
        config.train_ending_date,
        config.ticker_list,
    )
    test_df = data_prep(
        config.test_starting_date,
        config.test_ending_date,
        config.ticker_list,
    )
    maximum_sharpe_ratio_model_result = maximum_sharpe_ratio_model(train_df,test_df)
    minimum_tail_risk_model_result = minimum_tail_risk_model(train_df,test_df)
    equally_weighted_portfolio_result = equally_weighted_portfolio(config)
    minimum_variance_portfolio_result = minimum_variance_portfolio(train_df,test_df)

    print("\n-----Running Baseline Models ----\n")
    
    print("---Maximum Sharpe Ratio Model---")
    print(f"Portfolio_sharpe_ratio: {maximum_sharpe_ratio_model_result['portfolio_sharpe_ratio']}")
    print(f"Weights: {maximum_sharpe_ratio_model_result['weights']}")

    print("---Minimum Tail Risk Model---")
    print(f"Portfolio_sharpe_ratio: {minimum_tail_risk_model_result['portfolio_sharpe_ratio']}")
    print(f"Weights: {minimum_tail_risk_model_result['weights']}")

    print("---Equally Weighted Portfolio---")
    print(f"Portfolio_sharpe_ratio: {equally_weighted_portfolio_result['portfolio_sharpe_ratio']}")
    print(f"Weights: {equally_weighted_portfolio_result['weights']}")
    
    print("---Minimum Variance Portfolio---")
    print(f"Portfolio_sharpe_ratio: {minimum_variance_portfolio_result['portfolio_sharpe_ratio']}")
    print(f"Weights: {minimum_variance_portfolio_result['weights']}")

if __name__ == "__main__":
    main()
    
    
    