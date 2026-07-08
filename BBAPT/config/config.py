from dataclasses import dataclass, field

@dataclass
class appConfig():
    initial_balance: float = 100000
    allow_short_selling: bool = False
    technical_indicator_list: list = field(default_factory=lambda: ['MACD','RSI_14','CCI_20','SMA_20','EMA_20']) #,'ADX_14'
    transaction_cost: float = 0.001
    tickers: list[str]= field(default_factory=lambda: ["BTC-USD", "ETH-USD", "LTC-USD", "LINK-USD", "BCH-USD", "UNI-USD", "XLM-USD", "FIL-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "SHIB-USD", "TON-USD", "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "ETC-USD"])
    train_starting_date:str = "2020-09-22" #Paper date: 2019-02-06, #All assets are available: 2020-09-22
    train_ending_date:str = "2022-06-07"
    test_starting_date:str = "2022-06-08"
    test_ending_date:str = "2023-01-03"
    sharpe_ratio_window:int = 22
    
    #PARAMETERS FOR BEHAVIOURAL MAPPING 
    THRESHOLD_PARAMETER:float = 0.015    
    #overconfidence
    oc_upper_threshold: float = 0.01
    oc_lower_threshold: float = -0.01
    oc_k_loss: float = 0.1
    oc_k_gain: float = 0.1
    oc_n: float = 1.22              
    #risk averse
    ra_upper_threshold: float = 0.01            
    ra_lower_threshold: float = -0.01   
    ra_k_loss: float = 0.1
    ra_k_gain: float = 0.1
    ra_n: float = 0.725

    def __post_init__(self):
        self.n_stocks = len(self.tickers)
        self.n_indicators = len(self.technical_indicator_list) if self.technical_indicator_list is not None else 0
        self.ticker_list = sorted(self.tickers)