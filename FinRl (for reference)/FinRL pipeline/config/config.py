"""Application configuration for the portfolio allocation experiment."""

from dataclasses import dataclass, field

from finrl import config_tickers

@dataclass(frozen=True)
class DirectoryConfig:
    DATA_SAVE_DIR = "data"
    TRAINED_MODEL_DIR = "trained_models"
    TENSORBOARD_LOG_DIR = "tensorboard_log"
    RESULTS_DIR = "results"


@dataclass(frozen=True)
class DataPrepConfig:
    start_date: str = "2008-01-01"
    end_date: str = "2022-06-02"
    train_start_date: str = "2009-04-01"
    train_end_date: str = "2020-03-31"
    lookback: int = 252
    ticker_list: list[str] = field(default_factory=lambda: config_tickers.DOW_30_TICKER)
    use_technical_indicator: bool = True
    use_turbulence: bool = False
    user_defined_feature: bool = False


@dataclass(frozen=True)
class EnvironmentConfig:
    hmax: int = 100
    initial_amount: int = 1_000_000
    transaction_cost_pct: float = 0
    reward_scaling: float = 1e-1
    tech_indicator_list: list[str] = field(
        default_factory=lambda: ["macd", "rsi_30", "cci_30", "dx_30"]
    )


@dataclass(frozen=True)
class TrainingConfig:
    a2c_total_timesteps: int = 40_000
    ppo_total_timesteps: int = 40_000
    a2c_params: dict[str, float | int] = field(
        default_factory=lambda: {
            "n_steps": 10,
            "ent_coef": 0.005,
            "learning_rate": 0.0004,
        }
    )
    ppo_params: dict[str, float | int] = field(
        default_factory=lambda: {
            "n_steps": 2048,
            "ent_coef": 0.005,
            "learning_rate": 0.001,
            "batch_size": 128,
        }
    )


@dataclass(frozen=True)
class EvaluationConfig:
    trade_start_date: str = "2020-04-01"
    trade_end_date: str = "2022-05-31"
    baseline_ticker: str = "^DJI"
    baseline_start_date: str = "2020-07-01"
    baseline_end_date: str = "2021-09-01"
    multi_step_window: int = 20


@dataclass(frozen=True)
class TestingConfig:
    random_state: int = 0
    rf_max_depth: int = 35
    rf_min_samples_split: int = 10
    dt_max_depth: int = 35
    dt_min_samples_split: int = 10
    svm_epsilon: float = 0.14


@dataclass(frozen=True)
class LoggingConfig:
    wandb_project: str = "bbapt-portfolio-allocation"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    log_to_wandb: bool = True


@dataclass(frozen=True)
class AppConfig:
    data_prep: DataPrepConfig = field(default_factory=DataPrepConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
