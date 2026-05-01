
# %%
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import os
import optuna 
from config import engine


def run_optimization():
    pd.options.display.float_format = '{:.10f}'.format

    df_features = pd.read_sql('features', engine)
    df_prices = pd.read_sql("SELECT ticker, Date, Close, Open FROM all_prices", engine)

    # Standardization of time format and removal of potential duplicates in market data
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], utc=True).dt.tz_localize(None)
    df_prices = df_prices.sort_values(['ticker', 'Date']).reset_index(drop=True)
    df_prices = df_prices.drop_duplicates(subset=['ticker', 'Date'])


    # Defining the Target
    # Position Entry: Closing price on the next day following the signal
    df_prices['Reference_Value_Start'] = df_prices['Close'].shift(-1)
    # Position Exit: Closing price after 9 trading sessions
    df_prices['Reference_Value_End'] = df_prices.groupby('ticker')['Close'].shift(-9)

    # Calculating logarithmic returns
    df_prices['Target_10d'] = np.log(df_prices['Reference_Value_End']) - np.log(df_prices['Reference_Value_Start'])
    df_prices['Date_signal'] = df_prices['Date']

    # Merging features with the Target variable
    df_features = df_features.merge(
        df_prices[['ticker', 'Date_signal', 'Target_10d']],
        left_on=['ticker','Date'],
        right_on=['ticker','Date_signal'],
        how='left'
    )

    target = 'Target_10d'
    df_features_copy = df_features.copy()

    # Final dataset cleaning
    df_features = df_features.dropna(subset=[target])

    # Elimination of extreme outliers for returns
    df_features = df_features[df_features['Target_10d'].between(-0.5, 0.5)]

    # Removal of companies with low nominal prices (Penny Stocks)
    df_features = df_features[df_features['Close'] > 20]

    # Removal of records with undefined sectors and chronological ordering
    df_features = df_features[~df_features['sector'].isin(['unknown'])]
    df_features = df_features.reset_index(drop=True)
    df_features = df_features.sort_values('Date').reset_index(drop=True)

    # Definition of columns to be dropped before the training process (Feature Selection)
    drop_cols = [
        'Date', 'Date_signal', 'ticker', 
        target, 'Target_10d', 'Target_rel', 
        'Close', 'Open', 'Reference_Value_Start', 'Reference_Value_End','High','Low',
        'Volume', 'Return_1d',
        'sector', 
        'Vol_vs_Market_Panic_log','VIX_Zscore_60d', 'VIX_raw',
        'gross_margin', 'gross_margin_log_delta_QoQ', 'gross_margin_delta_vs_sector_z',
        'Revenue Growth QoQ','sales_per_share_log_delta_QoQ','EPS_QoQ',
        'Volatility_5d_ewm','Volatility_10d_ewm','debt_to_equity_log_delta_QoQ',
        'margin_gross_log_delta_QoQ', 'margin_operating_log_delta_QoQ', 'roa_log_delta_QoQ',
        'roe_log_delta_QoQ', 'expense_ratio_operating_log_delta_QoQ', 'expense_ratio_cogs_log_delta_QoQ',
        'debt_to_assets_log_delta_QoQ', 'cash_to_assets_log_delta_QoQ', 'cash_to_payables_log_delta_QoQ',
        'asset_turnover_log_delta_QoQ', 'ppe_turnover_log_delta_QoQ', 'eps_basic_log_delta_QoQ',
        'bvps_log_delta_QoQ', 'margin_gross_delta_vs_sector_z', 'margin_operating_delta_vs_sector_z',
        'margin_net_delta_vs_sector_z', 'roa_delta_vs_sector_z', 'roe_delta_vs_sector_z',
        'expense_ratio_operating_delta_vs_sector_z', 'expense_ratio_cogs_delta_vs_sector_z',
        'debt_to_equity_delta_vs_sector_z', 'cash_to_payables_delta_vs_sector_z',
        'asset_turnover_delta_vs_sector_z', 'ppe_turnover_delta_vs_sector_z',
        'bvps_delta_vs_sector_z', 'sales_per_share_delta_vs_sector_z',
        'Return_1d_lagged_log', 'cash_to_payables', 'ppe_turnover', 'Log_Assets' 
    ]

    # Cleaning up remnants from previous versions
    old_features_to_drop = [
        'ROE_vs_peers', 'Momentum_20d_vs_peers',
        'Volatility_20d_vs_peers', 'Leverage', 'Leverage_vs_sector',
        'Volatility_5d', 'Volatility_10d', 'Volatility_vs_VIX',
        'Return_10d_pre'
    ]
    drop_cols.extend(old_features_to_drop)


    # Time series cross-validation configuration
    # gap=10 prevents data leakage given the 10-day target horizon
    n_splits = 8
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=10)

    # Initialization of structures for prediction results and feature importance
    wszystkie_predykcje = {fold: [] for fold in range(1, n_splits + 1)}
    importance_lista = []

    # Cleaning the feature matrix X of identifiers and datetime data types
    X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    X = X.select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])

    # Removal of remaining columns containing dates in the name
    X = X.loc[:, ~X.columns.str.contains('Date', case=False)]

    # Verification of data type correctness
    print("\nCHECK DATA TYPES:")
    print(X.dtypes.value_counts())
    print("\nOBJECT COLS:")
    print(X.select_dtypes(include='object').columns.tolist())
    print("Kolumny datetime w X:")
    print(X.select_dtypes(include=['datetime64[ns]']).columns.tolist())

    # Replacement of infinite values with missing values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Removal of columns with more than 80% missing values
    X = X.loc[:, X.isnull().mean() < 0.8]

    # Filling missing values with the previous value within a given company
    X = X.groupby(df_features['ticker'], group_keys=False).apply(lambda g: g.ffill(limit=1))

    # Unifying data type to float64
    X = X.astype('float64')


    # Defining the target variable and the date vector for time-based validation
    y_reg = df_features[target]
    dates = df_features['Date']


    # Time series validation configuration (5 splits with a 10-day gap)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=10)


    returns = []
    random_returns = []
    importance_lista = []
    cumulative_performance = 1.0
    equity_curve = []
    all_positions = []
    TRANSACTION_COST = 0.0015  

    # Removal of duplicates in the feature table [Date, Ticker]
    df_features = df_features.drop_duplicates(subset=['Date', 'ticker'])

    def run_full_backtest(model, X, y_reg, target, TRANSACTION_COST, tscv, df_features, drop_cols, signal_threshold, position_size):
        all_positions = []
        
        # Walk-Forward Validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

            # Split into training and test sets for the given fold
            df_train = df_features.iloc[train_idx].copy()
            df_test_fold = df_features.iloc[test_idx].copy()

            # Preparation of the feature matrix by removing unnecessary columns
            X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
            X_test_fold = df_test_fold.drop(columns=[c for c in drop_cols if c in df_test_fold.columns], errors='ignore')

            # Alignment of the column set and unification of data types to float64
            cols = [c for c in X_train.columns if c in X_test_fold.columns]
            X_train = X_train[cols].select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns, UTC]']).astype('float64')
            X_test_fold = X_test_fold[cols].select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns, UTC]']).astype('float64')


            # Model fitting and generation of predictions for the current fold
            y_train = y_reg.iloc[train_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_fold)

            df_test_fold['pred'] = y_pred
            
            # Generating signals based on a decision threshold optimized by Optuna
            top_signals = df_test_fold[df_test_fold['pred'] <= signal_threshold].copy()
            
            if not top_signals.empty:
                temp_pos = pd.DataFrame({
                    'Date': pd.to_datetime(top_signals['Date']),
                    'ret': -top_signals[target] - TRANSACTION_COST 
                })
                all_positions.append(temp_pos)

        # Returning a penalty value in case no signals are generated
        if not all_positions:
            return -999.0

        # Merging all positions into a single, chronologically ordered dataset
        df_positions = pd.concat(all_positions).sort_values('Date').reset_index(drop=True)

        total_signals = len(df_positions)

        # Determining the number of years based on unique years in the results
        years_in_test = df_positions['Date'].dt.year.nunique()
        years_in_test = max(years_in_test, 1)

        # Returning a penalty value for an insufficient or excessive number of positions in the period
        if total_signals < (years_in_test * 50) or total_signals > (years_in_test * 350):
            return -999.0

        # Determining the position exit date
        df_positions['exit_date'] = df_positions['Date'] + pd.Timedelta(days=10)
        
        # Generating a timeline covering the entire test period (business days only)
        dates = pd.date_range(start=df_positions['Date'].min(), end=df_positions['exit_date'].max(), freq='B')
        
        cumulative_performance = 1.0
        active_positions = []
        equity_curve = []
        last_date = dates[-1]

        
        for current_date in dates:
            still_active = []
            for pos in active_positions:
                if pos['exit_date'] <= current_date:
                    # Updating the cumulative score
                    cumulative_performance *= (1 + pos['ret'] * pos['size'])
                else:
                    still_active.append(pos)
            active_positions = still_active

            
            if current_date != last_date:
        
                new_positions = df_positions[df_positions['Date'] == current_date]
                for _, row in new_positions.iterrows():
                    # Managing maximum exposure
                    max_exposure = 1.0
                    current_exposure = sum(p['size'] for p in active_positions)
                    available = max_exposure - current_exposure
                    
                    # Determining the optimal weight for a new sample
                    if available > 0:
                        pos_size = min(position_size, available)
                        active_positions.append({
                            'ret': row['ret'],
                            'exit_date': row['exit_date'],
                            'size': pos_size
                        })
            
            equity_curve.append(cumulative_performance)

        df_equity = pd.DataFrame({'equity': equity_curve})
        df_equity['returns'] = df_equity['equity'].pct_change().fillna(0)
        
        # Protection against zero standard deviation
        if df_equity['returns'].std() == 0:
            return -999.0
        
        return cumulative_performance

    all_best_params = {}

    training_years_list = [
        [2021],
        [2021, 2022],
        [2021,2022,2023],
        [2021, 2022, 2023, 2024]
    ]

    n_splits = 8
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=10)
    TRANSACTION_COST = 0.0050
    target = 'Target_10d' 
    drop_cols = ['Date', 'ticker', 'sector', 'Close', target]

    for train_years in training_years_list:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION FOR PERIOD: {train_years}")
        print(f"{'='*60}")

        # Creating a data subset for the specified training years
        df_current = df_features[df_features['Date'].dt.year.isin(train_years)].copy().reset_index(drop=True)
        
        if df_current.empty:
            print(f"MISSING DATA FOR PERIOD: {train_years}")
            continue

        # Cleaning the feature matrix X by removing identifiers and datetime data types
        X_current = df_current.drop(columns=[c for c in drop_cols if c in df_current.columns], errors='ignore')
        X_current = X_current.select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns, UTC]']).astype('float64')

        # Removing remaining columns containing dates in their names
        X_current = X_current.loc[:, ~X_current.columns.str.contains('Date', case=False)]
        
        # Replacing infinite values with NaN
        X_current = X_current.replace([np.inf, -np.inf], np.nan)

        # Removing columns that have more than 80% missing values
        X_current = X_current.loc[:, X_current.isnull().mean() < 0.8]

        # Filling missing values with the previous available value (forward fill) within a specific company
        X_current = X_current.groupby(df_features['ticker'], group_keys=False).apply(lambda g: g.ffill(limit=1))


        X_current = X_current.astype('float64')
        y_reg_current = df_current[target]

        def objective(trial):
            params = {
                "n_estimators" : trial.suggest_int("n_estimators", 300, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
                "min_child_weight": trial.suggest_int("min_child_weight", 4, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1
            }
            signal_threshold = trial.suggest_float('signal_threshold', -0.075, -0.025, step=0.005)
            position_size = trial.suggest_float('position_size', 0.1, 0.20, step=0.01)

            model = XGBRegressor(**params)

            return run_full_backtest(
                model=model,
                X=X_current,
                y_reg=y_reg_current,
                target=target,
                TRANSACTION_COST=TRANSACTION_COST,
                tscv=tscv,
                df_features=df_current,
                drop_cols=drop_cols,
                signal_threshold=signal_threshold,
                position_size=position_size
            )

        # Optimization
        sampler = optuna.samplers.TPESampler(
        multivariate=True,
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        n_startup_trials=1,
        gamma=lambda n: int(0.15 * np.sqrt(n)),
        constant_liar=True,
        seed=42
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=1, timeout=None) 

        all_best_params[str(train_years)] = study.best_params
        
        print(f"\nFINISHED PERIOD: {train_years}")
        print(f"THE HIGHEST FINAL CAPITAL: {study.best_value:.4f}")
        print(f"THE BEST HYPERPARAMETER CONFIGURATION: {study.best_params}")

    # Final summary of all periods
    print("\n" + "#"*30)
    print("PARAMS RAPORT")
    print("#"*30)
    for years, params in all_best_params.items():
        print(f"YEARS: {years}:")
        print(params)

    periods_output = []
    for years, params in all_best_params.items():
        # Zabezpieczenie: jeśli klucz 'years' jest stringiem (np. "[2021]"), zamieniamy na listę, 
        # jeśli jest już krotką/listą, po prostu używamy.
        if isinstance(years, str):
            import ast
            train_years = ast.literal_eval(years)
        else:
            train_years = list(years)
            
        # Wyliczenie test year (największa wartość w train + 1)
        test_year = max(train_years) + 1
        
        # Tworzenie zagnieżdżonej struktury z wymaganymi kluczami
        period_dict = {
            'train': train_years,
            'test': test_year,
            'xgb_params': {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'min_child_weight': int(params['min_child_weight']),
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'random_state': 42,
                'n_jobs': -1
            },
            # Obsługa różnego nazewnictwa thresholda
            'threshold': params.get('signal_threshold', params.get('short_threshold')),
            'position_size': params['position_size']
        }
        periods_output.append(period_dict)

    import json
    import os

    # 2. Ustalenie ścieżki do folderu, w którym znajduje się ten skrypt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'optimized_params.json')

    # 3. Zapis listy słowników do pliku JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(periods_output, f, indent=4)

if __name__ == "__main__":
    run_optimization()











