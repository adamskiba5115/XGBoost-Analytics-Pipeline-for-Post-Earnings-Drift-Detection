import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, accuracy_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
import json
import os
from config import engine, BASE_DIR

def run_model():
    df_features = pd.read_sql('features', engine)
    # df_features = df_features[df_features['sector'] != 'Healthcare'].copy()



    # Loading price data
    df_ceny_ml = pd.read_sql("SELECT ticker, Date, Close, Open FROM all_prices", engine)

    # Standardizing time formats and removing potential duplicates in market data
    df_ceny_ml['Date'] = pd.to_datetime(df_ceny_ml['Date'], utc=True).dt.tz_localize(None)
    df_ceny_ml = df_ceny_ml.sort_values(['ticker', 'Date']).reset_index(drop=True)
    df_ceny_ml = df_ceny_ml.drop_duplicates(subset=['ticker', 'Date'])


    # Defining the Target variable
    # Opening a position: Closing price on the next day following the signal
    df_ceny_ml['Reference_Value_Start'] = df_ceny_ml['Close'].shift(-1)
    # Closing a position: Closing price after 9 trading days
    df_ceny_ml['Reference_Value_End'] = df_ceny_ml.groupby('ticker')['Close'].shift(-9)


    # Calculating logarithmic rates of return
    df_ceny_ml['Target_10d'] = np.log(df_ceny_ml['Reference_Value_End']) - np.log(df_ceny_ml['Reference_Value_Start'])
    df_ceny_ml['Date_signal'] = df_ceny_ml['Date']


    # Merging features with the target variable
    df_features = df_features.merge(
        df_ceny_ml[['ticker', 'Date_signal', 'Target_10d']],
        left_on=['ticker','Date'],
        right_on=['ticker','Date_signal'],
        how='left'
    )

    target = 'Target_10d'
    df_features_copy = df_features.copy()

    # Final cleaning of the dataset
    df_features = df_features.dropna(subset=[target])

    # Eliminating extreme outliers for the rate of return
    df_features = df_features[df_features['Target_10d'].between(-0.5, 0.5)]

    # Removing companies with low nominal prices (Penny Stocks)
    df_features = df_features[df_features['Close'] > 20]

    # Removing records with an undefined sector and performing chronological sorting
    df_features = df_features[~df_features['sector'].isin(['unknown'])]
    df_features = df_features.reset_index(drop=True)
    df_features = df_features.sort_values('Date').reset_index(drop=True)

    print(f"Number of records after filtering: {len(df_features)}")

    # Defining columns to be removed before the training process (Feature Selection)
    drop_cols = [
        'Date', 'Date_signal', 'ticker', 
        target, 'Target_10d', 'Target_rel',
        'Close', 'Open', 'Reference_Value_Start', 'Reference_Value_End','High','Low',
        'Volume', 'Return_1d',
        'sector', 
        'Vol_vs_Market_Panic_log',
        'VIX_Zscore_60d',
        'VIX_raw',
        'gross_margin',
        'gross_margin_log_delta_QoQ',
        'gross_margin_delta_vs_sector_z',
        'Revenue Growth QoQ',
        'sales_per_share_log_delta_QoQ',
        'EPS_QoQ',
        'Volatility_5d_ewm',
        'Volatility_10d_ewm',
        'debt_to_equity_log_delta_QoQ',
        'margin_gross_log_delta_QoQ',
        'margin_operating_log_delta_QoQ',
        'roa_log_delta_QoQ',
        'roe_log_delta_QoQ',
        'expense_ratio_operating_log_delta_QoQ',
        'expense_ratio_cogs_log_delta_QoQ',
        'debt_to_assets_log_delta_QoQ',
        'cash_to_assets_log_delta_QoQ',
        'cash_to_payables_log_delta_QoQ',
        'asset_turnover_log_delta_QoQ',
        'ppe_turnover_log_delta_QoQ',
        'eps_basic_log_delta_QoQ',
        'bvps_log_delta_QoQ',
        'margin_gross_delta_vs_sector_z',
        'margin_operating_delta_vs_sector_z',
        'margin_net_delta_vs_sector_z',
        'roa_delta_vs_sector_z',
        'roe_delta_vs_sector_z',
        'expense_ratio_operating_delta_vs_sector_z',
        'expense_ratio_cogs_delta_vs_sector_z',
        'debt_to_equity_delta_vs_sector_z',
        'cash_to_payables_delta_vs_sector_z',
        'asset_turnover_delta_vs_sector_z',
        'ppe_turnover_delta_vs_sector_z',
        'bvps_delta_vs_sector_z',
        'sales_per_share_delta_vs_sector_z',
        'Return_1d_lagged_log',
        'cash_to_payables',
        'ppe_turnover',
        'Log_Assets',
    ]
    # Cleaning up leftovers from previous versions
    old_features_to_drop = [
        'ROE_vs_peers', 'Momentum_20d_vs_peers',
        'Volatility_20d_vs_peers', 'Leverage', 'Leverage_vs_sector',
        'Volatility_5d', 'Volatility_10d', 'Volatility_vs_VIX',
        'Return_10d_pre' 
    ]
    drop_cols.extend(old_features_to_drop)

    # Removing identifiers and datetime data types from the feature matrix X
    X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    X = X.select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])

    # Removing any remaining columns that contain dates in their names
    X = X.loc[:, ~X.columns.str.contains('Date', case=False)]

    # Replacing infinite values with missing values (NaN)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Removing columns with more than 80% missing values
    X = X.loc[:, X.isnull().mean() < 0.8]

    # Filling missing values using the previous available value (forward fill) within each specific company
    X = X.groupby(df_features['ticker'], group_keys=False).apply(lambda g: g.ffill(limit=1))

    # Unifying the data type to float64
    X = X.astype('float64')

    # Defining the target variable and the date vector for time-based validation
    y_reg = df_features[target]
    dates = df_features['Date']


    random_returns = []
    importance_list = []
    cumulative_performance = 1.0
    performance_curve = []
    all_positions = []
    COST_SHORT = 0.005 

    # Removing duplicates in the feature table [Date, Ticker]
    df_features = df_features.drop_duplicates(subset=['Date', 'ticker'])


    def run_backtest_for_period(df_features, train_years, test_year, target, drop_cols, xgb_params, threshold, position_size, transaction_costs =0.0050):
        
        # Splitting the data into training and test sets
        mask_train = df_features['Date'].dt.year.isin(train_years)
        mask_test = df_features['Date'].dt.year == test_year
        
        # Converting masks into index lists
        train_idx = df_features[mask_train].index.tolist()
        test_idx = df_features[mask_test].index.tolist()

        # Protecting against missing data
        if len(train_idx) == 0 or len(test_idx) == 0:
            return None

        df_train = df_features.iloc[train_idx].copy().reset_index(drop=True)
        df_test = df_features.iloc[test_idx].copy().reset_index(drop=True)
        
        # Defining columns to be removed before training
        base_drop = ['Date', 'ticker', 'sector', 'Close', target, 'Target_10d']
        
        # Preparing the training dataset
        X_train = df_train.drop(columns=[c for c in base_drop if c in df_train.columns])
        X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], errors='ignore')
        X_train = X_train.select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])
        X_train = X_train.replace([np.inf, -np.inf], np.nan).astype('float64')
        
        # Preparing the test dataset
        X_test = df_test.drop(columns=[c for c in base_drop if c in df_test.columns])
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors='ignore')
        
        # Ensuring the training and test sets have an identical set of columns
        cols = [c for c in X_train.columns if c in X_test.columns]
        X_train = X_train[cols]
        X_test = X_test[cols].replace([np.inf, -np.inf], np.nan).astype('float64')

        y_train = df_train[target]
        
        

        # Training the model and recording feature importance
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        importance_list.append(model.feature_importances_)
        y_pred = model.predict(X_test)

        
        # Identifying positions based on the optimized threshold (Optuna)
        df_test['pred'] = y_pred
        top_signals = df_test[df_test['pred'] <= threshold]
        
        # Handling cases where no signals are generated during a given test period
        if len(top_signals) == 0:
            return {'test_year': test_year, 'final_positions': 0, 'raw_ret': 0, 'ret_net': 0}

        # Calculating average gross and net returns
        ret_raw = -top_signals[target].mean()
        ret_net = ret_raw - transaction_costs

        final_positions = top_signals[['Date', 'ticker', target]].copy()
        final_positions['ret'] = -final_positions[target] - transaction_costs 
        final_positions['position_size'] = position_size
        all_positions.append(final_positions)
        

        # Returning position details for subsequent analysis
        return {
            'test_year': test_year,
            'final_positions': len(top_signals),
            'ret_raw': ret_raw,
            'ret_net': ret_net,
            'cols' : cols
        }



    # Period dictionary (walk-forward with dynamic parameters)
  
    json_path = os.path.join(BASE_DIR, 'optimized_params.json')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            periods = json.load(f)
    except FileNotFoundError:
        periods = []
    except Exception as e:
        periods = []

    results_summary = []
    cumulative_performance = 1.0
    cols_lista = []

    # Iteration over defined time windows (walk-forward)
    for period in periods:
        train_years = period['train']
        test_year = period['test']
        
        print(f"Train years: {train_years} --> Test years: {test_year}")
        

        # Executing predictions and model evaluation for the selected time window
        res = run_backtest_for_period(
            df_features=df_features,
            train_years=train_years,
            test_year=test_year,
            target=target, 
            drop_cols=drop_cols, 
            xgb_params=period['xgb_params'], 
            threshold=period['threshold'],
            position_size=period['position_size'],  
            transaction_costs=0.0050
        )
        if res is not None:
            if 'cols' in res:
                cols_lista.append(res['cols'])

        # Updating cumulative model score and recording metrics
        if res is not None:
            cumulative_performance *= (1 + res['ret_net'])
            res['performance_cummulative'] = cumulative_performance
            res['position_size'] = period['position_size']
            results_summary.append(res)
            
            print(f"  Number of identified observations: {res['final_positions']}")
            print(f"  Prediction efficiency (net return): {res['ret_net']:.4f}")
            print(f"  Cumulative model performance (total performance): {cumulative_performance:.4f}\n")
        else:
            print(f"  No data available for the test period {test_year}.\n")


    df_results = pd.DataFrame(results_summary)
    print("=== Summary of model results (WALK-FORWARD) ===")
    print(df_results[['test_year', 'final_positions', 'ret_raw', 'ret_net', 'performance_cummulative', 'position_size']])


    # Feature importance analysis
    avg_importance = np.mean(importance_list, axis=0)
    feat_imp = pd.Series(avg_importance, index=cols_lista[0]).sort_values(ascending=False)

    # Visualization of the top 15 variables with the highest impact on model results
    feat_imp.head(15).plot(kind='barh')
    plt.tight_layout()
    plt.show()

    df_positions = pd.concat(all_positions).sort_values('Date').reset_index(drop=True)
    df_positions.to_sql('all_positions', engine, if_exists='replace', index=False)

if __name__ == "__main__":
    run_model()