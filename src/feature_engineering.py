import pandas as pd
import yfinance as yf
import numpy as np
from sqlalchemy import create_engine
import os
from config import engine


def run_feature_engineering():
    pd.options.display.float_format = '{:.10f}'.format

    query_prices    = "SELECT ticker, Date, Open, Close, Volume FROM all_prices ORDER BY ticker, Date"
    df_p = pd.read_sql(query_prices, engine)
    df_final = pd.read_sql('df_final', engine)

    def calculate_indicators(df):
        # Chronological data ordering within groups for each company
        df = df.sort_values(['ticker','Date']).copy()    

        def safe_ratio(numerator, denominator, index):
            if numerator is not None and denominator is not None:
                return np.where(
                    (numerator.notnull()) & (denominator.notnull()) & (denominator != 0),
                    numerator / denominator,
                    np.nan
                )
            return pd.Series(np.nan, index=index, dtype='float64')
        



    # Extraction of key items from the balance sheet and income statement
        revenue = df.get('revenue')
        operating_income = df.get('operating_income')
        net_income = df.get('net_income')
        shares = df.get('shares')
        tax_provision = df.get('tax_provision')
        operating_expenses = df.get('operating_expenses')
        cost_revenue = df.get('cost_revenue')
        gross_profit = df.get('gross_profit')
        assets = df.get('assets')
        liabilities = df.get('liabilities')
        equity = df.get('equity')
        cash = df.get('cash')
        payables = df.get('payables')
        ppe = df.get('ppe')

        # Calculation of profitability, debt, and operational efficiency ratios
        df['margin_gross'] = safe_ratio(gross_profit, revenue, df.index)
        df['margin_operating'] = safe_ratio(operating_income, revenue, df.index)
        df['margin_net'] = safe_ratio(net_income, revenue, df.index)
        df['gross_margin'] = safe_ratio(gross_profit, revenue,df.index)
        df['roa'] = safe_ratio(net_income, assets, df.index)
        df['roe'] = safe_ratio(net_income, equity, df.index)
        
        # Calculation of cost structure and solvency ratios
        df['expense_ratio_operating'] = safe_ratio(operating_expenses, revenue, df.index)
        df['expense_ratio_cogs'] = safe_ratio(cost_revenue, revenue, df.index)
        df['debt_to_equity'] = safe_ratio(liabilities, equity, df.index)
        df['debt_to_assets'] = safe_ratio(liabilities, assets, df.index)
        
        # Calculation of liquidity and asset turnover ratios
        df['cash_to_assets'] = safe_ratio(cash, assets, df.index)
        df['cash_to_payables'] = safe_ratio(cash, payables, df.index)
        df['asset_turnover'] = safe_ratio(revenue, assets, df.index)
        df['ppe_turnover'] = safe_ratio(revenue, ppe, df.index)
        
        # Calculation of market values per share
        df['eps_basic'] = safe_ratio(net_income, shares, df.index)
        df['bvps'] = safe_ratio(equity, shares, df.index)
        df['sales_per_share'] = safe_ratio(revenue, shares, df.index)


        
    # Calculation of revenue growth dynamics quarter-over-quarter (QoQ)
        df['Revenue Growth QoQ'] = (
            df.groupby('ticker')['revenue']
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )
        
        def safe_log(series, index):
            if series is not None:
                return pd.Series(
                    np.where(
                        (series.notnull()) & (series > -1),
                        np.log1p(series),
                        np.nan
                    ),
                    index=index, dtype='float64'
                )
            return pd.Series(np.nan, index=index, dtype='float64')
        


    # Logarithmic transformation (revenue and assets)
        df['Log_Revenue'] = safe_log(df.get('revenue'), df.index)
        df['Log_Assets']  = safe_log(df.get('assets'), df.index)
        
        log_rev = df.get('Log_Revenue')
        log_assets = df.get('Log_Assets')

        # Calculation of logarithmic differences
        if log_rev is not None:
            df['Log_Revenue_delta'] = log_rev.groupby(df['ticker']).diff()
        else:
            df['Log_Revenue_delta'] = pd.Series(np.nan, index=df.index)

        if log_assets is not None:
            df['Log_Assets_delta'] = log_assets.groupby(df['ticker']).diff()
        else:
            df['Log_Assets_delta'] = pd.Series(np.nan, index=df.index)


    # Calculation of earnings per share growth dynamics (EPS QoQ)
        eps = df.get('eps_basic')
        if eps is not None:
            prev_eps = eps.groupby(df['ticker']).shift(1)
            diff = eps - prev_eps
            # Application of an adjusted denominator (sMAPE measure) to handle transitions from negative to positive values
            denom = (eps.abs() + prev_eps.abs()) / 2 + 1e-9
            df['EPS_QoQ'] = diff / denom
            df['EPS_QoQ'] = df['EPS_QoQ'].replace([np.inf, -np.inf], np.nan)
        else:
            df['EPS_QoQ'] = pd.Series(np.nan, index=df.index)


    # Calculation of the EPS Surprise ratio
        eps_estimate = df.get('EPS Estimate')
        eps_reported = df.get('Reported EPS')

        if eps_estimate is not None and eps_reported is not None:
            # Difference between reported earnings and analyst estimates
            diff = eps_reported - eps_estimate
            
            denom = (eps_reported.abs() + eps_estimate.abs()) / 2 + 1e-9
            
            df['EPS_Surprise'] = pd.Series(
                np.where(
                    (eps_estimate.notnull()) & (eps_reported.notnull()),
                    diff / denom,
                    np.nan
                ),
                index=df.index,
                dtype='float64'
            )
        else:
            df['EPS_Surprise'] = pd.Series(np.nan, index=df.index, dtype='float64')





    # List of financial ratios
        ratios = [
            'margin_gross','margin_operating','margin_net',
            'roa','roe',
            'expense_ratio_operating','expense_ratio_cogs','debt_to_equity', 'debt_to_assets',
            'cash_to_assets', 'cash_to_payables', 'asset_turnover', 'ppe_turnover', 'eps_basic',
            'bvps', 'sales_per_share'
        ]


        def signed_log(series):
            return np.sign(series) * np.log1p(series.abs())

        df = df.sort_values(['ticker', 'Date'])


        # Calculation of fundamental dynamics
        for w in ratios:
            col = df.get(w)
            if col is not None:
                log_col = signed_log(col)
                
                # Calculation of inter-period changes (QoQ) on logarithmic values
                raw_delta = log_col.groupby(df['ticker']).diff()
                
                final_delta = (
                    raw_delta.replace(0, np.nan)
                    .groupby(df['ticker']).ffill()
                    .fillna(0) 
                )

                df[f'{w}_log_delta_QoQ'] = final_delta.astype('float64').replace([np.inf, -np.inf], np.nan)
            else:
                df[f'{w}_log_delta_QoQ'] = np.nan


        # Relative analysis – Z-Score relative to sector
        for w in ratios:
            delta_col_name = f'{w}_log_delta_QoQ'
            col = df.get(delta_col_name)

            if col is None or col.isna().all():
                df[f'{w}_delta_vs_sector_z'] = np.nan
                continue

            df['_tmp_delta'] = col 

            # 1. Aggregation of daily average changes within sectors
            sector_daily = (
                df.groupby(['sector', 'Date'])['_tmp_delta']
                .mean()
                .rename('daily_mean')
                .reset_index()
                .sort_values('Date')
            )

            # Calculation of rolling sector statistics (90-day window)
            stats = (
                sector_daily.groupby('sector')
                .rolling('90D', on='Date', min_periods=10)['daily_mean']
                .agg(['mean', 'std'])
                .reset_index()
                .rename(columns={'mean': 'rolling_mean', 'std': 'rolling_std'})
            )

            # 3. Merging sector statistics with the main dataset
            sector_daily = sector_daily.merge(stats, on=['sector', 'Date'], how='left')
            df = df.merge(
                sector_daily[['sector', 'Date', 'rolling_mean', 'rolling_std']],
                on=['sector', 'Date'],
                how='left'
            )

            # Result standardization (Z-Score) – company evaluation against the sector
            df[f'{w}_delta_vs_sector_z'] = (
                (df['_tmp_delta'] - df['rolling_mean']) /
                df['rolling_std'].replace(0, np.nan)
            )

            # Cleaning temporary variables
            df.drop(columns=['_tmp_delta', 'rolling_mean', 'rolling_std'], inplace=True)

        return df

    features_ratio = calculate_indicators(df_final)




    def calculate_market_features(df):
        # Sorting data before calculating returns
        df = df.sort_values(['ticker','Date']).copy()
        
        # Calculating logarithmic return from the previous day
        # This prevents using the closing price of the day the model is intended to make a decision
        df['Return_1d_lagged_log'] = (
            np.log(df['Close']) - np.log(df.groupby('ticker')['Close'].shift(1))
        ).groupby(df['ticker']).shift(1)
        
        # Calculating Overnight Return
        # Measures market reaction between the previous session close and the current opening
        df['Return_Overnight_log'] = (
            np.log(df['Open']) - np.log(df.groupby('ticker')['Close'].shift(1))
        )
        

    # Calculation of Momentum indicators for various time horizons (monthly, quarterly, semi-annual)
        for d in [20, 60, 120]:
            # Logarithmic return with a 1-day lag
            df[f'Momentum_{d}d_log'] = (
                np.log(df['Close']) - np.log(df.groupby('ticker')['Close'].shift(d))
            ).groupby(df['ticker']).shift(1)
            
            # Standardization of momentum relative to a yearly rolling window (Z-Score)
            df[f'Momentum_{d}d_log_z'] = df.groupby('ticker')[f'Momentum_{d}d_log'].transform(
                lambda x: (x - x.rolling(252, min_periods=d).mean()) / x.rolling(252, min_periods=d).std()
            )
            df.drop(columns=[f'Momentum_{d}d_log'], inplace=True)

        # Calculation of historical volatility using exponentially weighted moving averages (EWM Volatility)
        for d in [5, 10, 20]:
            # Utilization of lagged returns to calculate current risk
            df[f'Volatility_{d}d_ewm'] = (
                df.groupby('ticker')['Return_1d_lagged_log']
                .ewm(span=d, adjust=False, min_periods=d)
                .std()
                .reset_index(level=0, drop=True) 
            )

    # Analysis of Volume Spikes
        grouped_vol = df.groupby('ticker')['Volume']
        v_yesterday = grouped_vol.shift(1)
        
        # Comparison of yesterday's volume to the average of the last month (20 days)
        v_avg_short = grouped_vol.shift(1).rolling(20, min_periods=10).mean()
        ratio = v_yesterday / v_avg_short
        
        # Normalization of volume spikes using Z-Score relative to one-year history
        spike_log = np.log1p(ratio.fillna(0)) 
        spike_grouped = spike_log.groupby(df['ticker'])
        r_mean = spike_grouped.shift(1).rolling(252, min_periods=20).mean()
        r_std = spike_grouped.shift(1).rolling(252, min_periods=20).std()
        
        z_raw = (spike_log - r_mean) / r_std.replace(0, np.nan)
        
        # Limiting the impact of outliers 
        df['Volume_Spike_Z'] = z_raw.clip(lower=-4, upper=4)
        
        # Combining volume strength with the direction of price change
        df['Volume_Direction_Z'] = df['Volume_Spike_Z'] * np.sign(df['Return_1d_lagged_log'])
        
        return df



    features_market = calculate_market_features(df_p)
    features_ratio = calculate_indicators(df_final)

    features_ratio['Date'] = pd.to_datetime(features_ratio['Date'])
    features_market['Date'] = pd.to_datetime(features_market['Date'])

    # Merging all features into a single dataset
    features = pd.merge(
        features_ratio,
        features_market[['ticker','Date','Return_1d_lagged_log','Return_Overnight_log','Momentum_20d_log_z',
                        'Momentum_60d_log_z','Momentum_120d_log_z','Volatility_5d_ewm','Volatility_10d_ewm',
                        'Volatility_20d_ewm','Volume_Direction_Z']],
        on=['ticker','Date'],
        how='left'
    )

    features = features.sort_values(['ticker','Date']).reset_index(drop=True)

    # Feature definition
    base_columns = [
        'Date','ticker','Close','sector',
        'margin_gross','margin_operating','margin_net',
        'roa','roe',
        'expense_ratio_operating','expense_ratio_cogs','debt_to_equity', 'debt_to_assets',
        'cash_to_assets', 'cash_to_payables', 'asset_turnover', 'ppe_turnover', 'eps_basic',
        'bvps', 'sales_per_share',
        'Revenue Growth QoQ',
        'Log_Revenue','Log_Assets','Log_Revenue_delta','Log_Assets_delta','eps_basic','EPS_QoQ','EPS_Surprise'
    ]

    market_columns = [
        'ticker','Date','Return_1d_lagged_log','Return_Overnight_log','Momentum_20d_log_z',
        'Momentum_60d_log_z','Momentum_120d_log_z','Volatility_5d_ewm','Volatility_10d_ewm',
        'Volatility_20d_ewm','Volume_Direction_Z'
    ]


    ratios = [
            'margin_gross','margin_operating','margin_net',
            'roa','roe',
            'expense_ratio_operating','expense_ratio_cogs','debt_to_equity', 'debt_to_assets',
            'cash_to_assets', 'cash_to_payables', 'asset_turnover', 'ppe_turnover', 'eps_basic',
            'bvps', 'sales_per_share'
    ]

    # Automatic generation of column names for growth dynamics indicators
    delta_columns = [f'{w}_log_delta_QoQ' for w in ratios]
    sector_columns = [f'{w}_delta_vs_sector_z' for w in ratios]

    # Creating the final feature matrix for machine learning
    df_features = features[base_columns + market_columns + delta_columns + sector_columns]


    # Downloading and preparing VIX index data (market volatility measure)
    vix = yf.download('^VIX', start='2020-01-01', auto_adjust=True)['Close'].reset_index()
    vix.columns = ['Date', 'VIX_raw']
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)

    # Normalizing VIX using Z-Score
    vix['VIX_log'] = np.log(vix['VIX_raw'])
    vix['VIX_Zscore_60d'] = (
        vix['VIX_log'] - vix['VIX_log'].rolling(60, min_periods=20).mean()
    ) / vix['VIX_log'].rolling(60, min_periods=20).std()

    # Shifting VIX data by one session
    vix_ready = vix.shift(1)
    vix_ready['Date'] = vix['Date'] 

    # Integrating VIX data with the main feature matrix
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]
    df_features = df_features.merge(vix_ready[['Date', 'VIX_raw', 'VIX_Zscore_60d']], on='Date', how='left')

    # Calculating annualized volatility and its relationship to the overall level of market panic
    vol_ann = df_features['Volatility_20d_ewm'] * np.sqrt(252)
    df_features['Vol_vs_Market_Panic_log'] = np.log(vol_ann / (df_features['VIX_raw'] / 100))



    kluczowe_wskazniki = ['roa','roe']

    # Usunięcie rekordów bez kluczowych wskaźników rentowności (ROA i ROE)
    df_features = df_features[df_features[kluczowe_wskazniki].notnull().sum(axis=1) >= 2].reset_index(drop=True)

    # Usuwanięcie rekordów bez historycznej stopy zwrotu
    df_features = df_features.dropna(subset=['Return_1d_lagged_log']).copy()

    # Zapis przetworzonych cech do bazy danych SQL
    df_features.to_sql('features', engine, if_exists='replace', index=False)

if __name__ == "__main__":
    run_feature_engineering()