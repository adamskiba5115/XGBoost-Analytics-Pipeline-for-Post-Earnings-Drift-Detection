import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import numpy as np
import os
import time
import simfin
from config import engine


def run_data_ingestion():
    pd.options.display.float_format = '{:.10f}'.format

    # Fetching the current list of US market tickers
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    tickers = []

    try:
        tickers = pd.read_csv(url, header=None)[0].dropna().tolist()
        
        # Standardizing ticker format
        tickers = [t.replace(".", "-") for t in tickers]
        
        # Removing warrants (WS, WT), rights (R), and units (U)
        exclude_suffixes = ('W', 'R', 'U', 'WS', 'WT')
        tickers = [t for t in tickers if not t.endswith(exclude_suffixes)]
        
    except Exception as e:
        print(f"Error while fetching ticker list: {e}")


    prices_list, balance_list, income_statement_list, earnings_list = [], [], [], []

    print(f"Fetching data for {len(tickers)} tickers")

    for ticker in tickers:
        try:
            tck = yf.Ticker(ticker)
            
            # Fetching fundamental data (quarterly reports)
            balance = tck.quarterly_balance_sheet.T
            income_statement = tck.quarterly_income_stmt.T
            
            # Fetching sector data
            sector = tck.info.get('sector', 'Unknown')

            # Fundamental data aggregation logic
            if not balance.empty:
                balance['ticker'] = ticker
                balance['sector'] = sector
                balance_list.append(balance.reset_index())
                
            if not income_statement.empty:
                income_statement['ticker'] = ticker
                income_statement['sector'] = sector
                income_statement_list.append(income_statement.reset_index())

            # Fetching historical quotes
            prices = tck.history(start='2022-01-01')
            if not prices.empty:
                prices['ticker'] = ticker
                prices_list.append(prices.reset_index())
                
                # Fetching earnings publication dates (Earnings Dates)
                earnings = tck.get_earnings_dates(limit=20)
                if earnings is not None and not earnings.empty:
                    earnings['ticker'] = ticker
                    earnings_list.append(earnings.reset_index())

            
            time.sleep(0.1)
            
        except Exception as e:
            
            print(f"Skipping ticker {ticker} due to error: {e}")



    if prices_list:
        all_prices = pd.concat(prices_list, ignore_index=True)
    if balance_list:
        all_balance = pd.concat(balance_list, ignore_index=True)
    if income_statement_list:
        all_income_statement = pd.concat(income_statement_list, ignore_index=True)
    if earnings_list:
        all_earnings = pd.concat(earnings_list, ignore_index=True)


    df_all_prices = pd.DataFrame(all_prices)
    df_all_balance = pd.DataFrame(all_balance)
    df_all_income = pd.DataFrame(all_income_statement)
    df_all_earnings = pd.DataFrame(all_earnings)

    # Removing columns where missing data exceeds 40%
    null_map_balance = df_all_balance.isnull().mean()
    df_balance_clean = df_all_balance.loc[:, null_map_balance < 0.4]
    null_map_income = df_all_income.isnull().mean()
    df_income_clean = df_all_income.loc[:, null_map_income < 0.4]

    # Exporting data to SQLite database
    df_balance_clean.to_sql('balance_sheets', engine, if_exists='replace', index=False)
    df_all_prices.to_sql('all_prices', engine, if_exists='replace', index=False)
    df_income_clean.to_sql('income_statements', engine, if_exists='replace', index=False)
    df_all_earnings.to_sql('earnings_clean', engine, if_exists='replace', index=False)


    # Fetching data from SQL database
    query_balance = "SELECT * FROM balance_sheets"
    query_income    = "SELECT * FROM income_statements" 
    query_prices    = "SELECT ticker, Date, Open, Close, Volume FROM all_prices ORDER BY ticker, Date"
    query_earnings = """SELECT ticker, "Earnings Date", "EPS Estimate", "Reported EPS" FROM earnings_clean"""
    df_b = pd.read_sql(query_balance, engine)
    df_i = pd.read_sql(query_income, engine)
    df_p = pd.read_sql(query_prices, engine)
    df_e = pd.read_sql(query_earnings, engine)
    df_raports = pd.read_sql('combined_database',engine)

    # Standardizing date format
    df_raports = df_raports.rename(columns={'index': 'Period_End'})
    df_raports['Period_End'] = pd.to_datetime(df_raports['Period_End']).dt.tz_localize(None)


    # Removing the 'Earnings Date' column, which is located in a separate table
    if 'Earnings Date' in df_raports.columns:
        df_raports = df_raports.drop(columns=['Earnings Date'])


    # Standardizing earnings publication dates
    df_e['Earnings Date'] = pd.to_datetime(df_e['Earnings Date'], utc=True).dt.tz_localize(None)


    # Session time adjustment: reports published after 4:00 PM are available to the market from the following day
    market_close_hour = 16
    df_e['Date'] = pd.to_datetime(df_e['Earnings Date'].dt.date)
    df_e['Date'] = pd.to_datetime(df_e['Date'])
    after_close = df_e['Earnings Date'].dt.hour >= market_close_hour
    df_e.loc[after_close, 'Date'] += pd.Timedelta(days=1)


    # Accounting for non-working days (shifting weekend publications to Monday)
    df_e.loc[df_e['Date'].dt.weekday == 5, 'Date'] += pd.Timedelta(days=2)
    df_e.loc[df_e['Date'].dt.weekday == 6, 'Date'] += pd.Timedelta(days=1)


    df_e = df_e.sort_values(['ticker', 'Date']).reset_index(drop=True)
    df_e['Date'] = pd.to_datetime(df_e['Date']).dt.tz_localize(None)


    # Sorting data before merge_asof
    df_raports = df_raports.sort_values(['Period_End', 'ticker']).reset_index(drop=True)
    df_e = df_e.sort_values(['Date', 'ticker']).reset_index(drop=True)


    # Aligning reporting periods with the nearest earnings publication dates
    df_raports = pd.merge_asof(
        df_raports,
        df_e[['ticker', 'Date', 'Earnings Date', 'EPS Estimate', 'Reported EPS']],
        left_on='Period_End',
        right_on='Date',
        by='ticker',
        direction='forward',
        tolerance=pd.Timedelta(days=180)
    )



    # Removing records without a matching publication date
    df_raports = df_raports.dropna(subset=['Date', 'Earnings Date']).copy()


    # Filtering logic errors
    df_raports = df_raports[
        df_raports['Period_End'] < df_raports['Earnings Date']
    ].copy()


    # Limiting maximum report publication delay to 180 days
    df_raports = df_raports[
        (df_raports['Earnings Date'] - df_raports['Period_End']).dt.days <= 180
    ].copy()


    # Standardizing date formats and cleaning stock tickers
    df_p['Date'] = pd.to_datetime(df_p['Date']).dt.tz_localize(None)
    df_p['ticker'] = df_p['ticker'].astype(str).str.strip()


    # Sorting datasets before final data merging
    df_raports = df_raports.sort_values(['Date', 'ticker']).reset_index(drop=True)
    df_p = df_p.sort_values(['Date', 'ticker']).reset_index(drop=True)


    # Mapping the last known market price to the report publication time
    df_final = pd.merge_asof(
        df_raports,
        df_p[['ticker', 'Date', 'Close', 'Volume', 'Open']],
        on='Date',
        by='ticker',
        direction='backward'
    )


    # Removing records with missing revenue data
    df_final = df_final[df_final['revenue'].notna()].reset_index(drop=True)


    # Eliminating [Date, Ticker] duplicates
    df_final = df_final.drop_duplicates()
    df_final = df_final.drop_duplicates(subset=['Date', 'ticker'])

    df_final = df_final.reset_index(drop=True)
    df_final.to_sql('df_final', engine, if_exists='replace', index=False)

if __name__ == "__main__":
    run_data_ingestion()






