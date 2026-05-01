import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from config import engine

def run_backtest():
    pd.options.display.float_format = '{:.10f}'.format

    df_positions = pd.read_sql('all_positions', engine)

    # Timeframe configuration for model efficiency analysis
    df_positions['Date'] = pd.to_datetime(df_positions['Date'])
    df_positions['Date'] = df_positions['Date'].shift(-1)
    df_positions['exit_date'] = df_positions['Date'] + pd.Timedelta(days=9)

    cumulative_performance = 1.0
    active_samples = []
    equity_curve = []
    daily_exposures = []

    # Defining the timeline for the test
    dates = pd.date_range(
        start=df_positions['Date'].min(),
        end=df_positions['exit_date'].max(),
        freq='B'
    )

    final_date = dates[-1] 
    for current_date in dates:
        still_active = []
        for sample in active_samples:
            if sample['exit_date'] <= current_date:
                # Updating cumulative performance
                cumulative_performance *= (1 + sample['ret'] * sample['size'])
            else:
                still_active.append(sample)
                
        active_samples = still_active


        
        if current_date != final_date:
            new_signals = df_positions[df_positions['Date'] == current_date]
            
            for _, row in new_signals.iterrows():
                
                # Maximum exposure management
                max_exposure = 1.0
                current_exposure = sum(p['size'] for p in active_samples)
                available = max_exposure - current_exposure
                
                # Determining optimal weight for the new sample
                if available > 0:
                    pos_size = min(row['position_size'], available)
                    
                    active_samples.append({
                        'ret': row['ret'],
                        'exit_date': row['exit_date'],
                        'size': pos_size
                    })
                else:
                    continue

        # Recording daily exposure
        current_exposure = sum(p['size'] for p in active_samples)
        daily_exposures.append(current_exposure)
                    
        # Updating the model performance curve
        equity_curve.append(cumulative_performance)



    # Dataframe with model performance results
    df_performance = pd.DataFrame({
        'Date': dates,
        'model_output': equity_curve
    })



    df_performance.to_sql('model_x_score', engine, if_exists='replace', index=False)
    # Statistical analysis of model results
    df_performance['peak'] = df_performance['model_output'].cummax()
    df_performance['drawdown'] = df_performance['model_output'] / df_performance['peak'] - 1
    avg_exposure = np.mean(daily_exposures) * 100 
    df_performance['returns'] = df_performance['model_output'].pct_change().fillna(0)
    sharpe = (
        df_performance['returns'].mean() / df_performance['returns'].std()
    ) * np.sqrt(252)
    days = (df_performance['Date'].iloc[-1] - df_performance['Date'].iloc[0]).days
    years = days / 365
    cagr = df_performance['model_output'].iloc[-1] ** (1 / years) - 1
    max_dd = abs(df_performance['drawdown'].min())
    calmar = cagr / max_dd if max_dd != 0 else np.nan
    vol = df_performance['returns'].std() * np.sqrt(252)
    prediction_accuracy = (df_positions['ret'] > 0).mean()
    avg_sample_performance = df_positions['ret'].mean()

    # Final performance metrics of the predictive system
    print("\n====== SYSTEM EVALUATION SUMMARY ======")
    print(f"Final cumulative performance: {df_performance['model_output'].iloc[-1]:.4f}")
    print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Annualized Volatility: {vol:.2%}")
    print(f"Max drawdown: {df_performance['drawdown'].min():.2%}")
    print(f"Calmar Ratio: {calmar:.3f}")
    print(f"Signal classification accuracy: {prediction_accuracy:.2%}")
    print(f"Average return per single observation: {avg_sample_performance:.4f}")
    print(f"Average capital exposure during test period: {avg_exposure:.2f}%")


    # Statistical significance analysis of the results (t test)
    t_stat, p_value = stats.ttest_1samp(df_performance['returns'], 0)
    print(f"t-stat: {t_stat:.3f}, p-value: {p_value:.3f}")

    # Visualization of cumulative model performance over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_performance['Date'], df_performance['model_output'], label='Model Performance')
    plt.title('Model Cumulative Performance')
    plt.xlabel('Date')
    plt.ylabel('Unit Value')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Defining parameters for comparison with the market benchmark
    reference_ticker = "SPY"
    start_date = df_performance['Date'].min().strftime('%Y-%m-%d')
    end_date = df_performance['Date'].max().strftime('%Y-%m-%d')


    # Downloading benchmark data (SPY)
    reference_data = yf.download(reference_ticker, start=start_date, end=end_date, auto_adjust=True)

    # Data structure normalization (MultiIndex handling)
    if isinstance(reference_data.columns, pd.MultiIndex):
        reference_data.columns = reference_data.columns.get_level_values(0)

    # Calculating cumulative growth dynamics
    reference_data['returns'] = reference_data['Close'].pct_change().fillna(0)
    reference_data['reference_index'] = (1 + reference_data['returns']).cumprod()

    df_compare = pd.merge(df_performance[['Date', 'model_output']], 
                        reference_data[['reference_index']].reset_index(), 
                        on='Date', how='left').fillna(method='ffill')


    # Visualization of model performance compared to the market
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare['Date'], df_compare['model_output'], label='Predictive Algorithm (Model)', color='green', linewidth=2)
    plt.plot(df_compare['Date'], df_compare['reference_index'], label='SPY Benchmark', color='gray', linestyle='--', alpha=0.7)
    plt.title('Comparison of Model Performance vs. Base Index', fontsize=14)
    plt.xlabel('Analysis Period')
    plt.ylabel('Capital (Base = 1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    reference_returns = df_compare['reference_index'].pct_change().fillna(0)
    model_returns = df_compare['model_output'].pct_change().fillna(0)

    # Calculation of Beta coefficient and Alpha indicator
    beta = model_returns.cov(reference_returns) / reference_returns.var()
    alpha = (model_returns.mean() - beta * reference_returns.mean()) * 252

    print(f"\n====== PERFORMANCE ANALYSIS VS MARKET ({reference_ticker}) ======")
    print(f"Beta Coefficient: {beta:.3f}")
    print(f"Alpha Indicator (annualized): {alpha:.2%}")

if __name__ == "__main__":
    run_backtest()
