from binance_data_scraper import download_binance_candles_daterange
from datetime import datetime, timedelta

# Calculate date ranges:
# End date is today
end_date = datetime.now()

# Start date is 2.5 years ago
start_date = end_date - timedelta(days=365*2.5)  # 2.5 years

print(f"Downloading historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Total duration: {(end_date - start_date).days} days")
print(f"Training period: {start_date.strftime('%Y-%m-%d')} to {(end_date - timedelta(days=180)).strftime('%Y-%m-%d')}")
print(f"Backtest period: {(end_date - timedelta(days=180)).strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

download_binance_candles_daterange(
    pair="BTCUSDT",
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    granularity="1h",
    save_directory="./data"
)