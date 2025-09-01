import os
import pandas as pd

# -------------------------------
# 1. Get stock ticker input
# -------------------------------
ticker = input("Enter the stock ticker (e.g., BAJFINANCE.NS): ").strip()
file_path = f"data/{ticker}.parquet"

# -------------------------------
# 2. Check if file exists
# -------------------------------
if not os.path.exists(file_path):
    print(f"No data found for {ticker}. Please run the pipeline first.")
    exit()

# -------------------------------
# 3. Load data and get date range
# -------------------------------
df = pd.read_parquet(file_path)

if df.empty:
    print(f"No data in {file_path}")
else:
    start_date = df.index.min()
    end_date = df.index.max()
    total_days = len(df)
    print(f"{ticker} data range: {start_date.date()} to {end_date.date()}")
    print(f"Total trading days downloaded: {total_days}")
