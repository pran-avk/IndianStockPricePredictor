import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import time

# --- 1. Setup Folders ---
# Ensure that the necessary 'data' and 'models' directories exist.
# os.makedirs with exist_ok=True is a safe way to create them.
print("Step 1: Setting up directories...")
for folder in ["data", "models"]:
    os.makedirs(folder, exist_ok=True)
print("Directories 'data' and 'models' are ready.\n")

# --- 2. Load Tickers ---
# Read the list of stock tickers from the 'tickers.txt' file.
# Each line in the file should contain one ticker symbol (e.g., WIPRO.NS).
print("Step 2: Loading tickers from tickers.txt...")
try:
    with open("tickers.txt") as f:
        # Read each line, strip whitespace, and filter out any empty lines.
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        print("Error: tickers.txt is empty or not found. Please add stock tickers to the file.")
        exit()
    print(f"Found {len(tickers)} tickers to process.\n")
except FileNotFoundError:
    print("Error: tickers.txt not found. Please create this file and add stock tickers.")
    exit()

# --- 3. Download or Update Stock Data Function ---
def update_data(ticker):
    """
    Downloads the latest data for a ticker. If data already exists,
    it only fetches new data since the last entry.
    """
    # Filenames are kept with the original ticker format (e.g., WIPRO.NS.parquet).
    file_path = f"data/{ticker}.parquet"

    # Default start date if no data exists.
    start_date = "2015-01-01"

    # If a data file already exists, find the last date and start downloading from the next day.
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path)
        if not df_existing.empty:
            last_date = df_existing.index[-1]
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        df_existing = pd.DataFrame()

    end_date = datetime.today().strftime("%Y-%m-%d")

    # Only download if the start date is before today.
    if start_date < end_date:
        print(f"  -> Downloading {ticker} data from {start_date} to {end_date}...")
        new_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    else:
        print(f"  -> {ticker} data is already up-to-date.")
        return df_existing

    if not new_data.empty:
        # Combine old and new data.
        df_combined = pd.concat([df_existing, new_data])
        # Remove any duplicate index entries, keeping the latest one.
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        # Save the updated data back to the Parquet file.
        df_combined.to_parquet(file_path)
        return df_combined

    return df_existing

# --- 4. Feature Engineering Function ---
def add_features(df):
    """
    Adds technical analysis features to the DataFrame that will be used for training.
    """
    df_featured = df.copy()
    df_featured["Return"] = df_featured["Close"].pct_change()
    df_featured["MA5"] = df_featured["Close"].rolling(window=5).mean()
    df_featured["MA10"] = df_featured["Close"].rolling(window=10).mean()

    # Define the target variables for prediction (the next day's prices).
    df_featured["Target_Close"] = df_featured["Close"].shift(-1)
    df_featured["Target_High"] = df_featured["High"].shift(-1)
    df_featured["Target_Low"] = df_featured["Low"].shift(-1)

    # Remove rows with NaN values created by the rolling means and shifts.
    df_featured = df_featured.dropna()
    return df_featured

# --- 5. Train and Save Model Function ---
def train_and_save_model(ticker, df):
    """
    Trains a RandomForestRegressor model and saves it to a file.
    """
    # Ensure there is enough data to train a meaningful model.
    if len(df) < 50:
        print(f"  -> Skipping {ticker}: Not enough data for training ({len(df)} rows).")
        return None

    # Define the features (X) and the target values (y).
    X = df[["Return", "MA5", "MA10"]]
    y = df[["Target_Close", "Target_High", "Target_Low"]]

    print(f"  -> Training model for {ticker}...")
    # Initialize and train the model.
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Save the trained model to the 'models' directory.
    model_file = f"models/{ticker}_model.joblib"
    dump(model, model_file)
    print(f"  -> Model for {ticker} trained and saved successfully.")
    return model


print("Step 3: Processing each ticker...")
for i, ticker in enumerate(tickers, 1):
    print(f"\nProcessing {i}/{len(tickers)}: {ticker}")
    try:
        # Update data, add features, and train the model.
        full_df = update_data(ticker)
        if full_df.empty:
            print(f"  -> No data found for {ticker}.")
            continue
        featured_df = add_features(full_df)
        train_and_save_model(ticker, featured_df)
        time.sleep(1) # Add a small delay to avoid hitting API rate limits.
    except Exception as e:
        print(f"  -> An error occurred while processing {ticker}: {e}")

print("\nPipeline finished.")
