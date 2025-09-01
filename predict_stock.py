import os
import pandas as pd
from joblib import load
from datetime import datetime, timedelta

# --- 1. Get User Input ---
# Prompt the user to enter the stock ticker they want a prediction for.
# .strip() removes any accidental leading/trailing whitespace, and .upper() standardizes the input.
ticker = input("Enter the stock ticker (e.g., WIPRO.NS): ").strip().upper()

# --- 2. Define File Paths ---
# This pathing logic directly matches the file-saving logic in your pipeline.py.
# It uses the ticker exactly as entered to find the correct files.
data_path = f"data/{ticker}.parquet"
model_path = f"models/{ticker}_model.joblib"

# --- 3. Validate File Existence ---
# Check if both the required data file and the model file exist before proceeding.
# This is a critical guard clause to prevent the script from failing later.
if not os.path.exists(data_path) or not os.path.exists(model_path):
    print(f"\nError: Data or model not found for ticker '{ticker}'.")
    print(f"Searching for data at: '{data_path}'")
    print(f"Searching for model at: '{model_path}'")
    print("\nPlease ensure you have run 'pipeline.py' for this ticker first.")
    # Exit the script cleanly if files are missing.
    exit()

try:
    # --- 4. Load Data and Model ---
    # Load the historical stock data from the Parquet file.
    df = pd.read_parquet(data_path)
    # Load the pre-trained machine learning model from the joblib file.
    model = load(model_path)

    # --- 5. Feature Engineering ---
    # To make a valid prediction, we must create the exact same features
    # that the model was trained on. This section mirrors the feature
    # engineering step in your pipeline.py script.
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    # Drop rows with NaN values that were created by the feature calculations.
    df = df.dropna()

    # --- 6. Prepare Latest Data for Prediction ---
    # Isolate the most recent row of data and select only the feature columns.
    # model.predict() expects a DataFrame-like input, so we use iloc[-1:].
    latest_features = df.iloc[-1:][["Return", "MA5", "MA10"]]

    # --- 7. Make Prediction ---
    # Use the loaded model to predict the next day's price points.
    # The model will output an array with [Close, High, Low].
    predicted_values = model.predict(latest_features)
    pred_close, pred_high, pred_low = predicted_values[0]

    # --- 8. Determine Trading Signal & Get Last Values ---
    # Extract the last known actual values for comparison and display.
    # We explicitly cast to float() to ensure we have single numbers.
    last_trade_date = df.index[-1]
    last_close = float(df['Close'].iloc[-1])
    last_high = float(df['High'].iloc[-1])
    
    # Define the percentage change that is considered neutral (e.g., 0.5%)
    neutral_threshold = 0.005
    percentage_change = (pred_close - last_close) / last_close

    if percentage_change > neutral_threshold:
        signal = "Bullish"
    elif percentage_change < -neutral_threshold:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # --- 9. Display Results ---
    # Print the previous day's data and the forecast in a clear format.
    tomorrow_date = datetime.today() + timedelta(days=1)
    print("\n" + "="*50)
    print(f"Forecast for {ticker} on {tomorrow_date.strftime('%Y-%m-%d')}")
    print("="*50)
    print("Previous Day's Data:")
    print(f"  - Date:  {last_trade_date.strftime('%Y-%m-%d')}")
    print(f"  - High:  {last_high:.2f}")
    print(f"  - Close: {last_close:.2f}")
    print("-" * 50)
    print("Predicted Next Day's Data:")
    print(f"  - High:  {pred_high:.2f}")
    print(f"  - Low:   {pred_low:.2f}")
    print(f"  - Close: {pred_close:.2f}")
    print(f"\n  Signal: {signal}")
    print("="*50)


except Exception as e:
    print(f"\nAn error occurred during the prediction process: {e}")
    print("Please check that the data and model files are not corrupted.")

