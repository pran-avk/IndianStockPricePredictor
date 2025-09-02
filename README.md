📈 Indian Stock Price Predictor

A machine learning project that predicts future stock prices for Indian companies listed on NSE & BSE. The system automatically downloads historical stock data, trains predictive models, and generates daily forecasts.

🔹 Features

Fetches real-time stock data using Yahoo Finance API (yfinance)

Trains Random Forest Regressor models for each stock

Stores data, trained models, and predictions in organized folders (data/, models/, predictions/)

Predicts next-day maximum stock price movement

Scalable to run for multiple tickers daily

Easy-to-use pipeline (pipelines.py) to automate workflow

🔹 Tech Stack

Python

Pandas, NumPy for data handling

Scikit-learn for ML models

Joblib for model persistence

yfinance for stock market data

🔹 Folder Structure
📂 Indian-Stock-Predictor
 ┣ 📂 data/  [csv format]
 ┣ 📂 models/   [processed model]
 ┣ 📂 predictions/  
 ┣ 📜 pipelines.py  [model]
 ┣ 📜predict_stock.py [predictor model]
 ┣ 📜tickers.txt [Stockname to be used]
 ┗ 📜 README.md     

🔹 How It Works

Collects 1–5 years of historical stock data

Prepares training datasets with technical indicators

Trains a model per stock & saves it in /models

Generates next-day predictions and saves in /predictions

Can be scheduled to run once daily with CRON/Task Scheduler




# #USAGE
Store all the Stock name in tickers.txt to be processed
run the pipelines.py file and wait until all the files get processed
run the predict_stock.py and give the name of the stock to be predicted
**Warning: This Model need to be Updated daily before running the predict_stock.py**
**running pipelines.py Updates the Model**

## Hope you like it 😊
