from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Paths to the model and scaler
MODEL_PATH = "model/lstm_model.h5"
SCALER_PATH = "model/scaler.pkl"

# Check if model and scaler exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file is missing. Please check the 'model/' folder.")

# Load LSTM model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def prepare_input(data, time_steps=60):
    """Prepare the last `time_steps` of stock prices for the LSTM model."""
    data = np.array(data).reshape(-1, 1)
    if data.shape[0] < time_steps:
        raise ValueError(f"Not enough data: need at least {time_steps} prices.")
    data_scaled = scaler.transform(data)
    return np.array([data_scaled[-time_steps:]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get ticker from form input
        ticker = request.form['ticker'].strip().upper()

        # Download stock data from yfinance
        df = yf.download(ticker, period='1y')
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Invalid stock ticker or no closing price data available.")

        # Extract closing prices
        past_prices = df['Close'].dropna().values.tolist()

        # Prepare input and make prediction
        input_data = prepare_input(past_prices)
        prediction_scaled = model.predict(input_data)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        # Plot recent prices and predicted price
        plt.figure(figsize=(10, 4))
        recent_days = past_prices[-30:]
        plt.plot(range(len(recent_days)), recent_days, label='Past 30 Days')
        plt.plot(len(recent_days), prediction, 'ro', label='Predicted Next Day Price')
        plt.title(f"{ticker} - Stock Price Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot to static folder
        plot_path = os.path.join('static', 'future_plot.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('predict.html', final_price=round(prediction, 2))

    except ValueError as ve:
        return render_template('predict.html', final_price=f"Input Error: {str(ve)}")
    except Exception as e:
        return render_template('predict.html', final_price=f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
