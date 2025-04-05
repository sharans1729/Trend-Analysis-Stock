import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Download data using yfinance
df = yf.download("RELIANCE.NS", start="2015-01-01", end="2024-01-01")
df = df[['Close']]
df.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Save the scaler
joblib.dump(scaler, "model/scaler.pkl")

# Prepare the data for LSTM
X, y = [], []
for i in range(60, len(df_scaled)):
    X.append(df_scaled[i - 60:i, 0])
    y.append(df_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Save the trained model
model.save("model/lstm_model.h5")
