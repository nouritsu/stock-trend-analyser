import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = "2010-01-01"
end = "2019-12-31"

st.title = "Stock Trend Predictor"
usr_inp = st.text_input("Enter Stock Ticker", "AAPL")
df = data.DataReader(usr_inp, "yahoo", start, end)

# Describing Data
st.subheader(f"{usr_inp} Stocks from 2010-2020")
st.write(df.describe())

# Visualizations
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Plot Moving Average
st.subheader("Moving Average vs Time Chart")
st.text("Green > Moving Average (200 Days)\nRed > Moving Average (100 Days)")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, "r")
plt.plot(ma200, "g")
st.pyplot(fig)

# Scale Data
idx = int(len(df) * 0.7)  # Marks 70% point in dataframe
training = pd.DataFrame(df["Close"][0:idx])
testing = pd.DataFrame(df["Close"][idx:])

s = MinMaxScaler(feature_range=(0, 1))
training_arr = s.fit_transform(training)

# Split Training Data
x_train, y_train = [], []
for i in range(100, training_arr.shape[0]):  # 100 Days
    x_train.append(training_arr[i - 100 : i])  # Makes days zero indexed
    y_train.append(training_arr[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Load ML Model
model = load_model("./keras_model.h5")

# Testing Model
past100_days = training.tail(100)
finaldf = past100_days.append(testing, ignore_index=True)

inpdata = s.fit_transform(finaldf)

# Split Testing Data
x_test, y_test = [], []
idx = inpdata.shape[0]
for i in range(100, idx):
    x_test.append(inpdata[i - 100 : i])
    y_test.append(inpdata[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict Data
y_pred = model.predict(x_test)

s_factor = 1 / s.scale_
y_pred *= s_factor
y_test *= s_factor

# Final Graph
st.subheader("Predictions vs Original Price")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_pred, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
