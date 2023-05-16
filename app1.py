import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import keras.backend as K
import streamlit as st

# Load the data
df = pd.read_csv('data.csv')

# Set the date as the index
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Create a plot of the data
plt.plot(df["Price"])
plt.title("Stock price over time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Prepare the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Price"].to_numpy().reshape(-1, 1))

n_lookback = 60

# Register the custom loss function
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

custom_objects = {"root_mean_squared_error": root_mean_squared_error}

# Create the Streamlit app
st.set_page_config(page_title="Stock Price Prediction")
st.title("Stock Price Prediction App")

# Load the pre-trained LSTM model
model = load_model("model1.h5", custom_objects=custom_objects)

# User input form
st.sidebar.title("User Input")
start_date = st.sidebar.date_input("Start date", min_value=df.index[0])
end_date_default = df.index[-1]  # Set default to the latest available date
end_date_max = df.index[-1] + pd.DateOffset(days=1)  # Set the max value to the last date in the dataset plus one day
end_date = st.sidebar.date_input("End date", max_value=end_date_max, value=end_date_default)
n_days = st.sidebar.number_input("Number of days to predict", value=30, min_value=1, max_value=365)

if start_date >= end_date:
    st.sidebar.error("Error: End date must be after start date")

else:
    st.sidebar.success("Dates are valid")

    # Select the relevant data
    df_pred = df[start_date:end_date]
    last_date = df_pred.index[-1]
    next_date = last_date + pd.DateOffset(days=1)

    # Make predictions
    prediction = []
    for i in range(n_days):
        x_input = scaled_data[-n_lookback:].reshape(1, -1, 1)
        y_pred = model.predict(x_input)
        prediction.append(scaler.inverse_transform(y_pred)[0, 0])
        scaled_data = np.vstack((scaled_data, y_pred))

    # Create a DataFrame of the predicted prices
    pred_index = pd.date_range(start=next_date, periods=n_days, freq="D")
    df_pred = pd.concat([df_pred, pd.DataFrame(data=prediction, index=pred_index, columns=["Predicted Price"])])

    # Display the predictions
    st.subheader(f"Predictions for the next {n_days} days:")
    st.line_chart(df_pred["Predicted Price"])

    # Display a table of the predicted prices
    st.subheader("Predicted Prices")
    st.dataframe(df_pred["Predicted Price"])