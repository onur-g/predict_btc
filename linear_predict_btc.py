import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import requests

# Use requests to download the CSV file from the internet
# url = "https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_d.csv"
# response = requests.get(url)

# Use pandas to read the downloaded data into a DataFrame
# df = pd.read_csv(response.content)

import os
df = pd.read_csv(os.path.abspath("Gemini_BTCUSD_d.csv"))

# Use the "Close" column as the target variable
y = df["close"]

# Use the "Open" and "Volume" columns as the input features
X = df[["open", "Volume BTC"]]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a linear regression model on the training data
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the testing data
predictions = reg.predict(X_test)

# Plot the actual and predicted values on a graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()