import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data
import os
df = pd.read_csv(os.path.abspath("Gemini_BTCUSD_d.csv"))

# Split the data into a training set and a test set
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make a prediction for a given date
date = input('Enter a date in YYYY-MM-DD format: ')
prediction_input = pd.DataFrame([date], columns=['Date'])
prediction = model.predict(prediction_input)[0]

# Print the prediction
print(f'The predicted price of Bitcoin on {date} is ${prediction:.2f}')