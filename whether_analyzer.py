import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
data = pd.read_csv("weather.csv")

# Features and target
X = data[["day"]]
y = data["temperature"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict future temperature (next 5 days)
future_days = np.array([[21], [22], [23], [24], [25]])
predictions = model.predict(future_days)

print("Predicted Temperatures for Next 5 Days:")
for i, temp in enumerate(predictions, start=21):
    print(f"Day {i}: {temp:.2f} °C")

# Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Weather Temperature Prediction")
plt.legend()
plt.show()