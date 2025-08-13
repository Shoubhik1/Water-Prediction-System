import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
# Load dataset
sheet_id = "12fheKw5AJ1n1tAYEocKu5lvfWUkzWm8laDX7PiqE4hk"
sheet_name = "Data"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)  # Replace with your dataset

# Selecting features and target variables
features = ["Temperature", "PPM", "Conductivity", "Turbidity"]
target_features = ['Temperature', 'PPM', 'Conductivity', 'Turbidity']
df_scaled = MinMaxScaler().fit_transform(df[features])

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df_scaled, test_size=0.2, random_state=42)

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(1, len(features))),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(len(features))  # Output layer with multiple target variables
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)


# Save model and scaler
model.save("lstm_model.h5")

# Predictions
y_pred = model.predict(X_test)

# Reverse scaling
y_test_inv = MinMaxScaler().fit(df[features]).inverse_transform(y_test)
y_pred_inv = MinMaxScaler().fit(df[features]).inverse_transform(y_pred)

# Prediction function
def predict_values(input_data):
    input_scaled = MinMaxScaler().fit(df[features]).transform(np.array(input_data).reshape(1, -1))
    input_reshaped = input_scaled.reshape((1, 1, len(features)))
    pred_scaled = model.predict(input_reshaped)
    pred_inv = MinMaxScaler().fit(df[features]).inverse_transform(pred_scaled)
    return {features[i]: pred_inv[0][i] for i in range(len(features))}

# User input for number of years
num_years = int(input("Enter the number of years for prediction: "))
predictions = []
current_data = df[features].iloc[-1].values

for year in range(1, num_years + 1):
    future_prediction = predict_values(current_data)
    predictions.append(future_prediction)
    current_data = np.array(list(future_prediction.values()))

# Print predictions for all years with drinkability
print("\nFuture Predictions for Each Year:")
print("Year  Temperature (°C)  PPM   Conductivity (EC)  Turbidity (NTU)  Drinkable")
print("-" * 75)

for i, prediction in enumerate(predictions, 1):
    drinkable = 1 if (prediction["PPM"] <= 300 and prediction["Conductivity"] <= 0.5 and prediction["Turbidity"] <= 5) else 0
    print(f"{i:<5}  {prediction['Temperature']:.2f}          {prediction['PPM']:.1f}      "
          f"{prediction['Conductivity']:.2f}              {prediction['Turbidity']:.2f}             {drinkable}")


# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
titles = ["Temperature", "PPM", "Conductivity", "Turbidity"]


for i, ax in enumerate(axes.flatten()):
    sorted_idx = np.argsort(y_test_inv[:, i])
    ax.scatter(range(len(y_test_inv)), y_test_inv[sorted_idx, i], color="blue", label="Actual", alpha=0.6)
    ax.plot(y_pred_inv[sorted_idx, i], color="red", linestyle="dashed", label="Predicted", linewidth=2)
    ax.set_title(f"Actual vs Predicted {titles[i]}")
    ax.set_xlabel("Data Points (Sorted)")
    ax.set_ylabel(titles[i])
    ax.legend()

plt.tight_layout()
plt.show()
prediction_df = pd.DataFrame(predictions)
# Plot individual graphs for each metric
plt.figure(figsize=(14, 8))
for i, col in enumerate(target_features, 1):
    plt.subplot(2, 2, i)
    plt.plot(prediction_df.index, prediction_df[col], marker='o', linestyle='-', label=f"{col} Prediction")  
    plt.xlabel("Years Ahead")
    plt.ylabel(f"{col} Value")
    plt.title(f"Predicted {col} Trend")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()
# Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
