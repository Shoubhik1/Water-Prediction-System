from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import io
import base64
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
sheet_id = "12fheKw5AJ1n1tAYEocKu5lvfWUkzWm8laDX7PiqE4hk"
sheet_name = "Data"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

features = ["Temperature", "PPM", "Conductivity", "Turbidity"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Load pre-trained model
model = load_model(r"C:\Users\SHOUBHIK-PC\OneDrive\Desktop\cc&iot_Final\cc&iot_Final\lstm_model.h5",custom_objects={"mse": MeanSquaredError()})

# WHO-like standards
standards = {
    "Drinking": {"PPM": 300, "Conductivity": 0.5, "Turbidity": 5},
    "Aquaculture": {"PPM": 500, "Conductivity": 2.0, "Turbidity": 25},
    "Agriculture": {"PPM": 1000, "Conductivity": 3.0, "Turbidity": 50},
    "Industrial": {"PPM": 1500, "Conductivity": 5.0, "Turbidity": 100}
}

# Prediction function
def predict_values(input_data):
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    input_reshaped = input_scaled.reshape((1, 1, len(features)))
    pred_scaled = model.predict(input_reshaped)
    pred_inv = scaler.inverse_transform(pred_scaled)
    return {features[i]: pred_inv[0][i] for i in range(len(features))}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        num_years = int(request.form['num_years'])
        predictions = []
        current_data = df[features].iloc[-1].values

        for year in range(1, num_years + 1):
            future_prediction = predict_values(current_data)
            predictions.append(future_prediction)
            current_data = np.array(list(future_prediction.values()))

        pred_df = pd.DataFrame(predictions)

        # Generate prediction plots
        img = io.BytesIO()
        plt.figure(figsize=(14, 8))
        for i, col in enumerate(features, 1):
            plt.subplot(2, 2, i)
            plt.plot(range(1, num_years + 1), pred_df[col], marker='o', linestyle='-', label=f"{col} Prediction")
            plt.xlabel("Years Ahead")
            plt.ylabel(f"{col} Value")
            plt.title(f"Predicted {col} Trend")
            plt.grid()
            plt.legend()
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Feature Correlation Heatmap
        heatmap_img = io.BytesIO()
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(heatmap_img, format='png')
        plt.close()
        heatmap_img.seek(0)
        heatmap_url = base64.b64encode(heatmap_img.getvalue()).decode('utf8')

        # Actual vs Predicted Scatter Plot
        actual_vs_predicted_img = io.BytesIO()
        y_test_inv = scaler.inverse_transform(df_scaled)
        y_pred_inv = scaler.inverse_transform(model.predict(df_scaled.reshape(df_scaled.shape[0], 1, df_scaled.shape[1])))

        plt.figure(figsize=(14, 8))
        for i, col in enumerate(features):
            plt.subplot(2, 2, i + 1)
            sorted_idx = np.argsort(y_test_inv[:, i])
            plt.scatter(range(len(y_test_inv)), y_test_inv[sorted_idx, i], color="blue", label="Actual", alpha=0.6)
            plt.plot(y_pred_inv[sorted_idx, i], color="red", linestyle="dashed", label="Predicted", linewidth=2)
            plt.xlabel("Data Points (Sorted)")
            plt.ylabel(col)
            plt.title(f"Actual vs Predicted {col}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(actual_vs_predicted_img, format='png')
        plt.close()
        actual_vs_predicted_img.seek(0)
        actual_vs_predicted_url = base64.b64encode(actual_vs_predicted_img.getvalue()).decode('utf8')

        return render_template('prediction.html', predictions=predictions, num_years=num_years, plot_url=plot_url, heatmap_url=heatmap_url, actual_vs_predicted_url=actual_vs_predicted_url)

    return render_template('prediction.html')

@app.route('/suitability', methods=['GET', 'POST'])
def suitability():
    if request.method == 'POST':
        num_years = int(request.form['num_years'])
        purpose_choice = int(request.form['purpose_choice'])
        purpose = ["Drinking", "Aquaculture", "Agriculture", "Industrial"][purpose_choice - 1]
        limits = standards[purpose]

        predictions = []
        current_data = df[features].iloc[-1].values

        for year in range(1, num_years + 1):
            future_prediction = predict_values(current_data)
            predictions.append(future_prediction)
            current_data = np.array(list(future_prediction.values()))

        prediction_df = pd.DataFrame(predictions)

        suitability_results = []
        for prediction in predictions:
            suitable = "YES" if (prediction["PPM"] <= limits["PPM"] and
                                 prediction["Conductivity"] <= limits["Conductivity"] and
                                 prediction["Turbidity"] <= limits["Turbidity"]) else "NO"
            suitability_results.append(suitable)

        return render_template('suitability.html', predictions=predictions, suitability_results=suitability_results, num_years=num_years, purpose=purpose)

    return render_template('suitability.html')

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('contactus.html')

if __name__ == "__main__":
    app.run(debug=True)
