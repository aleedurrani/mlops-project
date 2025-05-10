import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model():
    # Load preprocessed data
    df = pd.read_csv("data/processed_data.csv")

    # Features and target
    X = df[["humidity", "wind_speed"]]
    y = df["temperature"]

    # Apply normalization if needed
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define model hyperparameters
    params = {
        "fit_intercept": True
    }

    # Train model
    model = LinearRegression(**params)
    model.fit(X_scaled, y)

    # Calculate metrics (MSE on training data)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")

    # Create or set MLflow experiment
    mlflow.set_experiment("Weather Prediction")

    # Start an MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("mse", mse)

        # Set a tag for context
        mlflow.set_tag("Training Info", "Linear Regression model for temperature prediction using weather data")

        # Infer model signature
        signature = infer_signature(X_scaled, model.predict(X_scaled))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="weather_model",
            signature=signature,
            input_example=X_scaled[:1],  # Log one row as an example
            registered_model_name="weather-prediction-model"
        )

    # Save model and scaler locally
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model saved to models/model.pkl")
    print("Scaler saved to models/scaler.pkl")
    print(f"MLflow run completed. Model logged to {model_info.artifact_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model()