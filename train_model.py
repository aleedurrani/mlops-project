import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Load preprocessed data
    df = pd.read_csv("data/processed_data.csv")

    # Features and target
    X = df[["humidity", "wind_speed"]]
    y = df["temperature"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_model()