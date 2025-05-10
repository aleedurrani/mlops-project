import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw_data.csv")

    # Handle missing values
    df = df.fillna({
        "temperature": df["temperature"].mean(),
        "humidity": df["humidity"].mean(),
        "wind_speed": df["wind_speed"].mean(),
        "weather_condition": "Unknown"
    })

    # Normalize numerical fields
    scaler = StandardScaler()
    numerical_cols = ["temperature", "humidity", "wind_speed"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Save preprocessed data
    df.to_csv("data/processed_data.csv", index=False)
    print("Preprocessed data saved to data/processed_data.csv")

if __name__ == "__main__":
    preprocess_data()