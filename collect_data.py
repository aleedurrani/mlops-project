import requests
import pandas as pd
from datetime import datetime, timedelta


def collect_weather_data():
    # Open-Meteo API endpoint for historical data
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    # Coordinates for New York City
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": (
            datetime.now() - timedelta(days=5)
        ).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                  "weather_code"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract hourly data
        hourly = data["hourly"]
        df = pd.DataFrame({
            "date_time": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "weather_code": hourly["weather_code"]
        })

        # Map weather codes to conditions (simplified)
        weather_map = {
            0: "Clear",
            1: "Partly Cloudy",
            2: "Cloudy",
            3: "Overcast",
            61: "Rain",
            95: "Thunderstorm",
        }
        df["weather_condition"] = df["weather_code"].map(weather_map).fillna("Unknown")

        # Save to CSV
        df.to_csv("data/raw_data.csv", index=False)
        print("Data saved to data/raw_data.csv")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    collect_weather_data()
