# Weather Data Pipeline

## Setup
1. Install Anaconda, Docker Desktop, and Git.
2. Clone the repository: `git clone <repo-url>`
3. Create Anaconda environment: `conda create -n weather_pipeline python=3.9`
4. Install dependencies: `pip install -r requirements.txt`
5. Initialize DVC: `dvc init`
6. Set DVC storage: `dvc remote add -d local_storage D:\dvc_storage`
7. Run Airflow: `docker-compose up`
8. Run DVC pipeline: `dvc repro`

## Pipeline
- **Data Collection**: Fetches weather data from Open-Meteo API.
- **Preprocessing**: Handles missing values and normalizes data.
- **Model Training**: Trains a linear regression model.
- **Airflow**: Automates data collection and preprocessing.
- **DVC**: Versions data and model, with storage on D drive.