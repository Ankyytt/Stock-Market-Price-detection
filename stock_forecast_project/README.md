# Stock Price Forecasting Project

This project implements time series forecasting for stock prices using ARIMA and LSTM models with full MLOps integration.

## Project Structure

- `data/` - Raw and processed data storage
- `models/` - Trained model files
- `notebooks/` - Jupyter notebooks for exploration and experimentation
- `mlruns/` - MLflow experiment logs
- `dags/` - Airflow DAGs for scheduled retraining
- `train.py` - Training script template
- `forecast.py` - Forecasting script (to be implemented)
- `requirements.txt` - Project dependencies
- `Dockerfile` - Docker environment setup

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Run the training script:

```bash
python train.py
```

## Notes on Virtual Environment

- The virtual environment folder (`venv/` or `venv/`) is excluded from the repository via `.gitignore`.
- Collaborators should create their own virtual environments and install dependencies using the `requirements.txt` file.
- This ensures consistent environments without sharing local virtual environment files.

## Next Steps

- Implement forecasting script
- Add MLflow experiment tracking
- Set up Airflow DAGs for scheduled retraining
- Develop deployment with FastAPI or Streamlit
