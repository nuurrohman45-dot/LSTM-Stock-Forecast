# LSTM Stock Forecast

A deep learning-based stock price forecasting application using LSTM with Temporal Attention mechanism. The system provides 7-day (up to 30-day) deterministic forecasts with walk-forward validation metrics.

## 📊 Overview

This project implements a machine learning pipeline for stock price prediction:

- **Model**: LSTM with Temporal Attention mechanism
- **Features**: Technical indicators (volatility, momentum, volume z-score)
- **Validation**: Walk-forward cross-validation
- **Interface**: Streamlit web application and Flask REST API

## 🏗️ Project Structure

```
.
├── streamlit_app.py           # Streamlit web application
├── app.py                     # Flask REST API
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose orchestration
├── fly.toml                   # Fly.io deployment config
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── src/                       # Source code package
│   ├── __init__.py
│   ├── config/
│   │   └── config.py          # Configuration settings
│   ├── models/
│   │   ├── lstm.py            # AttnLSTM model architecture
│   │   ├── save_load.py       # Model serialization utilities
│   │   └── train.py           # Training logic
│   ├── data/
│   │   └── loader.py          # Data loading functions
│   ├── features/
│   │   ├── features.py        # Feature engineering
│   │   └── regime.py          # Market regime detection
│   ├── backtest/
│   │   ├── metrics.py         # Performance metrics
│   │   └── walk_forward.py    # Walk-forward validation
│   ├── portfolio/
│   │   └── position.py        # Position sizing
│   ├── experiments/
│   │   └── train_final.py     # Final model training
│   ├── mlflow_utils/
│   │   └── tracking.py        # MLflow utilities
│   └── artifacts/
│       └── models/
│           └── lstm_v1.pt     # Trained model weights
│
├── notebooks/                 # Jupyter notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   └── model_experiment.ipynb # Model experiment notebook
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_app.py            # Unit tests
│
└── .github/
    └── workflows/
        ├── ci.yml             # GitHub Actions CI workflow
        └── deploy.yml         # GitHub Actions CD workflow (Fly.io)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- Fly.io CLI (for deployment)

### Installation

1. **Install dependencies**
   
```
bash
pip install -r requirements.txt
```

2. **Run the Streamlit app**
   
```
bash
streamlit run streamlit_app.py
```

3. **Open in browser**
   Navigate to `http://localhost:8501`

### Alternative: Run Flask API

```
bash
python app.py
```

Then access at `http://localhost:5000`

### Using Docker Compose

```
bash
# Build and start all services
docker-compose up --build

# Start only Streamlit app
docker-compose up streamlit

# Start only Flask API
docker-compose up flask-api
```

## 💻 Usage

### Using the Streamlit Web Interface

The Streamlit interface provides:

1. **Sidebar Settings**:
   - Enter any stock ticker (default: AAPL)
   - Select forecast horizon (1-30 days)
   - Choose historical data range (30-730 days)
   - View model information

2. **Main Dashboard**:
   - Interactive candlestick chart with 50/200-day moving averages
   - Current price with daily change metrics
   - 52-week high/low and volume information

3. **Forecast Section**:
   - Click "Generate Forecast" to get predictions
   - Interactive forecast chart with Plotly
   - Detailed forecast table with daily predictions and returns
   - Summary statistics (avg/max/min daily return, total return)

## 📦 Dependencies

```
numpy           # Numerical computing
pandas          # Data manipulation
matplotlib      # Visualization
yfinance        # Stock data retrieval
scikit-learn    # Machine learning utilities
torch           # Deep learning framework
mlflow          # Experiment tracking
streamlit       # Web interface
flask           # REST API
flask-cors      # CORS support
plotly          # Interactive charts
seaborn         # Statistical graphics
statsmodels     # Statistical models
pytest          # Testing app
```

## 🐳 Docker

### Build and Run

```
bash
# Build the image
docker build -t lstm-stock-forecast .

# Run Streamlit app
docker run -p 8501:8501 lstm-stock-forecast

# Run Flask API
docker run -p 5000:5000 -e PORT=5000 lstm-stock-forecast python app.py
```

### Using Docker Compose

```
bash
# Start all services
docker-compose up --build

# Start specific service
docker-compose up streamlit
docker-compose up flask-api
docker-compose up mlflow
```

Services:
- Streamlit: http://localhost:8501
- Flask API: http://localhost:5000
- MLflow: http://localhost:5000

## ☁️ Cloud Deployment (Fly.io)

### Prerequisites

1. Create a Fly.io account at https://fly.io
2. Install Fly CLI:
   
```
bash
   curl -L https://fly.io/install.sh | sh
   
```

### Deploy

```
bash
# Login to Fly.io
fly auth login

# Deploy the app
fly deploy
```

The app will be available at `https://lstm-stock-forecast.fly.dev`

## 🔄 CI/CD with GitHub Actions

### CI Pipeline (ci.yml)

The CI workflow runs on every push and pull request:
- Python 3.10 setup
- Code quality checks (flake8, black)
- Unit tests with pytest
- Docker image build verification

### CD Pipeline (deploy.yml)

The CD workflow deploys to Fly.io automatically:
- Runs on push to main branch
- Builds Docker image
- Deploys to Fly.io

### Setup GitHub Secrets

1. Go to GitHub Repository Settings
2. Navigate to Secrets and variables > Actions
3. Add a new secret:
   - Name: `FLY_API_TOKEN`
   - Value: Get from `fly auth token`

## 🔬 MLflow Tracking

### Start MLflow Server

```
bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost --port 5000
```

### Using MLflow in Code

```
python
from mlflow_utils.tracking import start_run, log_params, log_metrics

with start_run("training"):
    log_params({"epochs": 15, "batch_size": 32})
    log_metrics({"loss": 0.05, "accuracy": 0.95})
```

## 🔬 Methodology

1. **Data Collection**: Historical stock data via yfinance
2. **Feature Engineering**: Technical indicators computation
3. **Sequence Creation**: Sliding window approach (60-day sequences)
4. **Model Training**: LSTM with attention mechanism
5. **Walk-Forward Validation**: Rolling train/test validation
6. **Forecasting**: Deterministic multi-step ahead prediction

## 📊 Performance

The model is validated using walk-forward cross-validation to ensure:
- No look-ahead bias
- Realistic performance estimation
- Robust generalization

## 📝 Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.
Trading financial instruments involves risk and may result in financial loss.


