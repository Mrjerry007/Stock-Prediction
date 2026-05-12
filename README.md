# Stock Prediction!

# AI Trading Dashboard

A full-stack Python project with:
- FastAPI backend
- Live market data from `yfinance`
- Candlestick charts with Plotly
- Technical indicators (SMA, EMA, RSI, MACD, volatility)
- XGBoost-based next-day close prediction
- Prediction intervals
- Alert generation
- Optional LSTM training script

## Project structure

```text
backend/
  app/
    main.py
    ml/
    services/
    utils/
  requirements.txt

frontend/
  index.html
  style.css
  app.js
```

## Setup

```bash
cd backend
python -m venv venv
venv\\Scripts\\activate   # Windows
# source venv/bin/activate # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `frontend/index.html` in a browser, or serve it with any static server.

## Train the XGBoost model

```bash
cd backend
python -m app.ml.train_xgboost --ticker AAPL --period 5y
```

The model is saved as `backend/app/ml/models/xgboost_model.joblib`.

## Train the optional LSTM model

```bash
cd backend
python -m app.ml.train_lstm --ticker AAPL --period 5y
```

## Final STEP — To Run Full Dashboard

```bash
cd backend
uvicorn app.main:app --reload

Keep this terminal OPEN.
```

```bash
cd frontend/index.html

Run index.html live server in VScode
```

## Notes

- The project uses a simple file-based model store.
- If no trained model exists, the API falls back to a lightweight heuristic forecast so the dashboard still works.
- For production, add authentication, persistent storage, scheduled retraining, and broker integration carefully.
