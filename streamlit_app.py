"""
Streamlit Interface for LSTM Stock Forecast.

This module provides a web interface for stock price forecasting using
an LSTM model with attention mechanism.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LSTM Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model_and_artifact():
    """
    Load the LSTM model and artifacts.
    
    Returns:
        tuple: (model, artifact, device)
    """
    from src.models.lstm import AttnLSTM
    
    MODEL_PATH = os.environ.get("MODEL_PATH", "src/artifacts/models/lstm_v1.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load artifact
    artifact = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Get hyperparameters from artifact
    hidden_size = artifact.get("hidden_size", 64)
    num_layers = artifact.get("num_layers", 2)
    
    # Create model
    model = AttnLSTM(
        input_size=len(artifact["features"]),
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    model.load_state_dict(artifact["model_state"])
    model.to(device)
    model.eval()
    
    return model, artifact, device


def load_stock_data(ticker, start_date, end_date=None):
    """
    Load stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        pd.DataFrame: Stock data
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df


def create_features(df, feature_cols):
    """
    Create features for the LSTM model.
    
    Args:
        df: DataFrame with stock data
        feature_cols: List of feature column names
    
    Returns:
        pd.DataFrame: DataFrame with features
    """
    from src.features.features import create_features as create_feat
    
    df = df.copy()
    df = create_feat(df)
    
    return df


def prepare_sequence(df, features, seq_len):
    """
    Prepare the last sequence for prediction.
    
    Args:
        df: DataFrame with features
        features: List of feature column names
        seq_len: Sequence length
    
    Returns:
        tuple: (last_seq, last_price) or (None, None) if insufficient data
    """
    if df is None or len(df) < seq_len + 5:
        return None, None
    
    last_price = float(df["Close"].iloc[-1])
    last_seq = df[features].iloc[-seq_len:].values
    
    return last_seq, last_price


def forecast_n_days(model, last_seq, n_days, artifact, device, current_price):
    """
    Generate n-day price forecast.
    
    Args:
        model: LSTM model
        last_seq: Last sequence of features
        n_days: Number of days to forecast
        artifact: Model artifact containing scalers
        device: Device to run inference on
        current_price: Current stock price
    
    Returns:
        tuple: (prices, returns)
    """
    # Get scalers from artifact
    x_scaler = artifact.get("x_scaler")
    y_scaler = artifact.get("y_scaler")
    
    # If scalers not saved, create simple ones
    if x_scaler is None or y_scaler is None:
        x_mean = artifact.get("x_mean", np.zeros(last_seq.shape[1]))
        x_std = artifact.get("x_std", np.ones(last_seq.shape[1]))
        
        class SimpleScaler:
            def transform(self, data):
                return (data - x_mean) / (x_std + 1e-8)
            
            def inverse_transform(self, data):
                return data * (x_std + 1e-8) + x_mean
        
        x_scaler = SimpleScaler()
        y_scaler = SimpleScaler()
    
    preds = []
    seq = last_seq.copy()
    
    for _ in range(n_days):
        # Scale input
        seq_scaled = x_scaler.transform(seq)
        X = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(X).item()
        
        # Inverse transform prediction
        pred = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred))
        
        # Roll sequence and update with prediction
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 0] = pred
    
    returns = np.array(preds)
    # Convert returns to actual prices using current price
    prices = current_price * np.exp(np.cumsum(returns))
    
    return prices, returns


# =========================================================
# UI COMPONENTS
# =========================================================
def show_model_info(artifact, device):
    """
    Display model information in the sidebar.
    
    Args:
        artifact: Model artifact
        device: Device model is running on
    """
    st.sidebar.header("📊 Model Information")
    
    st.sidebar.markdown("### Model Details")
    st.sidebar.info(f"**Model Type:** {artifact.get('model_type', 'AttnLSTM')}")
    st.sidebar.info(f"**Version:** {artifact.get('version', 'unknown')}")
    st.sidebar.info(f"**Device:** {device}")
    
    st.sidebar.markdown("### Configuration")
    st.sidebar.write(f"**Sequence Length:** {artifact.get('seq_len', 60)} days")
    st.sidebar.write(f"**Hidden Size:** {artifact.get('hidden_size', 64)}")
    st.sidebar.write(f"**Number of Layers:** {artifact.get('num_layers', 2)}")
    
    st.sidebar.markdown("### Features")
    features = artifact.get('features', [])
    for feat in features:
        st.sidebar.write(f"- {feat}")


def create_forecast_chart(forecast_dates, predicted_prices, current_price, ticker):
    """
    Create an interactive chart for forecast results.
    
    Args:
        forecast_dates: List of forecast dates
        predicted_prices: List of predicted prices
        current_price: Current stock price
        ticker: Stock symbol
    
    Returns:
        plotly.graph_objects.Figure: Chart figure
    """
    fig = go.Figure()
    
    # Add current price line
    fig.add_trace(go.Scatter(
        x=[forecast_dates[0] - timedelta(days=1)],
        y=[current_price],
        mode='markers+lines',
        name='Current Price',
        marker=dict(color='blue', size=10),
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Add predicted price line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Price',
        marker=dict(color='red', size=8),
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_historical_chart(df, ticker):
    """
    Create an interactive chart for historical data.
    
    Args:
        df: DataFrame with historical stock data
        ticker: Stock symbol
    
    Returns:
        plotly.graph_objects.Figure: Chart figure
    """
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'Close' in df.columns:
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA200'],
            mode='lines',
            name='200-Day MA',
            line=dict(color='purple', width=1)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Historical Data",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_forecast_table(forecast_dates, prices, returns):
    """
    Create a DataFrame for forecast results.
    
    Args:
        forecast_dates: List of forecast dates
        prices: List of predicted prices
        returns: List of predicted returns
    
    Returns:
        pd.DataFrame: Forecast results table
    """
    data = {
        "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "Day": range(1, len(forecast_dates) + 1),
        "Price ($)": [round(p, 2) for p in prices],
        "Return": [round(r, 6) for r in returns],
        "Return (%)": [round(r * 100, 4) for r in returns]
    }
    
    return pd.DataFrame(data)


# =========================================================
# MAIN APPLICATION
# =========================================================
def main():
    """Main function for the Streamlit application."""
    
    # Title and description
    st.title("📈 LSTM Stock Price Forecast")
    st.markdown("""
    This application uses an LSTM model with temporal attention mechanism 
    to forecast stock prices for the next N days.
    """)
    
    # Load model
    try:
        model, artifact, device = load_model_and_artifact()
        features = artifact.get("features", [])
        seq_len = artifact.get("seq_len", 60)
        
        # Show model info in sidebar
        show_model_info(artifact, device)
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar inputs
    st.sidebar.header("⚙️ Forecast Settings")
    
    # Ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    
    # Number of days to forecast
    n_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to forecast (max: 30)"
    )
    
    # Historical data range
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Historical Data")
    historical_days = st.sidebar.slider(
        "Historical Days",
        min_value=30,
        max_value=730,
        value=365,
        help="Number of days of historical data to display"
    )
    
    # Forecast button
    forecast_button = st.sidebar.button("🔮 Generate Forecast", type="primary")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    # Load historical data
    try:
        start_date = (date.today() - timedelta(days=historical_days + 30)).strftime("%Y-%m-%d")
        df_historical = load_stock_data(ticker, start_date)
        
        if df_historical.empty:
            st.error(f"No data found for ticker {ticker}")
            st.stop()
            
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        st.stop()
    
    # Display historical chart
    with col1:
        st.subheader(f"📊 {ticker} Historical Data")
        historical_chart = create_historical_chart(df_historical.tail(historical_days), ticker)
        st.plotly_chart(historical_chart, use_container_width=True)
    
    # Display current price
    with col2:
        current_price = float(df_historical["Close"].iloc[-1])
        price_change = float(df_historical["Close"].iloc[-1] - df_historical["Close"].iloc[-2])
        price_change_pct = (price_change / float(df_historical["Close"].iloc[-2])) * 100
        
        st.metric(
            label=f"{ticker} Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Market Stats")
        high_52w = float(df_historical["High"].tail(252).max())
        low_52w = float(df_historical["Low"].tail(252).min())
        volume = int(df_historical["Volume"].iloc[-1])
        
        st.write(f"**52-Week High:** ${high_52w:.2f}")
        st.write(f"**52-Week Low:** ${low_52w:.2f}")
        st.write(f"**Volume:** {volume:,}")
    
    st.markdown("---")
    
    # Generate forecast when button is clicked
    if forecast_button:
        with st.spinner("Generating forecast..."):
            try:
                # Prepare features
                df_features = create_features(df_historical, features)
                
                # Prepare sequence
                last_seq, last_price = prepare_sequence(df_features, features, seq_len)
                
                if last_seq is None:
                    st.error(f"Insufficient data for ticker {ticker}")
                    st.stop()
                
                # Generate forecast
                prices, returns = forecast_n_days(
                    model, last_seq, n_days, artifact, device, current_price
                )
                
                # Generate forecast dates
                forecast_dates = [
                    datetime.now() + timedelta(days=i+1)
                    for i in range(n_days)
                ]
                
                # Display forecast results
                st.subheader(f"🔮 {ticker} Price Forecast (Next {n_days} Days)")
                
                # Forecast chart
                forecast_chart = create_forecast_chart(
                    forecast_dates, prices, current_price, ticker
                )
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Forecast table
                st.subheader("📋 Forecast Details")
                forecast_df = create_forecast_table(forecast_dates, prices, returns)
                
                # Display with formatting
                st.dataframe(
                    forecast_df.style.format({
                        "Price ($)": "${:.2f}",
                        "Return": "{:.6f}",
                        "Return (%)": "{:.4f}%"
                    }).background_gradient(
                        subset=["Return (%)"],
                        cmap="RdYlGn"
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Summary statistics
                st.subheader("📊 Forecast Summary")
                col_avg, col_max, col_min, col_total = st.columns(4)
                
                with col_avg:
                    avg_return = np.mean(returns) * 100
                    st.metric("Avg Daily Return", f"{avg_return:.4f}%")
                
                with col_max:
                    max_return = np.max(returns) * 100
                    st.metric("Max Daily Return", f"{max_return:.4f}%")
                
                with col_min:
                    min_return = np.min(returns) * 100
                    st.metric("Min Daily Return", f"{min_return:.4f}%")
                
                with col_total:
                    total_return = (prices[-1] - current_price) / current_price * 100
                    st.metric("Total Forecast Return", f"{total_return:.2f}%")
                
                st.success("Forecast generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        st.info("👈 Click 'Generate Forecast' in the sidebar to get predictions!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>LSTM Stock Forecast | Model Version: {}</p>
    </div>
    """.format(artifact.get("version", "unknown")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
