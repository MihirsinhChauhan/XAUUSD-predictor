import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import pytz
import time

# Configuration
API_URL = "http://localhost:8000"  # FastAPI backend URL
REFRESH_INTERVAL = 60  # Default refresh interval in seconds

# Page configuration
st.set_page_config(
    page_title="XAUUSD Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: gold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .prediction-bullish {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.5);
    }
    .prediction-bearish {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.5);
    }
    .prediction-label {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }
    .prediction-bullish .prediction-label {
        color: #00FF00;
    }
    .prediction-bearish .prediction-label {
        color: #FF0000;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #CCC;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-green {
        background-color: #00FF00;
    }
    .status-red {
        background-color: #FF0000;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = REFRESH_INTERVAL
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'services_status' not in st.session_state:
    st.session_state.services_status = {}
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Helper functions
def fetch_market_data():
    """Fetch market data from the API"""
    try:
        response = requests.get(f"{API_URL}/market-data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Verify data structure
            if not data or "candles" not in data:
                st.session_state.error_message = "Received invalid data structure from API"
                return False
                
            # Convert to DataFrame
            candles = pd.DataFrame(data["candles"])
            
            if len(candles) == 0:
                st.session_state.error_message = "Received empty candles data from API"
                return False
                
            # Convert time strings to datetime
            candles['time'] = pd.to_datetime(candles['time'])
            
            st.session_state.market_data = candles
            st.session_state.last_update = datetime.now()
            st.session_state.error_message = None
            
            return True
        elif response.status_code == 404:
            # Try to trigger a fetch if data is not available
            fetch_response = requests.post(f"{API_URL}/market-data/fetch", timeout=10)
            st.session_state.error_message = f"Triggered market data fetch: {fetch_response.json().get('message', 'No message')}"
            return False
        else:
            st.session_state.error_message = f"API error: {response.status_code} - {response.text}"
            return False
    except requests.exceptions.Timeout:
        st.session_state.error_message = "API request timed out. Server might be busy."
        return False
    except requests.exceptions.ConnectionError:
        st.session_state.error_message = "Connection error. API server might be down."
        return False
    except Exception as e:
        st.session_state.error_message = f"Error fetching market data: {str(e)}"
        return False

def make_prediction(threshold=0.5):
    """Get prediction from the API"""
    try:
        params = {"threshold": threshold}
        
        response = requests.post(f"{API_URL}/predict", json=params, timeout=200)  # Longer timeout for predictions
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Validate prediction structure
            if not isinstance(prediction, dict) or "prediction" not in prediction:
                st.session_state.error_message = f"Invalid prediction format: {prediction}"
                return False
                
            st.session_state.prediction = prediction
            st.session_state.error_message = None
            return True
        elif response.status_code == 404:
            # Special handling for 404 (data not available)
            st.session_state.error_message = "Market data not available yet. Please fetch data first."
            return False
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text
            
            st.session_state.error_message = f"Error getting prediction: {error_detail}"
            return False
    except requests.exceptions.Timeout:
        st.session_state.error_message = "Prediction request timed out. The model might be processing a large dataset."
        return False
    except requests.exceptions.ConnectionError:
        st.session_state.error_message = "Connection error. API server might be down."
        return False
    except Exception as e:
        st.session_state.error_message = f"Error making prediction: {str(e)}"
        return False

def check_api_health():
    """Check the API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.services_status = response.json()
            return True
        else:
            st.session_state.error_message = f"API health check failed: {response.status_code}"
            return False
    except requests.exceptions.Timeout:
        st.session_state.error_message = "API health check timed out"
        return False
    except requests.exceptions.ConnectionError:
        st.session_state.error_message = "Connection error. API server might be down."
        return False
    except Exception as e:
        st.session_state.error_message = f"Error connecting to API: {str(e)}"
        return False

def get_model_info():
    """Get information about the current model"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            st.session_state.model_info = response.json().get('model_info', {})
            return True
        else:
            st.session_state.error_message = f"Failed to get model info: {response.status_code}"
            return False
    except Exception as e:
        st.session_state.error_message = f"Error getting model info: {str(e)}"
        return False

def reconnect_mt5():
    """Trigger a reconnection to MT5"""
    try:
        response = requests.post(f"{API_URL}/reconnect", timeout=10)
        if response.status_code == 200:
            st.success("Reconnected to MT5 successfully")
            return True
        else:
            st.error(f"Failed to reconnect to MT5: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error reconnecting to MT5: {str(e)}")
        return False

def trigger_data_fetch():
    """Manually trigger a data fetch"""
    try:
        response = requests.post(f"{API_URL}/market-data/fetch", timeout=10)
        if response.status_code == 200:
            st.success(response.json()["message"])
            return True
        else:
            st.error(f"Failed to trigger data fetch: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error triggering data fetch: {str(e)}")
        return False

def plot_candlestick_chart(df, prediction=None):
    """Create a candlestick chart with indicators"""
    try:
        if df is None or len(df) == 0:
            return None
        
        # Sort by time and use last 200 candles for display
        df = df.sort_values('time').tail(200)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.7, 0.3],
                            subplot_titles=('XAUUSD Price', 'Volume'))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='XAUUSD',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume bars
        colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['tick_volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Add prediction marker if available
        if prediction is not None:
            last_time = df['time'].iloc[-1]
            next_time = last_time + timedelta(minutes=5)
            
            # Get prediction details
            direction = prediction.get('prediction_label', 'Unknown')
            probability = prediction.get('probability', 0.5)
            
            # Determine color and symbol based on prediction
            color = 'green' if direction == "Bullish" else 'red'
            symbol = 'triangle-up' if direction == "Bullish" else 'triangle-down'
            
            # Add prediction marker
            fig.add_trace(
                go.Scatter(
                    x=[next_time],
                    y=[df['close'].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=color,
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=f'Prediction: {direction} ({probability:.1%})',
                    hoverinfo='name'
                ),
                row=1, col=1
            )
        
        # Customize layout
        fig.update_layout(
            title='XAUUSD 5-Minute Chart',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    # Service status
    st.markdown("#### Service Status")
    if check_api_health():
        services = st.session_state.services_status.get('services', {})
        
        # MT5 Connection
        mt5_status = services.get('connector', False)
        mt5_status_color = "status-green" if mt5_status else "status-red"
        st.markdown(f'<div><span class="status-indicator {mt5_status_color}"></span> MT5 Connection: {"Connected" if mt5_status else "Disconnected"}</div>', unsafe_allow_html=True)
        
        # Data status
        data_status = services.get('data', False)
        data_status_color = "status-green" if data_status else "status-red"
        st.markdown(f'<div><span class="status-indicator {data_status_color}"></span> Market Data: {"Available" if data_status else "Not available"}</div>', unsafe_allow_html=True)
        
        # Predictor status
        predictor_status = services.get('predictor', False)
        predictor_status_color = "status-green" if predictor_status else "status-red"
        st.markdown(f'<div><span class="status-indicator {predictor_status_color}"></span> Model: {"Ready" if predictor_status else "Not ready"}</div>', unsafe_allow_html=True)
        
        # Last fetch time
        last_fetch = services.get('last_fetch')
        if last_fetch:
            last_fetch_dt = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
            time_diff = datetime.now(pytz.UTC) - last_fetch_dt.replace(tzinfo=pytz.UTC)
            st.markdown(f"Last data fetch: {time_diff.total_seconds():.0f} seconds ago")
        
        # Reconnect button
        if not mt5_status:
            if st.button("Reconnect to MT5"):
                reconnect_mt5()
        
        # Manual data fetch button
        if st.button("Fetch Latest Data"):
            trigger_data_fetch()
    
    # Model info
    get_model_info()
    if st.session_state.model_info:
        st.markdown("#### Model Information")
        model_info = st.session_state.model_info
        
        st.markdown(f"**Model:** {model_info.get('model_name', 'Unknown')}")
        st.markdown(f"**Version:** {model_info.get('model_version', 'Unknown')}")
        
        # Display additional model details if available
        metadata = model_info.get('metadata', {})
        if metadata and isinstance(metadata, dict):
            metrics = metadata.get('performance_metrics', {})
            if metrics and isinstance(metrics, dict):
                st.markdown("#### Model Metrics")
                metrics_cols = st.columns(2)
                
                with metrics_cols[0]:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
                    st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
                
                with metrics_cols[1]:
                    st.metric("F1 Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
                    st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    
    # Prediction threshold
    st.markdown("#### Prediction Settings")
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Auto-refresh settings
    st.markdown("#### Refresh Settings")
    st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.number_input(
            "Refresh Interval (seconds)", 
            min_value=10, 
            max_value=300, 
            value=st.session_state.refresh_interval
        )
    
    # Debug section
    with st.expander("Debug Information", expanded=False):
        if st.button("Force Refresh"):
            st.session_state.last_update = None
            st.rerun()
        
        if st.button("Clear Cache"):
            # Clear session state except for essential items
            keys_to_keep = ['auto_refresh', 'refresh_interval']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.success("Cache cleared!")
            st.rerun()
        
        st.markdown("#### API Status")
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                st.json(response.json())
            except Exception as e:
                st.error(f"API Error: {str(e)}")

# Main content
st.markdown('<div class="main-header">📈 XAUUSD Prediction Dashboard</div>', unsafe_allow_html=True)

# Display error message if any
if st.session_state.error_message:
    st.warning(st.session_state.error_message)

# Add a manual refresh button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = None
        st.rerun()

# Check if we need to update data
current_time = datetime.now()
should_update = False

if st.session_state.last_update is None:
    should_update = True
elif st.session_state.auto_refresh and (current_time - st.session_state.last_update).total_seconds() > st.session_state.refresh_interval:
    should_update = True

# Fetch data if needed
if should_update:
    data_fetched = False
    prediction_made = False
    
    with st.spinner('Fetching market data...'):
        data_fetched = fetch_market_data()
    
    if data_fetched:
        with st.spinner('Making prediction...'):
            prediction_made = make_prediction(threshold=threshold)
                
        if not prediction_made:
            st.warning("Could not get prediction. Using last available prediction if any.")
    else:
        st.warning("Could not fetch fresh market data. Using last available data if any.")

# Display last update time
if st.session_state.last_update:
    st.markdown(f'<div style="text-align:right;color:#999;font-size:0.8rem;">Last updated: {st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

# Display prediction and chart
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price Chart")
    
    if st.session_state.market_data is not None:
        fig = plot_candlestick_chart(st.session_state.market_data, st.session_state.prediction)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to create chart")
    else:
        st.info("Waiting for market data...")

with col2:
    st.subheader("Prediction")
    
    if st.session_state.prediction:
        prediction = st.session_state.prediction
        direction = prediction.get('prediction_label', 'Unknown')
        probability = prediction.get('probability', 0.5)
        signal_strength = prediction.get('signal_strength', 'Neutral')
        
        # Display prediction box
        prediction_class = "prediction-bullish" if direction == "Bullish" else "prediction-bearish"
        st.markdown(f'''
        <div class="prediction-box {prediction_class}">
            <div class="prediction-label">{direction}</div>
            <div style="text-align: center; font-size: 1.2rem; margin: 0.5rem 0;">
                Signal Strength: {signal_strength}
            </div>
            <div style="text-align: center; font-size: 1.5rem; margin: 0.5rem 0;">
                Probability: {probability:.1%}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display additional metrics
        details = prediction.get('details', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Bullish Probability</div>
                <div class="metric-value">{prediction.get('bullish_probability', 0):.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Bearish Probability</div>
                <div class="metric-value">{prediction.get('bearish_probability', 0):.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Display model confidence
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">{prediction.get('confidence', 0):.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display prediction timestamp
        timestamp = prediction.get('timestamp')
        if timestamp:
            pred_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            st.caption(f"Prediction made at: {pred_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("Waiting for prediction...")

# Add a data table
with st.expander("Recent Candles Data"):
    if st.session_state.market_data is not None:
        # Display most recent candles first
        recent_candles = st.session_state.market_data.sort_values('time', ascending=False).head(20)
        
        # Format the dataframe
        recent_candles_display = recent_candles.copy()
        recent_candles_display['time'] = recent_candles_display['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        recent_candles_display = recent_candles_display.rename(columns={
            'time': 'Time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
        })
        
        st.dataframe(recent_candles_display, use_container_width=True)
    else:
        st.info("No data available")

# Add a simple performance metrics expander
if st.session_state.prediction and 'details' in st.session_state.prediction:
    with st.expander("Prediction Details & Performance Metrics"):
        # Convert prediction details to a dataframe for easy viewing
        details = st.session_state.prediction.get('details', {})
        
        # Show more detailed performance metrics if available
        if 'prediction_time_ms' in st.session_state.prediction:
            perf_cols = st.columns(3)
            
            with perf_cols[0]:
                st.metric("Prediction Time", f"{st.session_state.prediction.get('prediction_time_ms', 0):.0f} ms")
            
            with perf_cols[1]:
                if 'feature_prep_time_ms' in st.session_state.prediction:
                    st.metric("Feature Prep Time", f"{st.session_state.prediction.get('feature_prep_time_ms', 0):.0f} ms")
            
            with perf_cols[2]:
                total_time = st.session_state.prediction.get('prediction_time_ms', 0) + st.session_state.prediction.get('feature_prep_time_ms', 0)
                st.metric("Total Processing Time", f"{total_time:.0f} ms")
        
        # Show detailed prediction info
        st.json(details)

# Auto-refresh logic
if st.session_state.auto_refresh:
    time_to_refresh = st.session_state.refresh_interval
    if st.session_state.last_update:
        elapsed = (datetime.now() - st.session_state.last_update).total_seconds()
        time_to_refresh = max(0, st.session_state.refresh_interval - elapsed)
    
    st.markdown(f'''
    <div style="position: fixed; bottom: 20px; right: 20px; background-color: #1E1E1E; 
                padding: 10px; border-radius: 5px; font-size: 0.8rem; color: #CCC;">
        Next refresh in: {int(time_to_refresh)} seconds
    </div>
    ''', unsafe_allow_html=True)
    
    # Add auto-refresh using JavaScript
    if time_to_refresh <= 1:
        st.rerun()

# Footer
st.markdown("""
---
### About
This dashboard displays real-time predictions for XAUUSD (Gold) price movements using machine learning.
The model analyzes 5-minute candles and predicts whether the next candle will be bullish or bearish.
""")