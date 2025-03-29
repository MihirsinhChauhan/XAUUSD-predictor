from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime, timedelta
import time
import numpy as np
import json

# Import project modules
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from ml_model import CandlePredictionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# Custom JSON encoder to handle numpy and datetime types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# Initialize FastAPI app
app = FastAPI(title="XAUUSD Prediction API", description="API for predicting XAUUSD price movements")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
connector = None
processor = None
model = None
last_fetch_time = None
market_data = None
is_fetching = False

# Pydantic models
class PredictionResult(BaseModel):
    timestamp: str
    direction: str
    probability: float
    signal_strength: str
    
class ModelParams(BaseModel):
    threshold: float = 0.5
    model_version: Optional[str] = None

class MarketData(BaseModel):
    candles: List[Dict[str, Any]]
    last_updated: str

# Initialization function
def init_services():
    global connector, processor, model
    
    try:
        # Initialize MT5 connector
        logger.info("Initializing MT5 connector...")
        connector = MT5Connector()
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = DataProcessor(
            scale_features=True,
            feature_selection=True,
            n_features=30,
            handle_missing='fill_median',
            smooth_outliers=True
        )
        
        # Check if processor state exists
        processor_path = os.path.join("models", "xauusd_model_processor.pkl")
        if os.path.exists(processor_path):
            logger.info(f"Loading processor state from {processor_path}")
            processor.load_processor_state(processor_path)
        
        # Initialize model
        logger.info("Initializing prediction model...")
        model = CandlePredictionModel(
            model_dir="models",
            model_name="xauusd_model"
        )
        
        logger.info("Services initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return False

# Background task to fetch data
async def fetch_data_task():
    global connector, market_data, last_fetch_time, is_fetching
    
    if is_fetching:
        logger.info("Data fetch already in progress, skipping")
        return
    
    is_fetching = True
    
    try:
        logger.info("Fetching market data in background...")
        
        # Connect to MT5 if needed
        if not connector.connected:
            logger.info("Connecting to MT5...")
            if not connector.connect():
                logger.error("Failed to connect to MT5")
                is_fetching = False
                return
        
        # Fetch data
        df = connector.fetch_data(symbol="XAUUSD", timeframe=5, num_candles=1000)
        
        if df is not None:
            # Convert to list of dictionaries for JSON serialization
            candles = json.loads(json.dumps(df.to_dict('records'), cls=NpEncoder))
            
            # Update global market data
            market_data = {
                "candles": candles,
                "last_updated": datetime.now().isoformat()
            }
            
            last_fetch_time = datetime.now()
            logger.info(f"Market data updated successfully with {len(candles)} candles")
        else:
            logger.error("Failed to fetch market data")
        
    except Exception as e:
        logger.error(f"Error in background data fetch: {str(e)}")
    
    finally:
        is_fetching = False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    init_services()
    
# API Routes
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"status": "online", "message": "XAUUSD Prediction API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint with service statuses"""
    global connector, processor, model
    
    # Check if services are initialized
    services_status = {
        "connector": connector is not None and connector.connected,
        "processor": processor is not None,
        "model": model is not None,
        "data": market_data is not None,
        "last_fetch": last_fetch_time.isoformat() if last_fetch_time else None
    }
    
    return {
        "status": "healthy" if all([services_status["connector"], services_status["processor"], services_status["model"]]) else "degraded",
        "services": services_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reconnect")
async def reconnect_mt5():
    """Force reconnection to MT5"""
    global connector
    
    if connector is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    if connector.connect():
        return {"status": "success", "message": "Reconnected to MT5 successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to MT5")

@app.post("/market-data/fetch")
async def fetch_market_data(background_tasks: BackgroundTasks):
    """Trigger a market data fetch in the background"""
    global last_fetch_time
    
    # Check if we need to refresh (avoid too frequent refreshes)
    can_refresh = True
    if last_fetch_time is not None:
        time_since_last = datetime.now() - last_fetch_time
        if time_since_last < timedelta(seconds=30):
            can_refresh = False
    
    if can_refresh:
        background_tasks.add_task(fetch_data_task)
        return {"status": "fetching", "message": "Market data fetch started in background"}
    else:
        return {
            "status": "skipped", 
            "message": f"Last fetch was {(datetime.now() - last_fetch_time).total_seconds():.1f} seconds ago. Wait at least 30 seconds between refreshes."
        }

@app.get("/market-data")
async def get_market_data(background_tasks: BackgroundTasks):
    """Get the latest market data"""
    global market_data, last_fetch_time
    
    # Check if we have data or if it's stale
    if market_data is None or (last_fetch_time and (datetime.now() - last_fetch_time) > timedelta(minutes=5)):
        # Start a background fetch if data is missing or stale
        background_tasks.add_task(fetch_data_task)
        
        if market_data is None:
            raise HTTPException(status_code=404, detail="Market data not available yet, fetch in progress")
    
    return market_data

@app.post("/predict")
async def predict_next_candle(params: ModelParams = None):
    """Predict the next candle direction"""
    global connector, processor, model, market_data
    
    if params is None:
        params = ModelParams()
    
    # Check if services are initialized
    if connector is None or processor is None or model is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    # Check if we have market data
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not available, please fetch first")
    
    try:
        # Convert market data back to DataFrame
        df = pd.DataFrame(market_data["candles"])
        
        # Convert date strings back to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Prepare data for prediction
        feature_list = model.feature_list
        if not feature_list:
            raise HTTPException(status_code=500, detail="Model doesn't have feature list")
        
        # Load specific model version if provided
        if params.model_version:
            if not model.load_model_version(params.model_version):
                raise HTTPException(status_code=404, detail=f"Model version {params.model_version} not found")
        
        # Prepare latest data for prediction
        latest_features = processor.prepare_latest_for_prediction(df, feature_list)
        
        # Make prediction
        prediction = model.predict_single(latest_features, threshold=params.threshold)
        
        # Format response
        result = {
            "timestamp": datetime.now().isoformat(),
            "direction": prediction["prediction_label"],
            "probability": prediction["bullish_probability" if prediction["prediction_label"] == "Bullish" else "bearish_probability"],
            "signal_strength": prediction["signal_strength"],
            "details": prediction  # Include all prediction details
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/versions")
async def get_model_versions():
    """Get available model versions"""
    global model
    
    if model is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    try:
        # Get list of available model versions
        model_dir = "models"
        model_name = "xauusd_model"
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith(".pkl")]
        
        # Extract version info
        versions = []
        current_version = model.metadata.get("model_version", "unknown")
        
        for file in model_files:
            if "_v" in file:
                version = file.split("_v")[1].split(".")[0]
                versions.append({
                    "version": version,
                    "file": file,
                    "is_current": version == current_version
                })
        
        return {
            "current_version": current_version,
            "available_versions": sorted(versions, key=lambda x: x["version"], reverse=True)
        }
    
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)