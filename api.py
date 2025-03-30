from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import os
import logging
from datetime import datetime, timedelta
import time
import numpy as np
import json
import traceback
import asyncio
from contextlib import asynccontextmanager

# Import project modules
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from ml_model import CandlePredictionModel
from model_predictor import ModelPredictor

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

# Global state
class GlobalState:
    def __init__(self):
        self.connector = None
        self.predictor = None
        self.last_fetch_time = None
        self.market_data = None
        self.is_fetching = False
        self.fetch_semaphore = asyncio.Semaphore(1)
        self.prediction_semaphore = asyncio.Semaphore(1)
        self.config = {
            "model_dir": "models",
            "model_name": "xauusd_model",
            "processor_path": None,
            "model_version": None,
            "symbol": "XAUUSD",
            "timeframe": 5,
            "default_candles": 1000,
            "threshold": 0.5
        }
        
    def to_dict(self):
        return {
            "config": self.config,
            "connector_ready": self.connector is not None and self.connector.connected,
            "predictor_ready": self.predictor is not None,
            "last_fetch_time": self.last_fetch_time,
            "market_data_available": self.market_data is not None,
            "is_fetching": self.is_fetching
        }

# Global state instance
state = GlobalState()

# Lifespan for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize services
    init_services()
    yield
    # Shutdown: clean up resources
    cleanup_services()

# Initialize FastAPI app
app = FastAPI(
    title="XAUUSD Prediction API", 
    description="API for predicting XAUUSD price movements using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Pydantic models
class PredictionParams(BaseModel):
    threshold: float = Field(0.5, description="Probability threshold for classification")
    model_version: Optional[str] = Field(None, description="Specific model version to use")

class MarketDataParams(BaseModel):
    symbol: str = Field("XAUUSD", description="Trading symbol")
    timeframe: int = Field(5, description="Timeframe in minutes")
    num_candles: int = Field(1000, description="Number of candles to fetch")

class ConfigParams(BaseModel):
    model_dir: Optional[str] = Field(None, description="Directory with model files")
    model_name: Optional[str] = Field(None, description="Base name of the model")
    processor_path: Optional[str] = Field(None, description="Path to processor state file")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    timeframe: Optional[int] = Field(None, description="Timeframe in minutes")
    threshold: Optional[float] = Field(None, description="Prediction probability threshold")

# Initialization and cleanup functions
def init_services():
    """Initialize all services based on configuration"""
    global state
    
    try:
        # Initialize MT5 connector
        logger.info("Initializing MT5 connector...")
        state.connector = MT5Connector()
        
        # Initialize predictor
        init_predictor()
        
        logger.info("Services initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def init_predictor():
    """Initialize the model predictor"""
    global state
    
    try:
        logger.info(f"Initializing predictor with model: {state.config['model_name']}")
        
        state.predictor = ModelPredictor(
            model_dir=state.config['model_dir'],
            model_name=state.config['model_name'],
            processor_path=state.config['processor_path'],
            model_version=state.config['model_version'],
            threshold=state.config['threshold']
        )
        
        # Check if predictor initialized successfully
        model_info = state.predictor.get_model_info()
        if "error" in model_info:
            logger.error(f"Predictor initialization failed: {model_info['error']}")
            return False
        
        logger.info(f"Predictor initialized with model version: {model_info.get('model_version', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def cleanup_services():
    """Clean up resources before shutdown"""
    global state
    
    try:
        # Disconnect MT5 if connected
        if state.connector and state.connector.connected:
            state.connector.disconnect()
            logger.info("MT5 connector disconnected")
        
        # Clear any cached data
        state.market_data = None
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Background task to fetch data
async def fetch_data_task(symbol: str, timeframe: int, num_candles: int):
    """Background task to fetch market data"""
    global state
    
    try:
        # Try to acquire semaphore with timeout
        acquired = False
        try:
            acquired = await asyncio.wait_for(state.fetch_semaphore.acquire(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Another fetch operation is in progress and timeout occurred")
            return
            
        if not acquired:
            logger.warning("Could not acquire fetch semaphore")
            return
            
        state.is_fetching = True
        
        try:
            logger.info(f"Fetching market data for {symbol}, timeframe: {timeframe}, candles: {num_candles}")
            
            # Connect to MT5 if needed
            if not state.connector.connected:
                logger.info("Connecting to MT5...")
                if not state.connector.connect():
                    logger.error("Failed to connect to MT5")
                    return
            
            # Fetch data
            start_time = time.time()
            df = state.connector.fetch_data(symbol=symbol, timeframe=timeframe, num_candles=num_candles)
            fetch_time = time.time() - start_time
            
            if df is not None and not df.empty:
                # Validate data integrity
                required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Fetched data is missing required columns: {missing_columns}")
                    return
                
                # Convert to list of dictionaries for JSON serialization
                candles = json.loads(json.dumps(df.to_dict('records'), cls=NpEncoder))
                
                # Update global market data
                state.market_data = {
                    "candles": candles,
                    "last_updated": datetime.now().isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles_count": len(candles),
                    "fetch_time_seconds": fetch_time
                }
                
                state.last_fetch_time = datetime.now()
                logger.info(f"Market data updated successfully with {len(candles)} candles in {fetch_time:.2f} seconds")
            else:
                logger.error("Failed to fetch market data - empty dataset returned")
        finally:
            state.is_fetching = False
            state.fetch_semaphore.release()
    except Exception as e:
        logger.error(f"Error in background data fetch: {str(e)}")
        logger.error(traceback.format_exc())
        state.is_fetching = False
        if 'acquired' in locals() and acquired:
            state.fetch_semaphore.release()

# API Routes
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "status": "online", 
        "message": "XAUUSD Prediction API is running",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with service statuses"""
    global state
    
    # Get predictor info
    predictor_info = {}
    if state.predictor:
        predictor_info = state.predictor.get_model_info()
    
    # Check if services are initialized
    services_status = {
        "connector": state.connector is not None and state.connector.connected,
        "predictor": state.predictor is not None,
        "data": state.market_data is not None,
        "last_fetch": state.last_fetch_time.isoformat() if state.last_fetch_time else None,
        "model_info": predictor_info
    }
    
    return {
        "status": "healthy" if all([services_status["connector"], services_status["predictor"]]) else "degraded",
        "services": services_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reconnect")
async def reconnect_mt5():
    """Force reconnection to MT5"""
    global state
    
    if state.connector is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    if state.connector.connect():
        return {"status": "success", "message": "Reconnected to MT5 successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to MT5")

@app.post("/market-data/fetch")
async def fetch_market_data(
    background_tasks: BackgroundTasks,
    params: MarketDataParams = None
):
    """Trigger a market data fetch in the background"""
    global state
    
    if params is None:
        params = MarketDataParams()
    
    # Check if we need to refresh (avoid too frequent refreshes)
    can_refresh = True
    if state.last_fetch_time is not None:
        time_since_last = datetime.now() - state.last_fetch_time
        if time_since_last < timedelta(seconds=30):
            can_refresh = False
    
    if can_refresh:
        background_tasks.add_task(
            fetch_data_task, 
            params.symbol, 
            params.timeframe, 
            params.num_candles
        )
        return {
            "status": "fetching", 
            "message": "Market data fetch started in background",
            "params": {
                "symbol": params.symbol,
                "timeframe": params.timeframe,
                "num_candles": params.num_candles
            }
        }
    else:
        return {
            "status": "skipped", 
            "message": f"Last fetch was {(datetime.now() - state.last_fetch_time).total_seconds():.1f} seconds ago. Wait at least 30 seconds between refreshes."
        }

@app.get("/market-data")
async def get_market_data(background_tasks: BackgroundTasks):
    """Get the latest market data"""
    global state
    
    # Check if we have data or if it's stale
    if state.market_data is None or (state.last_fetch_time and (datetime.now() - state.last_fetch_time) > timedelta(minutes=5)):
        # Start a background fetch if data is missing or stale
        background_tasks.add_task(
            fetch_data_task, 
            state.config["symbol"], 
            state.config["timeframe"], 
            state.config["default_candles"]
        )
        
        if state.market_data is None:
            raise HTTPException(
                status_code=404, 
                detail="Market data not available yet, fetch in progress"
            )
    
    return state.market_data

@app.post("/predict")
async def predict_next_candle(params: PredictionParams = None):
    """Predict the next candle direction"""
    global state
    
    if params is None:
        params = PredictionParams()
    
    # Check if services are initialized
    if state.connector is None or state.predictor is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    # Check if we have market data
    if state.market_data is None:
        raise HTTPException(status_code=404, detail="Market data not available, please fetch first")
    
    try:
        # Acquire prediction semaphore with timeout
        acquired = False
        try:
            acquired = await asyncio.wait_for(state.prediction_semaphore.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Server is busy processing another prediction. Please try again shortly."
            )
            
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Could not acquire prediction semaphore"
            )
            
        try:
            # Convert market data back to DataFrame
            df = pd.DataFrame(state.market_data["candles"])
            
            if df.empty:
                raise HTTPException(status_code=400, detail="Market data DataFrame is empty")
            
            # Check required columns
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Market data is missing required columns: {missing_columns}"
                )
            
            # Convert date strings back to datetime
            df['time'] = pd.to_datetime(df['time'])
            
            # Ensure data is sorted
            df = df.sort_values('time')
            
            # Use the predictor for prediction
            prediction_start = time.time()
            
            # Override threshold if provided
            if params.threshold != state.config["threshold"]:
                state.predictor.threshold = params.threshold
            
            # Make prediction with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(state.predictor.predict, df),
                    timeout=20.0  # 20 second timeout
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="Prediction timed out. The model is taking too long to process data."
                )
            
            # Check if prediction had an error
            if "error" in result:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {result['error']}"
                )
            
            # Add timestamp and prediction time
            result["request_timestamp"] = datetime.now().isoformat()
            result["prediction_time_seconds"] = time.time() - prediction_start
            
            return result
        finally:
            state.prediction_semaphore.release()
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/versions")
async def get_model_versions():
    """Get available model versions"""
    global state
    
    if state.predictor is None:
        if not init_predictor():
            raise HTTPException(status_code=500, detail="Failed to initialize predictor")
    
    try:
        # Get model info
        model_info = state.predictor.get_model_info()
        
        # Get list of available model versions from filesystem
        model_dir = state.config["model_dir"]
        model_name = state.config["model_name"]
        
        # Search for model files
        versions = []
        model_files = []
        
        for ext in [".joblib", ".pkl"]:
            pattern = os.path.join(model_dir, f"{model_name}*{ext}")
            import glob
            model_files.extend(glob.glob(pattern))
        
        # Extract version info
        current_version = model_info.get("model_version", "unknown")
        
        for file in model_files:
            # Skip processor files
            if "_processor.pkl" in file:
                continue
                
            # Try to extract version info
            filename = os.path.basename(file)
            base_name = model_name
            
            # Check if file matches the current model
            if filename.startswith(base_name):
                # Try to extract version from filename
                version = "unknown"
                
                # Look for _v pattern for version
                if "_v" in filename:
                    version = filename.split("_v")[1].split(".")[0]
                # Look for timestamp pattern
                elif "_20" in filename:
                    # Use timestamp as version
                    version_parts = filename.replace(base_name, "").split("_")
                    version_parts = [p for p in version_parts if p and p[0].isdigit()]
                    if version_parts:
                        version = "_".join(version_parts[:2])  # Use first 2 timestamp parts
                
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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/config")
async def update_config(params: ConfigParams):
    """Update API configuration"""
    global state
    
    # Store old config for reverting if needed
    old_config = state.config.copy()
    config_changed = False
    
    try:
        # Update config parameters if provided
        if params.model_dir is not None:
            state.config["model_dir"] = params.model_dir
            config_changed = True
            
        if params.model_name is not None:
            state.config["model_name"] = params.model_name
            config_changed = True
            
        if params.processor_path is not None:
            state.config["processor_path"] = params.processor_path
            config_changed = True
            
        if params.model_version is not None:
            state.config["model_version"] = params.model_version
            config_changed = True
            
        if params.symbol is not None:
            state.config["symbol"] = params.symbol
            config_changed = True
            
        if params.timeframe is not None:
            state.config["timeframe"] = params.timeframe
            config_changed = True
            
        if params.threshold is not None:
            state.config["threshold"] = params.threshold
            config_changed = True
        
        # Reinitialize services if config changed
        if config_changed:
            if not init_predictor():
                # Revert to old config if initialization failed
                state.config = old_config
                raise HTTPException(status_code=500, detail="Failed to initialize with new configuration")
            
            # Clear market data if symbol or timeframe changed
            if params.symbol is not None or params.timeframe is not None:
                state.market_data = None
                state.last_fetch_time = None
        
        return {"status": "success", "config": state.config}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Revert to old config on error
        state.config = old_config
        logger.error(f"Error updating config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    global state
    
    return {"config": state.config}

@app.get("/debug/state")
async def debug_state():
    """Debug endpoint to get the current state"""
    global state
    
    return state.to_dict()

@app.post("/manage/gc")
async def force_garbage_collection():
    """Force garbage collection to free memory"""
    import gc
    
    before = gc.get_count()
    collected = gc.collect(generation=2)
    after = gc.get_count()
    
    return {
        "status": "success",
        "collected": collected,
        "before": before,
        "after": after
    }

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)