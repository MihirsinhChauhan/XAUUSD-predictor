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
predictor = None
last_fetch_time = None
market_data = None
is_fetching = False

# Pydantic models
class PredictionParams(BaseModel):
    threshold: float = 0.5
    model_version: Optional[str] = None

class MarketData(BaseModel):
    candles: List[Dict[str, Any]]
    last_updated: str

# Initialization function
# Initialization function
def init_services():
    global connector, predictor
    
    try:
        # Initialize MT5 connector
        logger.info("Initializing MT5 connector...")
        connector = MT5Connector()
        
        
        # Log success
        logger.info("Services initialized successfully")
        
        return True
            
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
        
        if df is not None and not df.empty:
            # Validate data integrity
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Fetched data is missing required columns: {missing_columns}")
                is_fetching = False
                return
            
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
            logger.error("Failed to fetch market data - empty dataset returned")
        
    except Exception as e:
        logger.error(f"Error in background data fetch: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
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
    global connector, predictor
    
    # Check if services are initialized
    services_status = {
        "connector": connector is not None and connector.connected,
        "predictor": predictor is not None,
        "data": market_data is not None,
        "last_fetch": last_fetch_time.isoformat() if last_fetch_time else None
    }
    
    return {
        "status": "healthy" if all([services_status["connector"], services_status["predictor"]]) else "degraded",
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
async def predict_next_candle(params: PredictionParams = None):
    """Predict the next candle direction"""
    global connector, predictor, market_data
    
    if params is None:
        params = PredictionParams()
    
    # Check if services are initialized
    if connector is None or predictor is None:
        logger.error("Services not initialized, attempting to initialize now")
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    # Check if we have market data
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not available, please fetch first")
    
    try:
        # Convert market data back to DataFrame
        df = pd.DataFrame(market_data["candles"])
        
        # Debug information
        logger.info(f"Making prediction with data shape: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        logger.info(f"First few rows: {df.head(2).to_dict()}")
        
        # Convert date strings back to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Check if required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Data is missing required columns: {missing_columns}")
        
        # Initialize predictor with the specific model
        logger.info("Initializing model predictor...")
        
        # Use the proper ModelPredictor class instead of a custom one
        from model_predictor import ModelPredictor
        
        # Set up paths
        model_dir = "C:/Users/Botmudra11/Desktop/project/XAUUSD-predictor/best_model"
        model_name = "xauusd_model_adaboost_none_config1_20250330_114015_20250330_114017.joblib"
        processor_path = os.path.join("best_model", "xauusd_model_processor.pkl")
        
        # Create the predictor using the proper class
        predictor = ModelPredictor(
            model_dir=model_dir,
            model_name=model_name,
            processor_path=processor_path,
            threshold=0.5
        )
        
        # Make prediction using the predictor
        logger.info("Calling predictor.predict...")
        
        # Try to get model info first to verify the model is loaded properly
        try:
            model_info = predictor.get_model_info()
            logger.info(f"Model info: {model_info}")
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            
        # Make the prediction
        prediction = predictor.predict(df)
        logger.info(f"Prediction result: {prediction}")
        
        # Check if there was an error in prediction
        if isinstance(prediction, dict) and "error" in prediction:
            logger.error(f"Prediction error: {prediction['error']}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {prediction['error']}")
        
        # Add timestamp
        prediction["timestamp"] = datetime.now().isoformat()
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{error_traceback}")

@app.get("/validate-model")
async def validate_model():
    """Validate model files and paths"""
    
    # Define expected paths
    model_dir = "best_model"
    model_name = "xauusd_model_adaboost_none_config1_20250330_114015_20250330_114017"
    processor_path = os.path.join("best_model", "xauusd_model_processor.pkl")
    
    results = {
        "model_dir_exists": os.path.exists(model_dir),
        "processor_file_exists": os.path.exists(processor_path),
        "model_files": [],
        "errors": []
    }
    
    # Check for model files with different extensions
    for ext in ['.joblib', '.pkl']:
        model_path = os.path.join(model_dir, f"{model_name}{ext}")
        if os.path.exists(model_path):
            results["model_files"].append({
                "path": model_path,
                "size_bytes": os.path.getsize(model_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            })
    
    # Check for metadata file
    for ext in ['.joblib', '.pkl']:
        metadata_path = os.path.join(model_dir, f"{model_name}{ext.replace('.', '_')}_metadata.json")
        if os.path.exists(metadata_path):
            results["metadata_file"] = {
                "path": metadata_path,
                "size_bytes": os.path.getsize(metadata_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(metadata_path)).isoformat()
            }
            
            # Try to load metadata
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                results["metadata_content"] = metadata
            except Exception as e:
                results["errors"].append(f"Failed to load metadata: {str(e)}")
    
    # Try to load processor
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        results["processor_load_attempt"] = processor.load_processor_state(processor_path)
        
        if results["processor_load_attempt"]:
            results["processor_features"] = {
                "selected_features": processor.selected_features[:10] if processor.selected_features else None,
                "expected_features": processor.expected_features[:10] if processor.expected_features else None,
                "feature_count": len(processor.expected_features) if processor.expected_features else 0
            }
    except Exception as e:
        results["errors"].append(f"Failed to load processor: {str(e)}")
    
    # Try to load model
    try:
        import joblib
        import pickle
        
        model_loaded = False
        
        # Try each model file found
        for model_file in results["model_files"]:
            model_path = model_file["path"]
            try:
                if model_path.endswith('.joblib'):
                    model_obj = joblib.load(model_path)
                    results["model_load_success"] = True
                    results["model_load_method"] = "joblib"
                else:
                    with open(model_path, 'rb') as f:
                        model_obj = pickle.load(f)
                    results["model_load_success"] = True
                    results["model_load_method"] = "pickle"
                
                # Check model properties
                results["model_attributes"] = dir(model_obj)
                model_loaded = True
                break
            except Exception as e:
                results["errors"].append(f"Failed to load model {model_path}: {str(e)}")
        
        if not model_loaded:
            results["model_load_success"] = False
            
    except Exception as e:
        results["errors"].append(f"Error in model loading process: {str(e)}")
    
    return results

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    global predictor
    
    if predictor is None:
        if not init_services():
            raise HTTPException(status_code=500, detail="Failed to initialize services")
    
    try:
        # Get model info
        model_info = predictor.get_model_info()
        
        return {
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)