# XAUUSD Prediction Application

This project provides an end-to-end solution for predicting XAUUSD (Gold) price movements using a FastAPI backend and a Streamlit-based frontend dashboard. The backend fetches market data, processes it, and serves predictions, while the frontend visualizes the data and predictions interactively.

## Features
- **FastAPI Backend**: Handles data fetching, processing, and predictions.
- **Streamlit Frontend**: Provides an interactive UI for visualization and user interaction.
- **MT5 Connector**: Fetches real-time market data.
- **Automated Deployment**: Starts both backend and frontend using a single script.

## Installation

### Prerequisites
- Python 3.8+
- MetaTrader 5 installed (if using MT5 for data)
- Virtual environment (optional but recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MihirsinhChauhan/XAUUSD-predictor.git
   cd XAUUSD-predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the Application
Run the following command to start both the FastAPI backend and the Streamlit frontend:
```bash
python main.py
```
Once running, you can access:
- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)

### API Endpoints
- `GET /health` - Check system status
- `GET /market-data` - Fetch market data
- `POST /predict` - Get price predictions
- `POST /reconnect` - Reconnect to MT5

## Troubleshooting
### API Not Responding
- Ensure `uvicorn` is running (`http://localhost:8000/health` should return a status check).
- Check logs in `api.log`.

### Streamlit UI Not Showing Graph
- Ensure the backend is running and returning market data (`http://localhost:8000/market-data`).
- Add `st.write(response.json())` in `streamlit_ui.py` for debugging.

## Contributing
Feel free to open issues and pull requests to improve this project.

## License
This project is licensed under the MIT License.

