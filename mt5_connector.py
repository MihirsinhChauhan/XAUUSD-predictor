import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

class MT5Connector:
    def __init__(self, login=79700174, server="Exness-MT5Trial8", password="Botmudra.com@01"):
        self.login = login
        self.server = server
        self.password = password
        self.connected = False
    
    def connect(self):
        """Connect to MT5 platform"""
        if not mt5.initialize(server=self.server, login=self.login, password=self.password):
            print(f"Failed to connect to MT5: {mt5.last_error()}")
            return False
        else:
            print("Connected to MT5 successfully")
            self.connected = True
            return True
    
    def disconnect(self):
        """Disconnect from MT5 platform"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MT5")
    
    def fetch_data(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, num_candles=1000):
        """Fetch historical data from MT5"""
        if not self.connected:
            if not self.connect():
                return None
        
        # Check if symbol exists
        if not mt5.symbol_info(symbol):
            print(f"{symbol} symbol not found")
            return None
        
        # Fetch the candle data
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to fetch {symbol} data")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time from timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def save_data_to_csv(self, df, filename="xauusd_M5.csv"):
        """Save the data to CSV file"""
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        return False

# Test the connector
if __name__ == "__main__":
    connector = MT5Connector()
    if connector.connect():
        df = connector.fetch_data()
        if df is not None:
            print(df.head())
            connector.save_data_to_csv(df)
        connector.disconnect()