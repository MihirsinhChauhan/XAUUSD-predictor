import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import os
import platform

class MT5Connector:
    def __init__(self, login=79700174, server="Exness-MT5Trial8", password="Botmudra.com@01"):
        self.login = login
        self.server = server
        self.password = password
        self.connected = False
        self.terminal_path = self._get_default_terminal_path()
   
    def _get_default_terminal_path(self):
        """Get the default path to MT5 terminal based on OS"""
        if platform.system() == "Windows":
            # Common paths for MT5 on Windows
            possible_paths = [
                "C:/Program Files/MetaTrader 5/terminal64.exe",
                "C:/Program Files (x86)/MetaTrader 5/terminal.exe",
                # Add path for custom installation if needed
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            
            return "C:/Program Files/MetaTrader 5/terminal64.exe"  # Default fallback
        else:
            return ""  # Non-Windows OS not directly supported by MT5
   
    def connect(self, max_retries=3, retry_delay=5):
        """Connect to MT5 platform with retries"""
        print("MetaTrader5 package author: ", mt5.__author__)
        print("MetaTrader5 package version: ", mt5.__version__)
        
        # Check if MT5 is already initialized
        if mt5.terminal_info() is not None:
            print("MT5 is already initialized")
            self.connected = True
            return True
            
        # Check if terminal path exists
        if not os.path.exists(self.terminal_path):
            print(f"Warning: MT5 terminal not found at {self.terminal_path}")
            print("Please make sure MetaTrader 5 is installed and the path is correct.")
        
        for attempt in range(max_retries):
            print(f"Connection attempt {attempt+1}/{max_retries}...")
            
            try:
                # Try to initialize MT5
                if not mt5.initialize(
                    path=self.terminal_path,
                    server=self.server,
                    login=self.login,
                    password=self.password,
                    timeout=120000):
                    
                    error_code, error_desc = mt5.last_error()
                    print(f"Failed to connect to MT5: ({error_code}, '{error_desc}')")
                    
                    # Specific error handling
                    if error_code == -10005:  # IPC Timeout
                        print("IPC Timeout Error. Possible causes:")
                        print("1. MT5 terminal is not running. Please start it manually.")
                        print("2. MT5 terminal path is incorrect.")
                        print("3. Login credentials may be incorrect.")
                        print("4. Firewall might be blocking the connection.")
                    
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                else:
                    print("Connected to MT5 successfully")
                    account_info = mt5.account_info()
                    if account_info is not None:
                        print(f"Account: {account_info.login} ({account_info.server})")
                        print(f"Balance: {account_info.balance} {account_info.currency}")
                    self.connected = True
                    return True
            except Exception as e:
                print(f"Error during connection: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        print("All connection attempts failed")
        return False
   
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
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"{symbol} symbol not found")
            print("Available symbols:")
            symbols = mt5.symbols_get()
            for s in symbols[:10]:  # Show first 10 available symbols
                print(f"- {s.name}")
            return None
       
        # Enable the symbol if needed
        if not symbol_info.visible:
            print(f"Enabling symbol {symbol} for use")
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to enable {symbol}")
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
    
    # You can set the path manually if needed
    # connector.terminal_path = "C:/Your/Custom/Path/To/terminal64.exe"
    
    if connector.connect():
        df = connector.fetch_data()
        if df is not None:
            print(df.head())
            connector.save_data_to_csv(df)
        connector.disconnect()