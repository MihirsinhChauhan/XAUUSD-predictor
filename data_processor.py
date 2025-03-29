import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataProcessor')

class DataProcessor:
    """
    data processor for preparing XAUUSD trading data for machine learning.
    Features:
    - Technical indicators implemented with pandas/numpy
    - Advanced feature engineering
    - Data cleaning and preprocessing
    - Feature scaling and selection
    - Market-specific features for gold trading
    """
    
    def __init__(self, 
                 scale_features: bool = False, 
                 scaler_type: str = 'robust',
                 feature_selection: bool = False,
                 n_features: int = 20,
                 handle_missing: str = 'drop',
                 smooth_outliers: bool = False):
        """
        Initialize the DataProcessor with configuration options.
        
        Args:
            scale_features: Whether to scale numerical features
            scaler_type: Type of scaler ('standard' or 'robust')
            feature_selection: Whether to perform automatic feature selection
            n_features: Number of top features to select if feature_selection is True
            handle_missing: How to handle missing data ('drop', 'fill_mean', 'fill_median', 'fill_zero')
            smooth_outliers: Whether to smooth extreme values
        """
        self.scale_features = scale_features
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.handle_missing = handle_missing
        self.smooth_outliers = smooth_outliers
        
        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.training_features = None  # Track all features available during training
        self.expected_features = None  # Track features expected by the scaler
        
        # Store column meanings for later reference
        self.feature_descriptions = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for machine learning.
        
        Parameters:
            df: DataFrame with OHLCV data (columns: open, high, low, close, tick_volume)
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        dataset = df.copy()
        
        # Ensure datetime format for time column if it exists
        if 'time' in dataset.columns:
            if not pd.api.types.is_datetime64_any_dtype(dataset['time']):
                dataset['time'] = pd.to_datetime(dataset['time'])
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # --- 1. Basic Price Features ---
        logger.info("Creating basic price features...")
        self._add_basic_price_features(dataset)
        
        # --- 2. Previous Candle Features ---
        logger.info("Creating previous candle features...")
        self._add_previous_candle_features(dataset)
        
        # --- 3. Technical Indicators ---
        logger.info("Creating technical indicators...")
        self._add_technical_indicators(dataset)
        
        # --- 4. Volatility Indicators ---
        logger.info("Creating volatility indicators...")
        self._add_volatility_indicators(dataset)
        
        # --- 5. Volume Indicators ---
        logger.info("Creating volume indicators...")
        self._add_volume_indicators(dataset)
        
        # --- 6. Price Patterns ---
        logger.info("Creating price pattern features...")
        self._add_price_patterns(dataset)
        
        # --- 7. Market Session & Time Features ---
        if 'time' in dataset.columns:
            logger.info("Creating time-based features...")
            self._add_time_features(dataset)
        
        # --- 8. Gold-Specific Features ---
        logger.info("Creating gold-specific indicators...")
        self._add_gold_specific_features(dataset)
        
        # --- 9. Handle Missing Data ---
        logger.info("Handling missing data...")
        dataset = self._handle_missing_data(dataset)
        
        # --- 10. Outlier Handling ---
        if self.smooth_outliers:
            logger.info("Smoothing outliers...")
            dataset = self._smooth_outliers(dataset)
        
        return dataset
    
    def _add_basic_price_features(self, df: pd.DataFrame) -> None:
        """Add basic price action features."""
        # Price changes
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_diff'] = df['high'] - df['low']
        df['high_low_range_pct'] = (df['high'] - df['low']) / df['open'] * 100
        
        # Candle direction and size
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_to_range_ratio'] = df['body_size'] / df['high_low_diff'].replace(0, np.nan)
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'price_change': 'Difference between close and open prices',
            'price_change_pct': 'Percentage change from open to close',
            'high_low_diff': 'Range between high and low prices',
            'high_low_range_pct': 'High-low range as percentage of open price',
            'is_bullish': 'Binary indicator if candle closed above its open (1=bullish)',
            'body_size': 'Size of candle body (absolute difference between open and close)',
            'body_to_range_ratio': 'Ratio of body size to full candle range',
            'upper_shadow': 'Size of upper shadow/wick',
            'lower_shadow': 'Size of lower shadow/wick'
        })
    
    def _add_previous_candle_features(self, df: pd.DataFrame) -> None:
        """Add features related to previous candles."""
        # Previous candle basic features
        for i in range(1, 4):  # Previous 3 candles
            suffix = f'_{i}' if i > 1 else ''
            
            df[f'prev{suffix}_close'] = df['close'].shift(i)
            df[f'prev{suffix}_open'] = df['open'].shift(i)
            df[f'prev{suffix}_high'] = df['high'].shift(i)
            df[f'prev{suffix}_low'] = df['low'].shift(i)
            df[f'prev{suffix}_volume'] = df['tick_volume'].shift(i) if 'tick_volume' in df.columns else None
            df[f'prev{suffix}_price_change'] = df['price_change'].shift(i)
            df[f'prev{suffix}_is_bullish'] = df['is_bullish'].shift(i)
            
            # Relationship to previous candle
            if i == 1:
                df['close_vs_prev_close'] = (df['close'] > df['prev_close']).astype(int)
                df['close_vs_prev_high'] = (df['close'] > df['prev_high']).astype(int)
                df['close_vs_prev_low'] = (df['close'] < df['prev_low']).astype(int)
                df['high_vs_prev_high'] = (df['high'] > df['prev_high']).astype(int)
                df['low_vs_prev_low'] = (df['low'] < df['prev_low']).astype(int)
                
                # Inside/Outside bars
                df['is_inside_bar'] = ((df['high'] <= df['prev_high']) & 
                                     (df['low'] >= df['prev_low'])).astype(int)
                df['is_outside_bar'] = ((df['high'] >= df['prev_high']) & 
                                      (df['low'] <= df['prev_low'])).astype(int)
            
        # Consecutive candles
        df['consecutive_bullish'] = ((df['is_bullish'] == 1) & 
                                    (df['prev_is_bullish'] == 1)).astype(int)
        df['consecutive_bearish'] = ((df['is_bullish'] == 0) & 
                                    (df['prev_is_bullish'] == 0)).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'prev_close': 'Close price of previous candle',
            'prev_open': 'Open price of previous candle',
            'close_vs_prev_close': 'Binary indicator if close is higher than previous close',
            'is_inside_bar': 'Binary indicator of inside bar pattern',
            'is_outside_bar': 'Binary indicator of outside bar pattern',
            'consecutive_bullish': 'Binary indicator of two consecutive bullish candles',
            'consecutive_bearish': 'Binary indicator of two consecutive bearish candles'
        })

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        
        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA (Exponential Weighted Moving Average)
        avg_gain = up.ewm(com=period-1, min_periods=period).mean()
        avg_loss = down.ewm(com=period-1, min_periods=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        middle = self._calculate_sma(series, period)
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (macd line, signal line, histogram)"""
        fast_ema = self._calculate_ema(series, fast_period)
        slow_ema = self._calculate_ema(series, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Fast %K
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Slow %K (or commonly known as %K)
        k = k.rolling(window=d_period).mean()
        
        # %D is the 3-period MA of %K
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        
        # True Range is the greatest of:
        # 1. Current High - Current Low
        # 2. |Current High - Previous Close|
        # 3. |Current Low - Previous Close|
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the simple moving average of TR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # This is a simplified ADX implementation
        # First, calculate TR and ATR
        atr = self._calculate_atr(high, low, close, period)
        
        # Calculate the +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff().abs() * -1  # Make negative for comparison
        
        plus_dm = ((high_diff > 0) & (high_diff > low_diff.abs())).astype(int) * high_diff
        minus_dm = ((low_diff < 0) & (low_diff.abs() > high_diff)).astype(int) * low_diff.abs()
        
        # Calculate +DI and -DI
        plus_di = 100 * self._calculate_ema(plus_dm, period) / atr
        minus_di = 100 * self._calculate_ema(minus_dm, period) / atr
        
        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = self._calculate_ema(dx, period)
        
        return adx
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        close_diff = close.diff()
        
        # Create a series where 1 = price up, -1 = price down, 0 = no change
        direction = pd.Series(0, index=close.index)
        direction[close_diff > 0] = 1
        direction[close_diff < 0] = -1
        
        # Multiply volume by direction and calculate cumulative sum
        obv = (volume * direction).cumsum()
        
        return obv
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate money flow
        money_flow = typical_price * volume
        
        # Get the direction
        direction = np.where(typical_price > typical_price.shift(1), 1, -1)
        
        # Separate positive and negative money flow
        positive_flow = pd.Series(0, index=money_flow.index)
        negative_flow = pd.Series(0, index=money_flow.index)
        
        positive_flow[direction > 0] = money_flow[direction > 0]
        negative_flow[direction < 0] = money_flow[direction < 0]
        
        # Sum positive and negative money flow over period
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add comprehensive technical indicators using pandas/numpy."""
        # --- Moving Averages ---
        for period in [5, 10, 20, 50, 100]:
            # Simple Moving Average
            df[f'sma_{period}'] = self._calculate_sma(df['close'], period)
            
            # Exponential Moving Average
            df[f'ema_{period}'] = self._calculate_ema(df['close'], period)
            
            # MA Direction
            df[f'sma_{period}_direction'] = (df[f'sma_{period}'] > df[f'sma_{period}'].shift(1)).astype(int)
            df[f'ema_{period}_direction'] = (df[f'ema_{period}'] > df[f'ema_{period}'].shift(1)).astype(int)
            
            # Price relative to MA
            df[f'close_over_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
            df[f'close_over_ema_{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
            
            # Distance from MA
            df[f'close_to_sma_{period}_pct'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
            df[f'close_to_ema_{period}_pct'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
        
        # --- Oscillators ---
        # RSI - Relative Strength Index
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        
        # RSI conditions
        df['rsi_14_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_14_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_14_direction'] = (df['rsi_14'] > df['rsi_14'].shift(1)).astype(int)
        
        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_hist_direction'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)
        
        # Stochastic Oscillator
        slowk, slowd = self._calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        df['stoch_oversold'] = ((df['stoch_k'] < 20) & (df['stoch_d'] < 20)).astype(int)
        df['stoch_overbought'] = ((df['stoch_k'] > 80) & (df['stoch_d'] > 80)).astype(int)
        df['stoch_k_above_d'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        
        # --- Other Indicators ---
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_close_upper_dist'] = (df['bb_upper'] - df['close']) / df['close'] * 100
        df['bb_close_lower_dist'] = (df['close'] - df['bb_lower']) / df['close'] * 100
        
        df['close_above_bb_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['close_below_bb_lower'] = (df['close'] < df['bb_lower']).astype(int)
        df['close_within_bb'] = ((df['close'] <= df['bb_upper']) & (df['close'] >= df['bb_lower'])).astype(int)
        
        # Awesome Oscillator (simplified version)
        df['ao'] = (self._calculate_sma((df['high'] + df['low']) / 2, 5) - 
                  self._calculate_sma((df['high'] + df['low']) / 2, 34))
        df['ao_direction'] = (df['ao'] > df['ao'].shift(1)).astype(int)
        
        # Moving Average of Oscillator (OsMA)
        df['osma'] = df['macd'] - df['macd_signal']
        df['osma_direction'] = (df['osma'] > df['osma'].shift(1)).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'sma_20': '20-period Simple Moving Average',
            'ema_20': '20-period Exponential Moving Average',
            'sma_20_direction': 'Direction of SMA (1=up, 0=down)',
            'close_over_sma_20': 'Binary indicator if close is above SMA',
            'rsi_14': '14-period Relative Strength Index',
            'rsi_14_oversold': 'Binary indicator if RSI is in oversold territory (<30)',
            'rsi_14_overbought': 'Binary indicator if RSI is in overbought territory (>70)',
            'macd': 'MACD line (12,26,9)',
            'macd_signal': 'MACD signal line',
            'macd_hist': 'MACD histogram',
            'macd_above_signal': 'Binary indicator if MACD is above signal line',
            'stoch_k': 'Stochastic oscillator %K line',
            'stoch_d': 'Stochastic oscillator %D line',
            'bb_upper': 'Upper Bollinger Band (20,2)',
            'bb_middle': 'Middle Bollinger Band (20-period SMA)',
            'bb_lower': 'Lower Bollinger Band (20,2)',
            'bb_width': 'Bollinger Band width as percentage of middle band',
            'close_above_bb_upper': 'Binary indicator if price closed above upper Bollinger Band',
            'close_below_bb_lower': 'Binary indicator if price closed below lower Bollinger Band'
        })
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> None:
        """Add volatility-based indicators."""
        # ATR - Average True Range
        df['atr_14'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
        df['atr_percent'] = df['atr_14'] / df['close'] * 100
        
        # Volatility over different periods
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['high_low_diff'].rolling(window=period).std()
            df[f'close_volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * 100
        
        # Normalized volatility
        df['normalized_volatility'] = df['volatility_10'] / df['close'] * 100
        
        # Volatility change
        df['volatility_change'] = df['volatility_10'] - df['volatility_10'].shift(5)
        df['volatility_change_pct'] = df['volatility_change'] / df['volatility_10'].shift(5) * 100
        
        # ADX - Average Directional Index (trend strength)
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'], 14)
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
        df['adx_very_strong_trend'] = (df['adx'] > 50).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'atr_14': '14-period Average True Range',
            'atr_percent': 'ATR as percentage of closing price',
            'volatility_10': 'Standard deviation of high-low range over 10 periods',
            'close_volatility_10': 'Standard deviation of percentage price changes over 10 periods',
            'normalized_volatility': 'Volatility normalized by price level',
            'adx': '14-period Average Directional Index (trend strength)',
            'adx_strong_trend': 'Binary indicator if ADX shows strong trend (>25)',
            'adx_very_strong_trend': 'Binary indicator if ADX shows very strong trend (>50)'
        })
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Add volume-based indicators if volume data is available."""
        if 'tick_volume' not in df.columns:
            logger.warning("Volume data not found, skipping volume indicators")
            return
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['tick_volume'].rolling(window=period).mean()
            
            # Volume relative to moving average
            df[f'volume_ratio_{period}'] = df['tick_volume'] / df[f'volume_sma_{period}']
        
        # Volume changes
        df['volume_change'] = df['tick_volume'] - df['tick_volume'].shift(1)
        df['volume_change_pct'] = df['volume_change'] / df['tick_volume'].shift(1) * 100
        
        # Volume trends
        df['volume_trend'] = (df['tick_volume'] > df['tick_volume'].shift(1)).astype(int)
        df['rising_volume'] = ((df['volume_trend'] == 1) & (df['volume_trend'].shift(1) == 1)).astype(int)
        df['falling_volume'] = ((df['volume_trend'] == 0) & (df['volume_trend'].shift(1) == 0)).astype(int)
        
        # Volume and price relationship
        df['volume_price_trend'] = ((df['is_bullish'] == 1) & (df['volume_trend'] == 1)).astype(int)
        df['price_up_volume_up'] = ((df['is_bullish'] == 1) & (df['volume_change'] > 0)).astype(int)
        df['price_down_volume_up'] = ((df['is_bullish'] == 0) & (df['volume_change'] > 0)).astype(int)
        
        # Relative Volume
        df['relative_volume_10'] = df['tick_volume'] / df['tick_volume'].rolling(window=10).mean()
        df['high_volume'] = (df['relative_volume_10'] > 1.5).astype(int)
        df['low_volume'] = (df['relative_volume_10'] < 0.5).astype(int)
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df['close'], df['tick_volume'])
        df['obv_direction'] = (df['obv'] > df['obv'].shift(1)).astype(int)
        
        # Money Flow Index (MFI)
        if 'tick_volume' in df.columns:
            df['mfi'] = self._calculate_mfi(df['high'], df['low'], df['close'], df['tick_volume'], 14)
            df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
            df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'volume_sma_10': '10-period Simple Moving Average of volume',
            'volume_ratio_10': 'Ratio of current volume to its 10-period SMA',
            'volume_change': 'Change in volume from previous candle',
            'volume_trend': 'Binary indicator if volume is higher than previous candle',
            'rising_volume': 'Binary indicator of two consecutive volume increases',
            'price_up_volume_up': 'Binary indicator of price rise with volume increase',
            'obv': 'On-Balance Volume indicator',
            'obv_direction': 'Direction of OBV (1=up, 0=down)',
            'mfi': '14-period Money Flow Index',
            'mfi_oversold': 'Binary indicator if MFI is in oversold territory (<20)'
        })
    
    def _add_price_patterns(self, df: pd.DataFrame) -> None:
        """Add basic candlestick pattern features without TA-Lib."""
        # Manually identify some common candlestick patterns
        
        # Doji pattern (open and close are very close)
        body_to_range = df['body_size'] / df['high_low_diff'].replace(0, np.nan)
        df['pattern_doji'] = (body_to_range < 0.1).astype(int)
        
        # Hammer pattern (small body at the top, long lower shadow)
        df['pattern_hammer'] = (
            (df['body_size'] / df['high_low_diff'] < 0.3) & 
            (df['lower_shadow'] > 2 * df['body_size']) & 
            (df['upper_shadow'] < 0.1 * df['high_low_diff'])
        ).astype(int)
        
        # Shooting Star (small body at the bottom, long upper shadow)
        df['pattern_shooting_star'] = (
            (df['body_size'] / df['high_low_diff'] < 0.3) & 
            (df['upper_shadow'] > 2 * df['body_size']) & 
            (df['lower_shadow'] < 0.1 * df['high_low_diff'])
        ).astype(int)
        
        # Engulfing patterns (current candle's body engulfs previous candle's body)
        bull_engulf = (
            (df['is_bullish'] == 1) & 
            (df['prev_is_bullish'] == 0) & 
            (df['open'] < df['prev_close']) & 
            (df['close'] > df['prev_open'])
        )
        
        bear_engulf = (
            (df['is_bullish'] == 0) & 
            (df['prev_is_bullish'] == 1) & 
            (df['open'] > df['prev_close']) & 
            (df['close'] < df['prev_open'])
        )
        
        df['pattern_engulfing'] = (bull_engulf | bear_engulf).astype(int)
        df['pattern_bull_engulfing'] = bull_engulf.astype(int)
        df['pattern_bear_engulfing'] = bear_engulf.astype(int)
        
        # Spinning top (small body with upper and lower shadows)
        df['pattern_spinning_top'] = (
            (df['body_size'] / df['high_low_diff'] < 0.3) & 
            (df['upper_shadow'] > df['body_size']) & 
            (df['lower_shadow'] > df['body_size'])
        ).astype(int)
        
        # Marubozu (no or very small shadows)
        df['pattern_marubozu'] = (
            (df['upper_shadow'] < 0.05 * df['high_low_diff']) & 
            (df['lower_shadow'] < 0.05 * df['high_low_diff']) & 
            (df['body_size'] / df['high_low_diff'] > 0.9)
        ).astype(int)
        
        # Composite pattern features
        df['bullish_pattern'] = ((df['pattern_hammer'] == 1) | 
                               (df['pattern_bull_engulfing'] == 1)).astype(int)
        
        df['bearish_pattern'] = ((df['pattern_shooting_star'] == 1) | 
                               (df['pattern_bear_engulfing'] == 1)).astype(int)
        
        df['indecision_pattern'] = ((df['pattern_doji'] == 1) | 
                                  (df['pattern_spinning_top'] == 1)).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'pattern_doji': 'Doji pattern detection (body very small compared to range)',
            'pattern_hammer': 'Hammer pattern detection (small body at top, long lower shadow)',
            'pattern_engulfing': 'Engulfing pattern detection (current body engulfs previous)',
            'pattern_bull_engulfing': 'Bullish engulfing pattern detection',
            'pattern_bear_engulfing': 'Bearish engulfing pattern detection',
            'bullish_pattern': 'Composite indicator of any bullish pattern detected',
            'bearish_pattern': 'Composite indicator of any bearish pattern detected',
            'indecision_pattern': 'Composite indicator of any indecision pattern detected'
        })
    
    def _add_time_features(self, df: pd.DataFrame) -> None:
        """Add time-based features if timestamp data is available."""
        # Extract time components
        df['day_of_week'] = df['time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['hour_of_day'] = df['time'].dt.hour
        
        # Trading session indicators (rough approximation)
        df['asian_session'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 8)).astype(int)
        df['european_session'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] < 16)).astype(int)
        df['american_session'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] < 21)).astype(int)
        df['overlap_session'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] < 16)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
            'hour_of_day': 'Hour of day (0-23)',
            'asian_session': 'Binary indicator of Asian trading session',
            'european_session': 'Binary indicator of European trading session',
            'american_session': 'Binary indicator of American trading session',
            'overlap_session': 'Binary indicator of session overlap period',
            'is_weekend': 'Binary indicator if day is weekend (Sat/Sun)'
        })
    
    def _add_gold_specific_features(self, df: pd.DataFrame) -> None:
        """Add gold-specific trading indicators and features."""
        # Gold volatility features
        if 'volatility_10' in df.columns and 'sma_20' in df.columns:
            df['gold_volatility_ratio'] = df['volatility_10'] / df['sma_20'] * 100
        
        # Gold-specific price levels (examples based on typical gold behavior)
        df['close_round_level'] = (abs(df['close'] % 10) < 0.5).astype(int)  # Near round numbers
        df['close_round_25'] = (abs(df['close'] % 25) < 1.0).astype(int)  # Near 25's
        df['close_round_50'] = (abs(df['close'] % 50) < 1.0).astype(int)  # Near 50's
        df['close_round_100'] = (abs(df['close'] % 100) < 1.5).astype(int)  # Near 100's
        
        # Rate of Change
        df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Historical gold behavior patterns
        df['higher_high'] = ((df['high'] > df['prev_high']) & 
                           (df['prev_high'] > df['prev_high'].shift(1))).astype(int)
        df['lower_low'] = ((df['low'] < df['prev_low']) & 
                         (df['prev_low'] < df['prev_low'].shift(1))).astype(int)
        df['higher_high_lower_low'] = ((df['higher_high'] == 1) & 
                                     (df['lower_low'] == 1)).astype(int)  # Expanding volatility
        
        # Add feature descriptions
        self.feature_descriptions.update({
            'gold_volatility_ratio': 'Gold volatility relative to price level',
            'close_round_level': 'Binary indicator if price is near a round number',
            'roc_5': '5-period Rate of Change',
            'higher_high': 'Binary indicator of consecutive higher highs',
            'lower_low': 'Binary indicator of consecutive lower lows',
            'higher_high_lower_low': 'Binary indicator of expanding price range (volatility)'
        })
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data based on configured method."""
        # Count missing values before processing
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"Missing values before handling: {missing_count[missing_count > 0].to_dict()}")
        
        # Apply handling method
        if self.handle_missing == 'drop':
            original_len = len(df)
            df = df.dropna()
            logger.info(f"Dropped {original_len - len(df)} rows with missing values")
            
        elif self.handle_missing == 'fill_mean':
            df = df.fillna(df.mean())
            logger.info("Filled missing values with column means")
            
        elif self.handle_missing == 'fill_median':
            df = df.fillna(df.median())
            logger.info("Filled missing values with column medians")
            
        elif self.handle_missing == 'fill_zero':
            df = df.fillna(0)
            logger.info("Filled missing values with zeros")
            
        elif self.handle_missing == 'ffill':
            df = df.fillna(method='ffill')
            logger.info("Filled missing values using forward fill method")
        
        return df
    
    def _smooth_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Smooth extreme values (outliers) in numerical columns.
        
        Args:
            df: DataFrame to process
            threshold: Z-score threshold to identify outliers
            
        Returns:
            DataFrame with smoothed outliers
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            # Skip binary columns
            if set(df[col].unique()).issubset({0, 1, np.nan}):
                continue
                
            # Calculate z-scores
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val == 0:  # Skip constant columns
                continue
                
            z_scores = (df[col] - mean_val) / std_val
            
            # Identify outliers
            outliers = abs(z_scores) > threshold
            
            if outliers.sum() > 0:
                logger.info(f"Smoothing {outliers.sum()} outliers in column {col}")
                
                # Cap outliers at threshold * std
                df.loc[z_scores > threshold, col] = mean_val + threshold * std_val
                df.loc[z_scores < -threshold, col] = mean_val - threshold * std_val
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for machine learning by creating features and labels.
        
        Parameters:
            df: DataFrame with OHLCV data
            
        Returns:
            tuple: (X, y, feature_list) where X is features, y is target, feature_list is feature names
        """
        logger.info("Preparing data for machine learning...")
        
        # Create all features
        dataset = self.prepare_features(df)
        
        # Create target: Next candle direction (1 for bullish, 0 for bearish)
        dataset['next_is_bullish'] = dataset['is_bullish'].shift(-1)
        
        # Drop NaN values (important to do this after creating target)
        original_len = len(dataset)
        dataset = dataset.dropna()
        logger.info(f"Dropped {original_len - len(dataset)} rows with missing values")

        self.training_features = dataset.columns.tolist()
        # Select features for prediction (using a comprehensive set of the created features)
        base_features = [
            # Price action features
            'price_change', 'price_change_pct', 'high_low_diff', 'body_size', 
            'body_to_range_ratio', 'upper_shadow', 'lower_shadow', 'is_bullish',
            
            # Previous candle relationships
            'prev_price_change', 'prev_is_bullish', 'close_vs_prev_close',
            'close_vs_prev_high', 'close_vs_prev_low', 'is_inside_bar', 
            'is_outside_bar', 'consecutive_bullish', 'consecutive_bearish',
            
            # Moving averages
            'close_over_sma_20', 'close_over_ema_10', 'sma_5_direction', 
            'sma_20_direction', 'ema_5_direction', 'ema_20_direction',
            'close_to_sma_20_pct', 'close_to_ema_20_pct',
            
            # Oscillators
            'rsi_14', 'rsi_14_oversold', 'rsi_14_overbought', 'rsi_14_direction',
            'macd_above_signal', 'macd_hist_direction',
            'stoch_k_above_d', 'stoch_oversold', 'stoch_overbought',
            
            # Volatility indicators
            'atr_percent', 'volatility_10', 'normalized_volatility',
            'adx', 'adx_strong_trend',
            
            # Bollinger Bands
            'bb_width', 'close_above_bb_upper', 'close_below_bb_lower',
            
            # Patterns
            'bullish_pattern', 'bearish_pattern', 'indecision_pattern',
            
            # Gold-specific
            'gold_volatility_ratio', 'close_round_level', 'roc_5',
            'higher_high', 'lower_low'
        ]
        
        # Add time features if available
        if 'hour_of_day' in dataset.columns:
            time_features = [
                'asian_session', 'european_session', 'american_session',
                'overlap_session', 'is_weekend'
            ]
            base_features.extend(time_features)
        
        # Add volume features if available
        if 'tick_volume' in dataset.columns:
            volume_features = [
                'volume_ratio_10', 'volume_change_pct', 'volume_trend',
                'price_up_volume_up', 'price_down_volume_up', 'high_volume',
                'obv_direction'
            ]
            if 'mfi' in dataset.columns:
                volume_features.extend(['mfi', 'mfi_oversold', 'mfi_overbought'])
                
            base_features.extend(volume_features)
        
        # Ensure all features exist in the dataset
        feature_list = [f for f in base_features if f in dataset.columns]
        logger.info(f"Selected {len(feature_list)} features for model training")
        
        # Extract features and target
        X = dataset[feature_list].copy()
        y = dataset['next_is_bullish'].copy()

        self.expected_features = X.columns.tolist()
        # Apply scaling if configured
        if self.scale_features:
            X = self._scale_features(X)
        
        # Apply feature selection if configured
        if self.feature_selection:
            X, feature_list = self._select_features(X, y, feature_list)
        
        return X, y, feature_list
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        # Create a new scaler if not already created
        if self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:  # robust by default
                self.scaler = RobustScaler()
        
        # Fit and transform the data
        scaled_data = self.scaler.fit_transform(X)
        
        # Convert back to DataFrame
        X_scaled = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        
        logger.info(f"Scaled features using {self.scaler_type} scaler")
        return X_scaled
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, 
                       feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most important features based on statistical tests.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_list: Original feature list
            
        Returns:
            tuple: (X_selected, selected_feature_list)
        """
        # Create a new feature selector if not already created
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
            
        # Fit and transform the data
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = self.feature_selector.get_support(indices=True)
        
        # Update feature list to include only selected features
        selected_feature_list = [feature_list[i] for i in selected_indices]
        
        # Store for future reference
        self.selected_features = selected_feature_list
        
        # Convert back to DataFrame
        X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_list, index=X.index)
        
        logger.info(f"Selected {len(selected_feature_list)} most important features")
        
        return X_selected_df, selected_feature_list
    
    def prepare_latest_for_prediction(self, df, feature_list=None):
        """
        Prepare the latest data point for prediction with feature consistency.
        """
        logger.info("Preparing latest data for prediction...")
        
        # Use the appropriate feature list
        if feature_list is None:
            if self.selected_features is not None:
                feature_list = self.selected_features
            elif self.expected_features is not None:
                feature_list = self.expected_features
            else:
                logger.error("No feature list provided and no saved features available")
                raise ValueError("Feature list is required")
        
        # Create all possible features
        dataset = self.prepare_features(df)
        
        # For scaling, we need exact feature set expected by scaler
        if self.scale_features and self.scaler is not None and self.expected_features is not None:
            # Create a new dataframe with all expected features
            temp_data = pd.DataFrame(index=dataset.index[-1:])
            
            # Add each expected feature, using 0 as default for missing ones
            for feature in self.expected_features:
                if feature in dataset.columns:
                    temp_data[feature] = dataset[feature].iloc[-1:]
                else:
                    logger.warning(f"Feature '{feature}' missing, using 0")
                    temp_data[feature] = 0
            
            # Now scale with exactly the features expected by scaler
            try:
                scaled_data = self.scaler.transform(temp_data)
                temp_data = pd.DataFrame(
                    scaled_data, 
                    columns=self.expected_features,
                    index=temp_data.index
                )
                logger.info("Successfully scaled prediction data")
            except Exception as e:
                logger.error(f"Scaling error: {str(e)}")
                # Fall back to unscaled data
                
            # If we're doing feature selection, filter to just the selected features
            if self.feature_selection and self.selected_features is not None:
                # Keep only the selected features
                available_selected = [f for f in self.selected_features if f in temp_data.columns]
                temp_data = temp_data[available_selected]
                    
            return temp_data
        
        # If we're not using the scaler, just extract the requested features
        latest_data = pd.DataFrame(index=dataset.index[-1:])
        
        # Add each feature, using 0 as default for missing ones
        for feature in feature_list:
            if feature in dataset.columns:
                latest_data[feature] = dataset[feature].iloc[-1:]
            else:
                logger.warning(f"Feature '{feature}' missing, using 0")
                latest_data[feature] = 0
        
        return latest_data
    
    def get_feature_importance(self, model, feature_list: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Create a mapping of feature names to their importance from a model.
        
        Parameters:
            model: Trained model with feature_importances_ attribute
            feature_list: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        # Use selected features if available and feature_list not provided
        if feature_list is None and self.selected_features is not None:
            feature_list = self.selected_features
        
        if feature_list is None:
            logger.warning("No feature list provided")
            return {}
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Check if lengths match
        if len(importances) != len(feature_list):
            logger.warning(f"Feature list length ({len(feature_list)}) doesn't match importances length ({len(importances)})")
            return {}
        
        # Map feature names to importances
        importance_dict = dict(zip(feature_list, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_feature_descriptions(self, feature_list: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get descriptions for features.
        
        Parameters:
            feature_list: List of features to describe (all if None)
            
        Returns:
            Dictionary mapping feature names to descriptions
        """
        if feature_list is None:
            return self.feature_descriptions
        
        # Filter for only requested features
        return {k: v for k, v in self.feature_descriptions.items() if k in feature_list}
    
    def save_processor_state(self, filename="data_processor_state.pkl"):
        """Save the processor state for consistent prediction later."""
        import pickle
        
        state = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'training_features': self.training_features,
            'expected_features': self.expected_features,
            'scale_features': self.scale_features,
            'scaler_type': self.scaler_type,
            'feature_selection': self.feature_selection
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved processor state to {filename}")
    
    def load_processor_state(self, filename="data_processor_state.pkl"):
        """Load the processor state for consistent prediction."""
        import pickle
        import os
        
        if not os.path.exists(filename):
            logger.error(f"Processor state file {filename} not found")
            return False
            
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
        self.scaler = state['scaler']
        self.feature_selector = state['feature_selector']
        self.selected_features = state['selected_features']
        self.training_features = state['training_features'] 
        self.expected_features = state['expected_features']
        self.scale_features = state['scale_features']
        self.scaler_type = state['scaler_type']
        self.feature_selection = state['feature_selection']
        
        logger.info(f"Loaded processor state from {filename}")
        return True

# Test the data processor
if __name__ == "__main__":
    import pandas as pd
    
    # Load data from CSV (assuming it's already saved)
    try:
        print("Loading data...")
        df = pd.read_csv("xauusd_M5.csv")
        
        print("Initializing data processor...")
        processor = DataProcessor(
            scale_features=True,
            feature_selection=True,
            n_features=30,
            handle_missing='fill_median',
            smooth_outliers=True
        )
        
        print("Preparing ML data...")
        X, y, feature_list = processor.prepare_ml_data(df)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Number of features: {len(feature_list)}")
        print("\nTop 10 features:")
        for feature in feature_list[:10]:
            desc = processor.feature_descriptions.get(feature, "No description available")
            print(f"- {feature}: {desc}")
        
        print("\nPreparing latest data for prediction...")
        latest = processor.prepare_latest_for_prediction(df, feature_list)
        print(f"Latest data shape: {latest.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()