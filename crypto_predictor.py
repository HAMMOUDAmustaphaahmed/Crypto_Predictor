import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# Binance API endpoints
BINANCE_API_BASE = "https://api.binance.com/api/v3"

def get_binance_usdt_pairs():
    """Get all available USDT trading pairs from Binance"""
    try:
        response = requests.get(f"{BINANCE_API_BASE}/exchangeInfo")
        data = response.json()
        
        # Filter for USDT pairs that are currently trading
        usdt_pairs = [
            symbol["symbol"] 
            for symbol in data["symbols"] 
            if symbol["quoteAsset"] == "USDT" and symbol["status"] == "TRADING"
        ]
        
        # Format them as BASE/USDT for display
        formatted_pairs = [f"{pair[:-4]}/USDT" for pair in usdt_pairs]
        
        return formatted_pairs
    except Exception as e:
        st.error(f"Error fetching pairs from Binance: {e}")
        # Fallback to common pairs if API fails
        return [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", 
            "DOT/USDT", "DOGE/USDT", "LINK/USDT", "MATIC/USDT", "LTC/USDT"
        ]

def calculate_fibonacci_levels(df, window=126):
    """
    Calculate Fibonacci retracement levels based on recent price action
    
    Parameters:
    df (DataFrame): Price data
    window (int): Lookback period for finding high and low points
    
    Returns:
    DataFrame with Fibonacci retracement levels
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate Fibonacci levels using rolling windows
    for i in range(window, len(df)):
        # Get the window
        window_data = df['close'].iloc[i-window:i]
        
        # Find high and low in the window
        high = window_data.max()
        low = window_data.min()
        
        # Calculate Fibonacci retracement levels
        diff = high - low
        
        # Key Fibonacci levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
        result_df.loc[df.index[i], 'fib_23.6'] = high - 0.236 * diff
        result_df.loc[df.index[i], 'fib_38.2'] = high - 0.382 * diff
        result_df.loc[df.index[i], 'fib_50.0'] = high - 0.5 * diff
        result_df.loc[df.index[i], 'fib_61.8'] = high - 0.618 * diff
        result_df.loc[df.index[i], 'fib_78.6'] = high - 0.786 * diff
    
    # Calculate distance to nearest Fibonacci level
    fib_cols = ['fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8', 'fib_78.6']
    
    # For each row after the window, find the distance to the nearest Fibonacci level
    for i in range(window, len(result_df)):
        price = result_df.loc[result_df.index[i], 'close']
        
        # Get Fibonacci levels for this row
        fib_levels = result_df.loc[result_df.index[i], fib_cols].values
        
        # Calculate distances
        distances = np.abs(fib_levels - price)
        
        # Nearest level (index of minimum distance)
        nearest_idx = np.argmin(distances)
        
        # Normalized distance to nearest level (as percentage of price)
        min_distance = distances[nearest_idx]
        norm_distance = (min_distance / price) * 100
        
        # Store the nearest level and distance
        result_df.loc[result_df.index[i], 'nearest_fib'] = fib_cols[nearest_idx]
        result_df.loc[result_df.index[i], 'fib_distance'] = norm_distance
    
    return result_df

def get_binance_historical_data(symbol, interval='1d', lookback_days=1095):
    """
    Get historical OHLCV data from Binance
    
    Parameters:
    symbol (str): Trading pair in Binance format (e.g., 'BTCUSDT')
    interval (str): Kline interval (default: '1d')
    lookback_days (int): Number of days to look back (default: 1095 days = ~3 years)
    
    Returns:
    pandas.DataFrame: OHLCV data with technical indicators
    """
    try:
        # Calculate start time (current time - lookback_days)
        start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        # Binance API has a limit of 1000 candles per request
        # For 3 years of daily data, we need to make multiple requests
        all_klines = []
        current_start = start_time
        
        # Loop until we get all the data
        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'limit': 1000
            }
            
            response = requests.get(f"{BINANCE_API_BASE}/klines", params=params)
            klines = response.json()
            
            if not klines or len(klines) == 0:
                break
                
            all_klines.extend(klines)
            
            # If we got less than 1000 candles, we're done
            if len(klines) < 1000:
                break
                
            # Update start time for the next request
            current_start = klines[-1][0] + 1
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Convert string values to floats
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Set index to open_time
        df.set_index('open_time', inplace=True)
        
        # Calculate technical indicators
        # ----------------------- ORIGINAL INDICATORS ----------------------- #
        # Simple Moving Average (20-day)
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # Relative Strength Index (14-day)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Manually compute ADI (Accumulation/Distribution Index)
        df['ADI'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['ADI_EMA'] = df['ADI'].ewm(span=14).mean()
        
        # Volatility (simple standard deviation of returns)
        df['Volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # ----------------------- NEW INDICATORS ----------------------- #
        # 1. Exponential Moving Averages (60-day and 225-day)
        df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()
        df['EMA_225'] = df['close'].ewm(span=225, adjust=False).mean()
        
        # 2. MACD (Moving Average Convergence Divergence)
        # MACD Line = 12-period EMA - 26-period EMA
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_line'] = df['EMA_12'] - df['EMA_26']
        
        # Signal Line = 9-period EMA of MACD Line
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        
        # MACD Histogram = MACD Line - Signal Line
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
        
        # 3. Stochastic Oscillator
        # Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        period = 14
        df['Stoch_K'] = 100 * ((df['close'] - df['low'].rolling(period).min()) / 
                              (df['high'].rolling(period).max() - df['low'].rolling(period).min()))
        
        # %D = 3-day SMA of %K
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # 4. Fibonacci Retracement Levels (requires more complex calculation)
        df = calculate_fibonacci_levels(df)
        
        # Select relevant columns for the model
        model_cols = [
            'close', 'ADI_EMA', 'RSI', 'SMA_20', 'Volatility',
            'EMA_60', 'EMA_225', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'Stoch_K', 'Stoch_D', 'fib_distance'
        ]
        
        df_filtered = df[model_cols].copy()
        df_filtered.dropna(inplace=True)
        
        return df_filtered, df
        
    except Exception as e:
        st.error(f"Error fetching historical data from Binance: {e}")
        return None, None

def create_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def adjust_lookback_for_interval(interval):
    """Adjust the lookback period based on the selected interval
    
    Parameters:
    interval (str): Time interval for Binance klines
    
    Returns:
    int: Appropriate lookback period in days
    """
    # Map intervals to appropriate lookback periods
    interval_lookback = {
        '15m': 21,         # 15 minutes - 3 weeks
        '30m': 30,         # 30 minutes - 1 month
        '1h': 60,          # 1 hour - 2 months
        '2h': 90,          # 2 hours - 3 months
        '4h': 180,         # 4 hours - 6 months
        '1d': 1095,        # 1 day - 3 years
        '1w': 1825,        # 1 week - 5 years
        '1M': 3650         # 1 month - 10 years
    }
    
    return interval_lookback.get(interval, 1095)  # Default to 1095 days (3 years)

# NEW FUNCTION: Find the biggest green candle and check COR strategy
def find_biggest_green_candle_strategy(data, num_candles_to_check):
    """
    Find the biggest green candle and check if subsequent candles follow the strategy
    
    Parameters:
    data (DataFrame): OHLCV data
    num_candles_to_check (int): Number of recent candles to check
    
    Returns:
    tuple: (success, cor_info, current_price)
    """
    if data is None or len(data) < num_candles_to_check + 2:  # Need at least num_candles_to_check + 2 candles
        return False, None, None
    
    # Get the last num_candles_to_check candles, excluding the 2 most recent
    recent_data = data.iloc[-(num_candles_to_check+2):-2].copy()
    
    # Calculate green candle size (close - open for green candles)
    recent_data['green_size'] = np.where(
        recent_data['close'] > recent_data['open'],
        recent_data['close'] - recent_data['open'],
        0
    )
    
    # Find the biggest green candle
    biggest_green_idx = recent_data['green_size'].idxmax()
    
    # If there's no green candle, return False
    if recent_data.loc[biggest_green_idx, 'green_size'] == 0:
        return False, None, None
    
    cor_open = recent_data.loc[biggest_green_idx, 'open']
    cor_close = recent_data.loc[biggest_green_idx, 'close']
    
    # Calculate the average (moyenne)
    moyenne = (cor_open + cor_close) / 2
    
    # Get candles after the COR
    post_cor_data = data.loc[biggest_green_idx:].iloc[1:]
    
    # Check if all post-COR candles have high and low between COR close and moyenne
    upper_bound = max(cor_close, moyenne)
    lower_bound = min(cor_close, moyenne)
    
    condition_met = True
    for idx, row in post_cor_data.iterrows():
        if row['high'] > upper_bound or row['low'] < lower_bound:
            condition_met = False
            break
    
    # Get current price (close of the latest candle)
    current_price = data['close'].iloc[-1]
    
    cor_info = {
        'date': biggest_green_idx,
        'open': cor_open,
        'close': cor_close,
        'moyenne': moyenne,
    }
    
    return condition_met, cor_info, current_price

# NEW FUNCTION: Scan all USDT pairs for the custom strategy
def scan_usdt_pairs_for_strategy(timeframe, num_candles):
    """
    Scan all USDT pairs for the custom strategy
    
    Parameters:
    timeframe (str): Kline interval for Binance
    num_candles (int): Number of candles to check
    
    Returns:
    list: Matching pairs with their COR information
    """
    # Get all USDT pairs
    pairs = get_binance_usdt_pairs()
    if not pairs:
        return []
    
    # Adjust lookback period
    lookback_days = adjust_lookback_for_interval(timeframe)
    
    # Initialize results list
    matching_pairs = []
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Iterate through pairs
    for i, pair in enumerate(pairs):
        # Update status and progress
        status_text.text(f"Scanning {pair} ({i+1}/{len(pairs)})")
        progress_bar.progress((i+1) / len(pairs))
        
        # Convert to Binance format
        binance_symbol = pair.replace("/", "")
        
        try:
            # Get historical data
            _, data = get_binance_historical_data(
                binance_symbol, 
                interval=timeframe, 
                lookback_days=lookback_days
            )
            
            # Check if the pair matches our strategy
            success, cor_info, current_price = find_biggest_green_candle_strategy(
                data, 
                num_candles
            )
            
            if success:
                matching_pairs.append({
                    'pair': pair,
                    'cor_info': cor_info,
                    'current_price': current_price
                })
        except Exception as e:
            # Continue if there's an error with this pair
            continue
    
    # Clear progress display
    status_text.empty()
    progress_bar.empty()
    
    return matching_pairs

def main():
    st.title("🚀 Enhanced Crypto Predictor")
    
    # Create a sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["Price Predictor", "Custom Strategy Scanner"]
    )
    
    if page == "Price Predictor":
        run_price_predictor()
    else:
        run_custom_strategy_scanner()

def run_price_predictor():
    st.header("Price Predictor")
    
    # Get USDT pairs from Binance
    with st.spinner("Fetching available trading pairs..."):
        crypto_pairs = get_binance_usdt_pairs()
    
    st.write(f"Found {len(crypto_pairs)} USDT trading pairs on Binance")
    
    # Create two columns for side-by-side selectors
    col1, col2 = st.columns(2)
    
    # Create dropdown for crypto selection in the first column
    with col1:
        crypto = st.selectbox("Select Crypto Pair", crypto_pairs)
    
    # Create dropdown for timeframe selection in the second column
    with col2:
        # Available timeframes on Binance
        timeframes = {
            '15 Minutes': '15m',
            '30 Minutes': '30m',
            '1 Hour': '1h',
            '2 Hours': '2h',
            '4 Hours': '4h',
            '1 Day': '1d',
            '1 Week': '1w',
            '1 Month': '1M'
        }
        timeframe_display = list(timeframes.keys())
        timeframe_selected = st.selectbox("Select Timeframe", timeframe_display, index=5)  # Default to 1 Day
        interval = timeframes[timeframe_selected]
    
    # Adjust forecast days based on timeframe
    forecast_days_default = 14
    forecast_days_max = 30
    
    # For smaller timeframes, increase the max forecast points
    if interval in ['15m', '30m', '1h']:
        forecast_days_label = "Forecast Points"
        forecast_days_max = 48
        forecast_days_default = 24
    else:
        forecast_days_label = "Forecast Days"
    
    # Modified slider for days/points
    days = st.slider(forecast_days_label, 7, forecast_days_max, forecast_days_default)
    
    # Convert trading pair format to Binance format
    binance_symbol = crypto.replace("/", "")
    
    # Adjust lookback period based on selected interval
    lookback_days = adjust_lookback_for_interval(interval)
    
    # Load data with a spinner
    with st.spinner(f"Loading historical {timeframe_selected} data for {crypto}..."):
        data, full_data = get_binance_historical_data(binance_symbol, interval=interval, lookback_days=lookback_days)
        
    if data is None or len(data) < 100:
        st.error(f"Could not retrieve sufficient historical data for {crypto} with {timeframe_selected} timeframe. Please select another pair or timeframe.")
        return
    
    st.success(f"Successfully loaded {len(data)} {timeframe_selected} candles of historical data with {len(data.columns)} technical indicators")
    
    # Add a progress indicator for model training
    progress_text = "Training LSTM model. This may take a moment..."
    progress_bar = st.progress(0)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Prepare sequences
    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Include all features
    
    # Train model with progress updates
    model = create_model((X.shape[1], X.shape[2]))
    
    # Set number of epochs for model training
    epochs = 50
    for i in range(epochs):
        model.fit(X, y, epochs=1, batch_size=64, verbose=0)
        progress_bar.progress((i + 1) / epochs)
    
    st.success("Model training completed!")
    
    # Generate forecast
    last_seq = scaled_data[-seq_length:]
    forecast = []
    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, seq_length, scaled_data.shape[1]), verbose=0)
        forecast.append(pred[0, 0])
        # Update last_seq with the prediction and previous features
        new_row = np.zeros(scaled_data.shape[1])
        new_row[0] = pred[0, 0]
        new_row[1:] = last_seq[-1, 1:]  # Keep other features the same
        last_seq = np.append(last_seq[1:], new_row.reshape(1, -1), axis=0)
    
    # Inverse transform to get original prices
    forecast_scaled = np.zeros((len(forecast), scaled_data.shape[1]))
    forecast_scaled[:, 0] = forecast
    forecast_prices = scaler.inverse_transform(forecast_scaled)[:, 0]
    
    # Get the current price for reference
    current_price = data['close'].iloc[-1]
    
    # Display predictions
    # Adjust subheader based on timeframe
    if interval in ['15m', '30m', '1h']:
        forecast_period = f"Next {days} {timeframe_selected} Periods"
    else:
        forecast_period = f"Next {days} {timeframe_selected}s"
    
    st.subheader(forecast_period)
    
    # Create a dataframe for forecast display
    forecast_df = pd.DataFrame({
        f"{timeframe_selected.rstrip('s')}": range(1, days + 1),
        'Price': [f"${price:.2f}" for price in forecast_prices],
        'Change': ['--'] + [f"{(forecast_prices[i] - forecast_prices[i-1]) / forecast_prices[i-1] * 100:.2f}%" 
                           for i in range(1, len(forecast_prices))],
        'Change from Current': [f"{(price - current_price) / current_price * 100:.2f}%" for price in forecast_prices]
    })
    
    st.table(forecast_df)
    
    # Plot price chart
    fig = go.Figure()
    
    # Add historical data (last 90 data points for the selected interval)
    display_points = min(90, len(data))
    
    fig.add_trace(go.Scatter(
        x=data.index[-display_points:],
        y=data['close'][-display_points:],
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add EMA lines to the main chart
    fig.add_trace(go.Scatter(
        x=data.index[-display_points:],
        y=data['EMA_60'][-display_points:],
        name='EMA 60',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index[-display_points:],
        y=data['EMA_225'][-display_points:],
        name='EMA 225',
        line=dict(color='purple', width=1)
    ))
    
    # Add forecast data
    # Calculate appropriate time delta based on interval
    interval_deltas = {
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
        '1M': timedelta(days=30)  # Approximate
    }
    
    time_delta = interval_deltas.get(interval, timedelta(days=1))
    forecast_dates = [data.index[-1] + (i+1) * time_delta for i in range(days)]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        name='Forecast',
        line=dict(color='red', dash='dot')
    ))
    
    # Improve layout
    fig.update_layout(
        title=f"{crypto} ({timeframe_selected}) Price Prediction with EMAs",
        xaxis_title="Date/Time",
        yaxis_title="Price (USDT)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add technical indicators charts
    st.subheader("Technical Indicators")
    
    # Create tabs for different indicators
    tab1, tab2, tab3 = st.tabs(["MACD & Stochastic", "RSI & Volatility", "Fibonacci Levels"])
    
    with tab1:
        # MACD and Stochastic chart
        fig_macd = go.Figure()
        
        # Add MACD line, Signal line and Histogram
        fig_macd.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['MACD_line'][-display_points:], 
            name='MACD Line',
            line=dict(color='blue')
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['MACD_signal'][-display_points:], 
            name='Signal Line',
            line=dict(color='red')
        ))
        
        # Add histogram as a bar chart
        fig_macd.add_trace(go.Bar(
            x=data.index[-display_points:], 
            y=data['MACD_hist'][-display_points:], 
            name='MACD Histogram',
            marker_color=['green' if x > 0 else 'red' for x in data['MACD_hist'][-display_points:]]
        ))
        
        # Add Stochastic oscillator on secondary y-axis
        fig_macd.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['Stoch_K'][-display_points:], 
            name='Stochastic %K',
            line=dict(color='purple', width=1),
            yaxis="y2"
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['Stoch_D'][-display_points:], 
            name='Stochastic %D',
            line=dict(color='orange', width=1),
            yaxis="y2"
        ))
        
        # Add horizontal lines for overbought/oversold on Stochastic
        fig_macd.add_shape(
            type="line", line=dict(dash='dash', color="rgba(128, 128, 128, 0.5)"),
            y0=80, y1=80, x0=data.index[-display_points], x1=data.index[-1], yref="y2"
        )
        
        fig_macd.add_shape(
            type="line", line=dict(dash='dash', color="rgba(128, 128, 128, 0.5)"),
            y0=20, y1=20, x0=data.index[-display_points], x1=data.index[-1], yref="y2"
        )
        
        # Configure dual y-axes
        fig_macd.update_layout(
            title=f"MACD and Stochastic Oscillator (Last {display_points} {timeframe_selected} Periods)",
            xaxis_title="Date/Time",
            yaxis_title="MACD",
            yaxis2=dict(
                title="Stochastic",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tab2:
        # RSI and Volatility chart
        fig_rsi = go.Figure()
        
        # Add RSI
        fig_rsi.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['RSI'][-display_points:], 
            name='RSI',
            line=dict(color='blue')
        ))
        
        # Add horizontal lines for overbought/oversold
        fig_rsi.add_shape(
            type="line", line=dict(dash='dash', color="rgba(255, 0, 0, 0.5)"),
            y0=70, y1=70, x0=data.index[-display_points], x1=data.index[-1]
        )
        
        fig_rsi.add_shape(
            type="line", line=dict(dash='dash', color="rgba(0, 255, 0, 0.5)"),
            y0=30, y1=30, x0=data.index[-display_points], x1=data.index[-1]
        )
        
        # Add Volatility on secondary y-axis
        fig_rsi.add_trace(go.Scatter(
            x=data.index[-display_points:], 
            y=data['Volatility'][-display_points:], 
            name='Volatility',
            line=dict(color='orange'),
            yaxis="y2"
        ))
        
        # Configure dual y-axes
        fig_rsi.update_layout(
                        title=f"RSI and Volatility (Last {display_points} {timeframe_selected} Periods)",
            xaxis_title="Date/Time",
            yaxis_title="RSI",
            yaxis2=dict(
                title="Volatility",
                overlaying="y",
                side="right"
            ),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    

    
    # Add model info section
    with st.expander("Model Information"):
        st.write("""
        This prediction uses an LSTM neural network with the following features:
        - Price data
        - Accumulation/Distribution Index (ADI)
        - Relative Strength Index (RSI)
        - 20-day Simple Moving Average (SMA)
        - 20-day Volatility (standard deviation of returns)
        - 60-day and 225-day Exponential Moving Averages (EMA)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        - Fibonacci Retracement Levels
        
        The model is trained on historical data with a sequence length of 60 periods.
        """)
    
    # Disclaimer
    st.caption("""
    **Disclaimer**: This prediction is for educational purposes only. Cryptocurrency markets are highly volatile 
    and unpredictable. Do not make investment decisions based solely on this forecast.
    """)

# Constantes pour les timeframes
TIMEFRAMES = {
    '15 Minutes': '15m',
    '30 Minutes': '30m',
    '1 Heure': '1h',
    '2 Heures': '2h',
    '4 Heures': '4h',
    '1 Jour': '1d'
}

# Configuration de la mise en page
STYLE = {
    'success': 'background-color: #1c4b27',
    'danger': 'background-color: #4b1c1c',
    'warning': 'background-color: #4b451c'
}

def run_custom_strategy_scanner():
    st.header("Scanner de Stratégie Personnalisé")
    
    with st.expander("📈 Paramètres du Scanner", expanded=True):
        num_candles = st.number_input(
            "Nombre de Bougies à Analyser", 
            min_value=10, 
            max_value=200, 
            value=50,
            help="Nombre de bougies récentes à analyser pour la stratégie COR"
        )
        
        if st.button("🔍 Lancer l'Analyse Complète"):
            # Créer un conteneur pour chaque timeframe
            timeframe_containers = {}
            for timeframe_name in TIMEFRAMES.keys():
                timeframe_containers[timeframe_name] = st.empty()
            
            # Créer un conteneur pour les résultats en direct
            live_results = st.container()
            with live_results:
                st.subheader("🔄 Résultats en Direct")
                results_table = st.empty()
                
                # Initialiser le DataFrame des résultats
                all_results_df = pd.DataFrame(
                    columns=["Timeframe", "Paire USDT", "COR Ouverture/Fermeture", 
                            "Prix Actuel", "Range %", "Zone de Trading"]
                )
                
                # Pour chaque timeframe
                for timeframe_name, timeframe_code in TIMEFRAMES.items():
                    container = timeframe_containers[timeframe_name]
                    with container:
                        st.markdown(f"**Analyse du {timeframe_name}**")
                        
                        matching_pairs = scan_usdt_pairs_for_strategy(timeframe_code, num_candles)
                        
                        if matching_pairs:
                            # Ajouter chaque nouvelle paire aux résultats
                            for pair in matching_pairs:
                                new_row = pd.DataFrame([{
                                    "Timeframe": timeframe_name,
                                    "Paire USDT": pair['pair'],
                                    "COR Ouverture/Fermeture": f"{pair['cor_info']['open']:.2f} / {pair['cor_info']['close']:.2f}",
                                    "Prix Actuel": f"{pair['current_price']:.2f}",
                                    "Range %": f"{pair['cor_info'].get('range_percentage', 0):.2f}%",
                                    "Zone de Trading": "✅" if pair['cor_info'].get('in_trading_zone', False) else "❌"
                                }])
                                
                                all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)
                                
                                # Mettre à jour le tableau des résultats
                                results_table.dataframe(
                                    all_results_df.style.apply(lambda x: ['background-color: #1c4b27' 
                                                                        if x['Zone de Trading'] == "✅" 
                                                                        else 'background-color: #4b1c1c' 
                                                                        for i in x], axis=1)
                                )
                        
                        # Afficher un message si aucune correspondance
                        else:
                            st.info(f"Aucune correspondance trouvée pour {timeframe_name}")

def display_all_results(all_results):
    """
    Affiche les résultats pour tous les timeframes
    """
    if not all_results:
        st.warning("Aucune correspondance trouvée sur aucun timeframe.")
        return
    
    # Créer des onglets pour chaque timeframe
    tabs = st.tabs(list(all_results.keys()))
    
    # Pour chaque timeframe
    for tab, (timeframe, pairs) in zip(tabs, all_results.items()):
        with tab:
            st.subheader(f"Résultats pour {timeframe}")
            
            # Créer le DataFrame des résultats
            results = []
            for pair in pairs:
                results.append([
                    pair['pair'],
                    f"{pair['cor_info']['open']:.2f} / {pair['cor_info']['close']:.2f}",
                    f"{pair['current_price']:.2f}",
                    f"{pair['cor_info'].get('range_percentage', 0):.2f}%",
                    "✅" if pair['cor_info'].get('in_trading_zone', False) else "❌"
                ])
            
            if results:
                results_df = pd.DataFrame(
                    results,
                    columns=["Paire USDT", "COR Ouverture/Fermeture", "Prix Actuel", "Range %", "Zone de Trading"]
                )
                
                # Afficher le tableau avec mise en forme conditionnelle
                st.dataframe(
                    results_df.style.apply(lambda x: ['background-color: #1c4b27' 
                                                    if x['Zone de Trading'] == "✅" 
                                                    else 'background-color: #4b1c1c' 
                                                    for i in x], axis=1)
                )
                
                # Ajouter des statistiques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de Paires", len(results))
                with col2:
                    in_zone = sum(1 for r in results if r[4] == "✅")
                    st.metric("En Zone de Trading", f"{in_zone} ({in_zone/len(results)*100:.1f}%)")
                with col3:
                    avg_range = sum(float(r[3].replace('%', '')) for r in results) / len(results)
                    st.metric("Range Moyen", f"{avg_range:.1f}%")
            else:
                st.info(f"Aucune correspondance trouvée pour {timeframe}")

def find_biggest_green_candle_strategy(data, num_candles_to_check):
    """
    Find the biggest green candle and check if subsequent candles follow the strategy
    
    Parameters:
    data (DataFrame): OHLCV data
    num_candles_to_check (int): Number of recent candles to check
    
    Returns:
    tuple: (success, cor_info, current_price)
    """
    if data is None or len(data) < num_candles_to_check + 2:  # Need at least num_candles_to_check + 2 candles
        return False, None, None
    
    # Get the last num_candles_to_check candles, excluding the 2 most recent
    recent_data = data.iloc[-(num_candles_to_check+2):-2].copy()
    
    # Calculate green candle size (close - open for green candles)
    recent_data['green_size'] = np.where(
        recent_data['close'] > recent_data['open'],
        recent_data['close'] - recent_data['open'],
        0
    )
    
    # Find the biggest green candle
    biggest_green_idx = recent_data['green_size'].idxmax()
    
    # If there's no green candle, return False
    if recent_data.loc[biggest_green_idx, 'green_size'] == 0:
        return False, None, None
    
    cor_open = recent_data.loc[biggest_green_idx, 'open']
    cor_close = recent_data.loc[biggest_green_idx, 'close']
    
    # Calculate the average (moyenne)
    moyenne = (cor_open + cor_close) / 2
    
    # Get candles after the COR
    post_cor_data = data.loc[biggest_green_idx:].iloc[1:]
    
    # Check if all post-COR candles have high and low between COR close and moyenne
    upper_bound = max(cor_close, moyenne)
    lower_bound = min(cor_close, moyenne)
    
    condition_met = True
    for idx, row in post_cor_data.iterrows():
        if row['high'] > upper_bound or row['low'] < lower_bound:
            condition_met = False
            break
    
    # Get current price (close of the latest candle)
    current_price = data['close'].iloc[-1]
    
    cor_info = {
        'date': biggest_green_idx,
        'open': cor_open,
        'close': cor_close,
        'moyenne': moyenne,
    }
    
    return condition_met, cor_info, current_price

def scan_usdt_pairs_for_strategy(timeframe, num_candles):
    """
    Analyse tous les pairs USDT pour un timeframe donné
    """
    # Récupérer les paires USDT
    pairs = get_binance_usdt_pairs()
    if not pairs:
        return []
    
    # Ajuster la période d'historique
    lookback_days = adjust_lookback_for_interval(timeframe)
    
    # Initialiser la liste des résultats
    matching_pairs = []
    
    # Configurer la barre de progression pour ce timeframe
    col1, col2 = st.columns([3, 1])
    with col1:
        progress_bar = st.progress(0)
    with col2:
        pairs_counter = st.empty()
    
    # Parcourir les paires
    for i, pair in enumerate(pairs):
        # Mettre à jour le compteur
        pairs_counter.text(f"Paire {i+1}/{len(pairs)}")
        progress_bar.progress((i + 1) / len(pairs))
        
        try:
            # Obtenir les données historiques
            _, data = get_binance_historical_data(
                pair.replace("/", ""),
                interval=timeframe,
                lookback_days=lookback_days
            )
            
            # Vérifier la stratégie
            success, cor_info, current_price = find_biggest_green_candle_strategy(
                data,
                num_candles
            )
            
            if success:
                matching_pairs.append({
                    'pair': pair,
                    'cor_info': cor_info,
                    'current_price': current_price
                })
        except Exception as e:
            continue
    
    # Nettoyer l'affichage
    progress_bar.empty()
    pairs_counter.empty()
    
    return matching_pairs

def main():
    st.set_page_config(
        page_title="🚀 Crypto Predictor Pro",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 Crypto Predictor Pro")
    
    # Créer une navigation dans la barre latérale
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choisir une section:",
        ["Prédicteur de Prix", "Scanner de Stratégie"]
    )
    
    if page == "Prédicteur de Prix":
        run_price_predictor()
    else:
        run_custom_strategy_scanner()

if __name__ == "__main__":
    main()
