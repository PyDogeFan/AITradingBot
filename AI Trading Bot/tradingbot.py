import ccxt
import pandas as pd
import numpy as np
from itertools import product

def fetch_ohlcv(exchange_id, symbol, timeframe='1d', since=None, limit=100):
    exchange = getattr(ccxt, exchange_id)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    return ohlcv

def create_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def compute_indicators(df, short_window, long_window, signal_window, bb_window=20, stoch_window=14):
    # Moving Averages
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # RSI
    df['rsi'] = compute_rsi(df['close'])
    
    # MACD
    df['macd'] = df['close'].ewm(span=short_window, adjust=False).mean() - df['close'].ewm(span=long_window, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=bb_window).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=bb_window).std()
    
    # Stochastic Oscillator
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(window=stoch_window).min()) / 
                           (df['high'].rolling(window=stoch_window).max() - df['low'].rolling(window=stoch_window).min()))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df):
    df['signal'] = 0
    df['signal'][(df['short_mavg'] > df['long_mavg']) & (df['rsi'] < 30) & (df['close'] < df['bb_lower']) & (df['stoch_k'] < 20)] = 1
    df['signal'][(df['short_mavg'] < df['long_mavg']) & (df['rsi'] > 70) & (df['close'] > df['bb_upper']) & (df['stoch_k'] > 80)] = -1
    df['signal'][(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) > df['macd_signal'].shift(1))] = -1  # Sell signal
    df['signal'][(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) < df['macd_signal'].shift(1))] = 1   # Buy signal
    df['positions'] = df['signal'].diff()
    return df

def backtest_strategy(df, initial_balance=1157.94, stop_loss=0.05):
    balance = initial_balance
    position = 0
    buy_price = 0
    trailing_stop = 0

    for i in range(1, len(df)):
        if df['positions'].iloc[i] == 1 and balance > 0:
            # Buy
            buy_price = df['close'].iloc[i]
            position = balance / buy_price
            balance = 0
            trailing_stop = buy_price * (1 - stop_loss)
        elif df['positions'].iloc[i] == -1 and position > 0:
            # Sell
            balance = position * df['close'].iloc[i]
            position = 0
        elif position > 0:
            if df['close'].iloc[i] < trailing_stop:
                # Trailing Stop-Loss
                balance = position * df['close'].iloc[i]
                position = 0
            else:
                trailing_stop = max(trailing_stop, df['close'].iloc[i] * (1 - stop_loss))

    # Final balance calculation
    if position > 0:
        balance = position * df['close'].iloc[-1]
    return balance

def grid_search(df, short_window_range, long_window_range, signal_window_range):
    best_balance = 0
    best_params = (0, 0, 0)
    for short_window, long_window, signal_window in product(short_window_range, long_window_range, signal_window_range):
        df_indicators = compute_indicators(df.copy(), short_window, long_window, signal_window)
        df_signals = generate_signals(df_indicators)
        final_balance = backtest_strategy(df_signals)
        if final_balance > best_balance:
            best_balance = final_balance
            best_params = (short_window, long_window, signal_window)
    return best_params, best_balance

# Fetch historical data
exchange_id = 'binance'
symbol = 'BTC/USDT'
ohlcv = fetch_ohlcv(exchange_id, symbol, '1d')

# Create DataFrame
df = create_dataframe(ohlcv)

# Define parameter ranges
short_window_range = range(10, 50, 5)
long_window_range = range(50, 200, 10)
signal_window_range = range(5, 20, 2)

# Perform grid search
best_params, best_balance = grid_search(df, short_window_range, long_window_range, signal_window_range)
print(f"Best Parameters: Short Window: {best_params[0]}, Long Window: {best_params[1]}, Signal Window: {best_params[2]}")
print(f"Best Balance after Grid Search: {best_balance:.2f} USD")
