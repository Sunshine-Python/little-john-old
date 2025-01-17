import streamlit as st
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from backtesting.lib import crossover
from datetime import datetime, timedelta


def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance at 5-minute intervals.
    """
    # Convert start_date and end_date to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Calculate the date 60 days ago
    sixty_days_ago = datetime.now() - timedelta(days=60)

    # If start_date is more than 60 days ago, adjust it
    if start_date < sixty_days_ago:
        print(f"Warning: Adjusted start date to {sixty_days_ago.strftime('%Y-%m-%d')} due to YFinance limitations.")
        start_date = sixty_days_ago

    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date, interval='2m')
    
    # If data is empty, try fetching daily data instead
    if data.empty:
        print("No 15-minute interval data available. Fetching daily data instead.")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Drop 'Adj Close' column if it exists
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])

    # Rename columns
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    return data


# Buy and Hold Strategy
class BuyAndHoldStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()

def buy_and_hold_viz():
    st.write("The Buy and Hold strategy simply buys the stock at the beginning of the period and holds it until the end. There are no parameters to adjust.")
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', name='Stock Price'))
    fig.update_layout(title='Buy and Hold Visualization', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)

# SMA Strategy
class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True

    def init(self):
        self.n1 = st.session_state.get('sma_n1', self.n1)
        self.n2 = st.session_state.get('sma_n2', self.n2)
        self.stop_loss_pct = st.session_state.get('sma_stop_loss', self.stop_loss_pct)
        self.take_profit_pct = st.session_state.get('sma_take_profit', self.take_profit_pct)
        self.enable_shorting = st.session_state.get('sma_enable_shorting', self.enable_shorting)
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.entry_price = 0
        self.position_type = None  # 'long' or 'short'

    def next(self):
        if self.position:
            # Check for stop loss or take profit
            if self.position_type == 'long':
                if self.data.Close[-1] <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None

        # If no position, check for new entry signals
        if not self.position:
            if crossover(self.sma1, self.sma2):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.sma2, self.sma1) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def sma_cross_viz():
    st.subheader('SMA Cross Parameters')
    n1 = st.slider('Short Window (n1)', key='sma_n1', min_value=5, max_value=50, value=10)
    n2 = st.slider('Long Window (n2)', key='sma_n2', min_value=20, max_value=100, value=20)
    stop_loss = st.slider('Stop Loss (%)', key='sma_stop_loss', min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    take_profit = st.slider('Take Profit (%)', key='sma_take_profit', min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    enable_shorting = st.checkbox('Enable Shorting', key='sma_enable_shorting', value=True)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    short_sma = np.convolve(y, np.ones(n1), mode='valid') / n1
    long_sma = np.convolve(y, np.ones(n2), mode='valid') / n2
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x[n1-1:], y=short_sma, mode='lines', name=f'SMA({n1})'))
    fig.add_trace(go.Scatter(x=x[n2-1:], y=long_sma, mode='lines', name=f'SMA({n2})'))
    fig.update_layout(title='SMA Cross Visualization', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{Short SMA} = \frac{1}{n1} \sum_{i=0}^{n1-1} P_{t-i}')
    st.latex(r'\text{Long SMA} = \frac{1}{n2} \sum_{i=0}^{n2-1} P_{t-i}')
    st.write(f"Stop Loss: {stop_loss}%")
    st.write(f"Take Profit: {take_profit}%")
    st.write(f"Shorting {'Enabled' if enable_shorting else 'Disabled'}")

# RSI Strategy
class RsiStrategy(Strategy):
    length = 14
    overbought = 70
    oversold = 30
    stop_loss_pct = 2.0
    take_profit_pct = 5.0

    def init(self):
        self.length = st.session_state.get('rsi_length', self.length)
        self.overbought = st.session_state.get('rsi_overbought', self.overbought)
        self.oversold = st.session_state.get('rsi_oversold', self.oversold)
        self.stop_loss_pct = st.session_state.get('rsi_stop_loss', self.stop_loss_pct)
        self.take_profit_pct = st.session_state.get('rsi_take_profit', self.take_profit_pct)
        close_prices = pd.Series(self.data.Close)
        if len(close_prices) >= self.length:
            self.rsi = self.I(ta.momentum.RSIIndicator(close=close_prices, window=self.length).rsi)
        self.entry_price = 0
        self.position_type = None  # 'long' or 'short'

    def next(self):
        if self.position:
            # Check for stop loss or take profit
            if self.position_type == 'long':
                if self.data.Close[-1] <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None

        # If no position, check for new entry signals
        if not self.position and hasattr(self, 'rsi'):
            if self.rsi[-1] < self.oversold:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.rsi[-1] > self.overbought:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def rsi_viz():
    st.subheader('RSI Parameters')
    length = st.slider('RSI Length', key='rsi_length', min_value=5, max_value=50, value=14)
    overbought = st.slider('Overbought Level', key='rsi_overbought', min_value=70, max_value=90, value=70)
    oversold = st.slider('Oversold Level', key='rsi_oversold', min_value=10, max_value=30, value=30)
    stop_loss = st.slider('Stop Loss (%)', key='rsi_stop_loss', min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    take_profit = st.slider('Take Profit (%)', key='rsi_take_profit', min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    rsi = ta.momentum.RSIIndicator(pd.Series(y), window=length).rsi()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rsi, mode='lines', name='RSI'))
    fig.add_shape(type="line", x0=0, y0=overbought, x1=100, y1=overbought, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=oversold, x1=100, y1=oversold, line=dict(color="green", width=2, dash="dash"))
    fig.update_layout(title='RSI Visualization', xaxis_title='Time', yaxis_title='RSI')
    st.plotly_chart(fig)
    
    st.latex(r'\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}')
    st.write(f"Stop Loss: {stop_loss}%")
    st.write(f"Take Profit: {take_profit}%")

# MACD Strategy
class MacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True

    def init(self):
        self.fast = st.session_state.get('macd_fast', self.fast)
        self.slow = st.session_state.get('macd_slow', self.slow)
        self.signal = st.session_state.get('macd_signal', self.signal)
        self.stop_loss_pct = st.session_state.get('macd_stop_loss', self.stop_loss_pct)
        self.take_profit_pct = st.session_state.get('macd_take_profit', self.take_profit_pct)
        self.enable_shorting = st.session_state.get('macd_enable_shorting', self.enable_shorting)
        
        close_series = pd.Series(self.data.Close)
        self.macd_indicator = ta.trend.MACD(close_series, window_slow=self.slow, window_fast=self.fast, window_sign=self.signal)
        self.macd_line = self.I(lambda: self.macd_indicator.macd())
        self.signal_line = self.I(lambda: self.macd_indicator.macd_signal())
        self.macd_diff = self.I(lambda: self.macd_indicator.macd_diff())
        
        self.entry_price = 0
        self.position_type = None  # 'long' or 'short'

    def next(self):
        if self.position:
            # Check for stop loss or take profit
            if self.position_type == 'long':
                if self.data.Close[-1] <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None

        # If no position, check for new entry signals
        if not self.position:
            if crossover(self.macd_line, self.signal_line):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.signal_line, self.macd_line) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def macd_viz():
    st.subheader('MACD Parameters')
    fast = st.slider('Fast Length', key='macd_fast', min_value=5, max_value=50, value=12)
    slow = st.slider('Slow Length', key='macd_slow', min_value=20, max_value=100, value=26)
    signal = st.slider('Signal Length', key='macd_signal', min_value=5, max_value=50, value=9)
    stop_loss = st.slider('Stop Loss (%)', key='macd_stop_loss', min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    take_profit = st.slider('Take Profit (%)', key='macd_take_profit', min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    enable_shorting = st.checkbox('Enable Shorting', key='macd_enable_shorting', value=True)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    macd = ta.trend.MACD(pd.Series(y), window_slow=slow, window_fast=fast, window_sign=signal)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=macd.macd(), mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=x, y=macd.macd_signal(), mode='lines', name='Signal'))
    fig.add_bar(x=x, y=macd.macd_diff(), name='Histogram')
    fig.update_layout(title='MACD Visualization', xaxis_title='Time', yaxis_title='Value')
    st.plotly_chart(fig)
    
    st.latex(r'\text{MACD Line} = \text{EMA}(fast) - \text{EMA}(slow)')
    st.latex(r'\text{Signal Line} = \text{EMA}(\text{MACD Line}, signal)')
    st.write(f"Stop Loss: {stop_loss}%")
    st.write(f"Take Profit: {take_profit}%")
    st.write(f"Shorting {'Enabled' if enable_shorting else 'Disabled'}")

# Bollinger Bands Strategy
class BollingerBandsStrategy(Strategy):
    window = 20
    window_dev = 2

    def init(self):
        self.window = st.session_state.get('bb_length', self.window)
        self.window_dev = st.session_state.get('bb_std_dev', self.window_dev)
        close_series = pd.Series(self.data.Close)
        indicator = ta.volatility.BollingerBands(close=close_series, window=self.window, window_dev=self.window_dev)
        self.upper = self.I(lambda: indicator.bollinger_hband())
        self.mid = self.I(lambda: indicator.bollinger_mavg())
        self.lower = self.I(lambda: indicator.bollinger_lband())

    def next(self):
        if self.data.Close[-1] < self.lower[-1]:
            self.buy()
        elif self.data.Close[-1] > self.upper[-1]:
            self.sell()

def bollinger_bands_viz():
    st.subheader('Bollinger Bands Parameters')
    length = st.slider('Length', key='bb_length', min_value=5, max_value=50, value=20)
    std_dev = st.slider('Number of Standard Deviations', key='bb_std_dev', min_value=1, max_value=3, value=2)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    bb = ta.volatility.BollingerBands(pd.Series(y), window=length, window_dev=std_dev)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_hband(), mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_mavg(), mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_lband(), mode='lines', name='Lower Band'))
    fig.update_layout(title='Bollinger Bands Visualization', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{Middle Band} = \text{SMA}(length)')
    st.latex(r'\text{Upper Band} = \text{Middle Band} + \text{std\_dev} \times \sigma')
    st.latex(r'\text{Lower Band} = \text{Middle Band} - \text{std\_dev} \times \sigma')

# Mean Reversion Strategy
class MeanReversionStrategy(Strategy):
    length = 30
    std_dev_multiplier = 2

    def init(self):
        self.length = st.session_state.get('mean_rev_length', self.length)
        self.std_dev_multiplier = st.session_state.get('std_dev_multiplier', self.std_dev_multiplier)
        self.sma = self.I(SMA, self.data.Close, self.length)

    def next(self):
        if self.data.Close[-1] < self.sma[-1] - self.std_dev_multiplier * self.data.Close.std():
            self.buy()
        elif self.data.Close[-1] > self.sma[-1] + self.std_dev_multiplier * self.data.Close.std():
            self.sell()

def mean_reversion_viz():
    st.subheader('Mean Reversion Parameters')
    length = st.slider('SMA Length', key='mean_rev_length', min_value=10, max_value=100, value=30)
    std_dev = st.slider('Standard Deviation Multiplier', key='std_dev_multiplier', min_value=1, max_value=5, value=2)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    sma = np.convolve(y, np.ones(length), mode='valid') / length
    std = np.std(y) * std_dev
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma, mode='lines', name='SMA'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma+std, mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma-std, mode='lines', name='Lower Band'))
    fig.update_layout(title='Mean Reversion Visualization', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{SMA} = \frac{1}{length} \sum_{i=0}^{length-1} P_{t-i}')
    st.latex(r'\text{Upper Band} = \text{SMA} + \text{std\_dev} \times \sigma')
    st.latex(r'\text{Lower Band} = \text{SMA} - \text{std\_dev} \times \sigma')

# Momentum Strategy
class MomentumStrategy(Strategy):
    period = 90

    def init(self):
        self.period = st.session_state.get('momentum_period', self.period)
        close_series = pd.Series(self.data.Close)
        self.momentum = self.I(ta.momentum.ROCIndicator(close=close_series, window=self.period).roc)

    def next(self):
        if self.momentum[-1] > 0:
            self.buy()
        elif self.momentum[-1] < 0:
            self.sell()

def momentum_viz():
    st.subheader('Momentum Parameters')
    period = st.slider('Momentum Period', key='momentum_period', min_value=10, max_value=200, value=90)
    
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    momentum = ta.momentum.ROCIndicator(pd.Series(y), window=period).roc()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[period:], y=momentum[period:], mode='lines', name='Momentum'))
    fig.add_shape(type="line", x0=period, y0=0, x1=100, y1=0, line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(title='Momentum Visualization', xaxis_title='Time', yaxis_title='Momentum')
    st.plotly_chart(fig)
    
    st.latex(r'\text{Momentum} = \frac{P_t - P_{t-period}}{P_{t-period}} \times 100')

# VWAP Strategy
class VwapStrategy(Strategy):
    def init(self):
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)
        close_series = pd.Series(self.data.Close)
        volume_series = pd.Series(self.data.Volume)
        typical_price = (high_series + low_series + close_series) / 3
        cum_vol_x_typical_price = (typical_price * volume_series).cumsum()
        cum_volume = volume_series.cumsum()
        self.vwap = self.I(lambda: cum_vol_x_typical_price / cum_volume)

    def next(self):
        if self.data.Close[-1] < self.vwap[-1]:
            self.buy()
        elif self.data.Close[-1] > self.vwap[-1]:
            self.sell()

def vwap_viz():
    st.write("The VWAP strategy uses the Volume Weighted Average Price. There are no parameters to adjust as it's calculated based on price and volume data.")
    
    x = np.arange(100)
    price = np.cumsum(np.random.randn(100)) + 100
    volume = np.random.randint(1000, 10000, 100)
    vwap = np.cumsum(price * volume) / np.cumsum(volume)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=price, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x, y=vwap, mode='lines', name='VWAP'))
    fig.update_layout(title='VWAP Visualization', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{VWAP} = \frac{\sum (Price \times Volume)}{\sum Volume}')

# Stochastic Strategy
class StochasticStrategy(Strategy):
    k_period = 14
    d_period = 3
    slowing = 3

    def init(self):
        self.k_period = st.session_state.get('stoch_k', self.k_period)
        self.d_period = st.session_state.get('stoch_d', self.d_period)
        self.slowing = st.session_state.get('stoch_slowing', self.slowing)
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)
        close_series = pd.Series(self.data.Close)
        self.stoch_indicator = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series, window=self.k_period, smooth_window=self.d_period)
        self.k = self.I(self.stoch_indicator.stoch)
        self.d = self.I(self.stoch_indicator.stoch_signal)

    def next(self):
        if self.k[-1] < 20 and self.d[-1] < 20:
            self.buy()
        elif self.k[-1] > 80 and self.d[-1] > 80:
            self.sell()

def stochastic_viz():
    st.subheader('Stochastic Parameters')
    k = st.slider('Stochastic %K', key='stoch_k', min_value=5, max_value=30, value=14)
    d = st.slider('Stochastic %D', key='stoch_d', min_value=3, max_value=30, value=3)
    
    x = np.arange(100)
    high = np.cumsum(np.random.randn(100)) + 110
    low = np.cumsum(np.random.randn(100)) + 90
    close = (high + low) / 2
    stoch = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), window=k, smooth_window=d)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=stoch.stoch(), mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=x, y=stoch.stoch_signal(), mode='lines', name='%D'))
    fig.add_shape(type="line", x0=0, y0=80, x1=100, y1=80, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=20, x1=100, y1=20, line=dict(color="green", width=2, dash="dash"))
    fig.update_layout(title='Stochastic Oscillator Visualization', xaxis_title='Time', yaxis_title='Value')
    st.plotly_chart(fig)
    
    st.latex(r'\%K = \frac{C_t - L_{14}}{H_{14} - L_{14}} \times 100')
    st.latex(r'\%D = \text{SMA}(\%K, 3)')




def strategy_params_and_viz(strategy):
    if strategy == 'Buy and Hold':
        buy_and_hold_viz()
    elif strategy == 'SMA Cross':
        sma_cross_viz()
    elif strategy == 'RSI':
        rsi_viz()
    elif strategy == 'MACD':
        macd_viz()
    elif strategy == 'Bollinger Bands':
        bollinger_bands_viz()
    elif strategy == 'Mean Reversion':
        mean_reversion_viz()
    elif strategy == 'Momentum':
        momentum_viz()
    elif strategy == 'VWAP':
        vwap_viz()
    elif strategy == 'Stochastic':
        stochastic_viz()
    else:
        st.error(f"Strategy '{strategy}' not implemented.")


st.set_page_config(layout="wide")

theme = st.sidebar.radio("Theme", ("Light", "Dark"))

# Custom CSS for light and dark themes
if theme == "Light":
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #414247;
    }
    .sidebar .sidebar-content {
        background-color: #414247;
    }
    h1, h2, h3, p, .stMarkdown {
        color: #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('📈 Advanced Stock Trading Strategy Backtester')

# Sidebar for user inputs
with st.sidebar:
    st.header('📊 Stock Selection')
    ticker = st.text_input('Enter stock ticker', value='AAPL')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2024-05-01'))
    with col2:
        end_date = st.date_input('End Date', value=pd.to_datetime('2024-07-01'))

    st.header('🧮 Strategy Selection')
    strategy_option = st.selectbox('Select Strategy', [
        'Buy and Hold', 'SMA Cross', 'RSI', 'MACD', 'Bollinger Bands', 'Mean Reversion', 'Momentum', 'VWAP', 'Stochastic'
    ])

    st.header('💰 Backtest Settings')
    cash = st.number_input('Starting Cash', min_value=1000, max_value=100000, value=10000)
    commission = st.slider('Commission (%)', min_value=0.0, max_value=0.05, value=0.002, step=0.001)






# Define strategy map
strategy_map = {
    'Buy and Hold': BuyAndHoldStrategy,
    'SMA Cross': SmaCross,
    'RSI': RsiStrategy,
    'MACD': MacdStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
    'Mean Reversion': MeanReversionStrategy,
    'Momentum': MomentumStrategy,
    'VWAP': VwapStrategy,
    'Stochastic': StochasticStrategy
}

# Main function to run the application
def run_backtest_app():
    ticker_data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if ticker_data is not None and not ticker_data.empty:
        # Create main layout
        col_strategy, col_results = st.columns([1, 3])

        with col_strategy:
            st.subheader("Strategy Parameters and Visualization")
            strategy_params_and_viz(strategy_option)

        with col_results:
            try:
                selected_strategy = strategy_map[strategy_option]
                bt = Backtest(ticker_data, selected_strategy, cash=cash, commission=commission)
                output = bt.run()

                display_results(output, ticker)
            except KeyError:
                st.error(f"Strategy '{strategy_option}' not implemented. Please select another strategy.")
    else:
        st.error("Error fetching data for the given ticker. Please check the ticker symbol and date range.")

def display_results(output, ticker):
    col_graphs, col_performance = st.columns([2, 1])

    with col_graphs:
        display_equity_curve(output, ticker)
        display_trade_log(output)

    with col_performance:
        display_performance_metrics(output)

def display_equity_curve(output, ticker):
    st.subheader('Equity Curve')
    fig_equity = go.Figure(data=[go.Scatter(x=output['_equity_curve'].index, y=output['_equity_curve']['Equity'], mode='lines')])
    fig_equity.update_layout(title=f'{ticker} Equity Curve', xaxis_title='Date', yaxis_title='Equity', height=400)
    st.plotly_chart(fig_equity, use_container_width=True)

def display_trade_log(output):
    st.subheader('Trade Log')
    st.dataframe(output['_trades'], use_container_width=True, height=300)

def display_performance_metrics(output):
    st.subheader('Performance Metrics')
    
    metrics = prepare_metrics(output)
    df_metrics = pd.DataFrame(metrics, index=['Value']).T
    
    display_key_metrics(df_metrics)
    display_return_comparison_chart(df_metrics)
    display_radar_chart(df_metrics)
    
    with st.expander("View All Metrics"):
        st.dataframe(df_metrics, use_container_width=True)

def prepare_metrics(output):
    key_metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
                   'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
                   'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
                   'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
                   'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
                   'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]']
    
    metrics = output.drop(['_strategy', '_equity_curve', '_trades'])
    return {k: metrics[k] for k in key_metrics if k in metrics}

def display_key_metrics(df_metrics):
    st.metric("Total Return", f"{df_metrics.loc['Return [%]', 'Value']:.2f}%")
    st.metric("Sharpe Ratio", f"{df_metrics.loc['Sharpe Ratio', 'Value']:.2f}")
    st.metric("Max Drawdown", f"{df_metrics.loc['Max. Drawdown [%]', 'Value']:.2f}%")
    
    strategy_return = df_metrics.loc['Return [%]', 'Value']
    bh_return = df_metrics.loc['Buy & Hold Return [%]', 'Value']
    outperformance = strategy_return - bh_return
    st.metric("Strategy vs. Buy & Hold", f"{outperformance:.2f}%", 
              delta=f"{outperformance:.2f}%", delta_color="normal")
    
    st.metric("Win Rate", f"{df_metrics.loc['Win Rate [%]', 'Value']:.2f}%")

def display_return_comparison_chart(df_metrics):
    strategy_return = df_metrics.loc['Return [%]', 'Value']
    bh_return = df_metrics.loc['Buy & Hold Return [%]', 'Value']
    
    fig_return_comparison = go.Figure(data=[
        go.Bar(name='Strategy', x=['Return'], y=[strategy_return]),
        go.Bar(name='Buy & Hold', x=['Return'], y=[bh_return])
    ])
    fig_return_comparison.update_layout(title='Strategy vs. Buy & Hold Return Comparison')
    st.plotly_chart(fig_return_comparison, use_container_width=True)

def display_radar_chart(df_metrics):
    radar_metrics = ['Return [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Win Rate [%]']
    radar_values = [df_metrics.loc[metric, 'Value'] for metric in radar_metrics]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=radar_values,
        theta=radar_metrics,
        fill='toself'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(radar_values)])
        ),
        showlegend=False,
        title='Strategy Performance Radar'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Call the main function
if __name__ == "__main__":
    run_backtest_app()



# Add an explanation of the selected strategy
st.markdown("---")
st.subheader("📚 Strategy Explanation")
strategy_explanations = {
    'Buy and Hold': "This strategy simply buys the stock at the beginning and holds it until the end of the period.",
    'SMA Cross': "This strategy uses two Simple Moving Averages (SMA) and generates buy/sell signals when they cross.",
    'RSI': "The Relative Strength Index (RSI) strategy buys when the RSI is oversold and sells when it's overbought.",
    'MACD': "The Moving Average Convergence Divergence (MACD) strategy generates signals based on the crossover of the MACD line and the signal line.",
    'Bollinger Bands': "This strategy uses Bollinger Bands to identify overbought and oversold conditions.",
    'Mean Reversion': "The Mean Reversion strategy assumes that prices and other indicators tend to move back towards their average over time.",
    'Momentum': "The Momentum strategy is based on the idea that trends in stock prices tend to continue for some time.",
    'VWAP': "The Volume Weighted Average Price (VWAP) strategy uses the VWAP as a benchmark for trading decisions.",
    'Stochastic': "The Stochastic Oscillator strategy uses overbought and oversold levels to generate trading signals."
}
st.write(strategy_explanations.get(strategy_option, "No explanation available for this strategy."))

# Add a footer with a disclaimer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Always do your own research before making investment decisions.")