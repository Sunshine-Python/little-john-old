import streamlit as st
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Include custom CSS
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Custom CSS to reduce spacing
st.markdown("""
<style>
.css-18e3th9 {
    padding: 2px 0px 2px 0px;
}
</style>
""", unsafe_allow_html=True)


def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        data = data.drop(columns=['Adj Close'])
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Buy and Hold Strategy
class BuyAndHoldStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()

def buy_and_hold_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', name='Stock Price'))
    fig.update_layout(title='Buy and Hold Visualization', xaxis_title='Time', yaxis_title='Price', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

# SMA Cross Strategy
class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.n1 = st.session_state.get('sma_n1', self.n1)
        self.n2 = st.session_state.get('sma_n2', self.n2)
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

def sma_cross_params():
    st.slider('Short Window (n1)', key='sma_n1', min_value=5, max_value=50, value=10)
    st.slider('Long Window (n2)', key='sma_n2', min_value=20, max_value=100, value=20)

def sma_cross_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    n1 = st.session_state.get('sma_n1', 10)
    n2 = st.session_state.get('sma_n2', 20)
    short_sma = np.convolve(y, np.ones(n1), mode='valid') / n1
    long_sma = np.convolve(y, np.ones(n2), mode='valid') / n2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x[n1-1:], y=short_sma, mode='lines', name=f'SMA({n1})'))
    fig.add_trace(go.Scatter(x=x[n2-1:], y=long_sma, mode='lines', name=f'SMA({n2})'))
    fig.update_layout(title='SMA Cross Visualization', xaxis_title='Time', yaxis_title='Price', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

# RSI Strategy
class RsiStrategy(Strategy):
    length = 14
    overbought = 70
    oversold = 30

    def init(self):
        self.length = st.session_state.get('rsi_length', self.length)
        self.overbought = st.session_state.get('rsi_overbought', self.overbought)
        self.oversold = st.session_state.get('rsi_oversold', self.oversold)
        close_prices = pd.Series(self.data.Close)
        self.rsi = self.I(ta.momentum.RSIIndicator(close=close_prices, window=self.length).rsi)

    def next(self):
        if self.rsi[-1] < self.oversold:
            self.buy()
        elif self.rsi[-1] > self.overbought:
            self.sell()

def rsi_params():
    st.slider('RSI Length', key='rsi_length', min_value=5, max_value=50, value=14)
    st.slider('Overbought Level', key='rsi_overbought', min_value=70, max_value=90, value=70)
    st.slider('Oversold Level', key='rsi_oversold', min_value=10, max_value=30, value=30)

def rsi_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    length = st.session_state.get('rsi_length', 14)
    rsi = ta.momentum.RSIIndicator(pd.Series(y), window=length).rsi()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rsi, mode='lines', name='RSI'))
    fig.add_shape(type="line", x0=0, y0=st.session_state.get('rsi_overbought', 70), x1=100, y1=st.session_state.get('rsi_overbought', 70), line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=st.session_state.get('rsi_oversold', 30), x1=100, y1=st.session_state.get('rsi_oversold', 30), line=dict(color="green", width=2, dash="dash"))
    fig.update_layout(title='RSI Visualization', xaxis_title='Time', yaxis_title='RSI', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

# MACD Strategy
class MacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9

    def init(self):
        self.fast = st.session_state.get('macd_fast', self.fast)
        self.slow = st.session_state.get('macd_slow', self.slow)
        self.signal = st.session_state.get('macd_signal', self.signal)
        close_series = pd.Series(self.data.Close)
        self.macd_indicator = ta.trend.MACD(close_series, window_slow=self.slow, window_fast=self.fast, window_sign=self.signal)
        self.macd_line = self.I(self.macd_indicator.macd)
        self.signal_line = self.I(self.macd_indicator.macd_signal)
        self.macd_diff = self.I(self.macd_indicator.macd_diff)

    def next(self):
        if crossover(self.macd_line, self.signal_line):
            self.buy()
        elif crossover(self.signal_line, self.macd_line):
            self.sell()

def macd_params():
    st.slider('Fast Length', key='macd_fast', min_value=5, max_value=50, value=12)
    st.slider('Slow Length', key='macd_slow', min_value=20, max_value=100, value=26)
    st.slider('Signal Length', key='macd_signal', min_value=5, max_value=50, value=9)

def macd_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    fast = st.session_state.get('macd_fast', 12)
    slow = st.session_state.get('macd_slow', 26)
    signal = st.session_state.get('macd_signal', 9)
    macd = ta.trend.MACD(pd.Series(y), window_slow=slow, window_fast=fast, window_sign=signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=macd.macd(), mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=x, y=macd.macd_signal(), mode='lines', name='Signal'))
    fig.add_bar(x=x, y=macd.macd_diff(), name='Histogram')
    fig.update_layout(title='MACD Visualization', xaxis_title='Time', yaxis_title='Value', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

# Bollinger Bands Strategy
class BollingerBandsStrategy(Strategy):
    window = 20
    window_dev = 2

    def init(self):
        self.window = st.session_state.get('bb_length', self.window)
        self.window_dev = st.session_state.get('bb_std_dev', self.window_dev)
        close_series = pd.Series(self.data.Close)
        self.bb_indicator = ta.volatility.BollingerBands(close=close_series, window=self.window, window_dev=self.window_dev)
        self.upper = self.I(self.bb_indicator.bollinger_hband)
        self.mid = self.I(self.bb_indicator.bollinger_mavg)
        self.lower = self.I(self.bb_indicator.bollinger_lband)

    def next(self):
        if self.data.Close[-1] < self.lower[-1]:
            self.buy()
        elif self.data.Close[-1] > self.upper[-1]:
            self.sell()

def bollinger_bands_params():
    st.slider('Length', key='bb_length', min_value=5, max_value=50, value=20)
    st.slider('Number of Standard Deviations', key='bb_std_dev', min_value=1, max_value=3, value=2)

def bollinger_bands_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    length = st.session_state.get('bb_length', 20)
    std_dev = st.session_state.get('bb_std_dev', 2)
    bb = ta.volatility.BollingerBands(pd.Series(y), window=length, window_dev=std_dev)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_hband(), mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_mavg(), mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=x, y=bb.bollinger_lband(), mode='lines', name='Lower Band'))
    fig.update_layout(title='Bollinger Bands Visualization', xaxis_title='Time', yaxis_title='Price', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

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

def mean_reversion_params():
    st.slider('SMA Length', key='mean_rev_length', min_value=10, max_value=100, value=30)
    st.slider('Standard Deviation Multiplier', key='std_dev_multiplier', min_value=1, max_value=5, value=2)

def mean_reversion_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    length = st.session_state.get('mean_rev_length', 30)
    std_dev = st.session_state.get('std_dev_multiplier', 2)
    sma = np.convolve(y, np.ones(length), mode='valid') / length
    std = np.std(y) * std_dev

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma, mode='lines', name='SMA'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma+std, mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=x[length-1:], y=sma-std, mode='lines', name='Lower Band'))
    fig.update_layout(title='Mean Reversion Visualization', xaxis_title='Time', yaxis_title='Price', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

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

def momentum_params():
    st.slider('Momentum Period', key='momentum_period', min_value=10, max_value=200, value=90)

def momentum_viz():
    x = np.arange(100)
    y = np.cumsum(np.random.randn(100)) + 100
    period = st.session_state.get('momentum_period', 90)
    momentum = ta.momentum.ROCIndicator(pd.Series(y), window=period).roc()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[period:], y=momentum[period:], mode='lines', name='Momentum'))
    fig.add_shape(type="line", x0=period, y0=0, x1=100, y1=0, line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(title='Momentum Visualization', xaxis_title='Time', yaxis_title='Momentum', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

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
    x = np.arange(100)
    price = np.cumsum(np.random.randn(100)) + 100
    volume = np.random.randint(1000, 10000, 100)
    vwap = np.cumsum(price * volume) / np.cumsum(volume)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=price, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=x, y=vwap, mode='lines', name='VWAP'))
    fig.update_layout(title='VWAP Visualization', xaxis_title='Time', yaxis_title='Price', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

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

def stochastic_params():
    st.slider('Stochastic %K', key='stoch_k', min_value=5, max_value=30, value=14)
    st.slider('Stochastic %D', key='stoch_d', min_value=3, max_value=30, value=3)

def stochastic_viz():
    x = np.arange(100)
    high = np.cumsum(np.random.randn(100)) + 110
    low = np.cumsum(np.random.randn(100)) + 90
    close = (high + low) / 2
    k = st.session_state.get('stoch_k', 14)
    d = st.session_state.get('stoch_d', 3)
    stoch = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), window=k, smooth_window=d)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=stoch.stoch(), mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=x, y=stoch.stoch_signal(), mode='lines', name='%D'))
    fig.add_shape(type="line", x0=0, y0=80, x1=100, y1=80, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=20, x1=100, y1=20, line=dict(color="green", width=2, dash="dash"))
    fig.update_layout(title='Stochastic Oscillator Visualization', xaxis_title='Time', yaxis_title='Value', height=300, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
    st.plotly_chart(fig)

def strategy_params_and_viz(strategy):
    if strategy == 'Buy and Hold':
        pass  # No parameters for Buy and Hold
    elif strategy == 'SMA Cross':
        sma_cross_params()
    elif strategy == 'RSI':
        rsi_params()
    elif strategy == 'MACD':
        macd_params()
    elif strategy == 'Bollinger Bands':
        bollinger_bands_params()
    elif strategy == 'Mean Reversion':
        mean_reversion_params()
    elif strategy == 'Momentum':
        momentum_params()
    elif strategy == 'VWAP':
        pass  # No parameters for VWAP
    elif strategy == 'Stochastic':
        stochastic_params()
    else:
        st.error(f"Strategy '{strategy}' not implemented.")

def strategy_viz(strategy):
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

def strategy_description(strategy):
    descriptions = {
        'Buy and Hold': "The Buy and Hold strategy simply buys the stock at the beginning of the period and holds it until the end. There are no parameters to adjust.",
        'SMA Cross': "This strategy uses two Simple Moving Averages (SMA) and generates buy/sell signals when they cross.",
        'RSI': "The Relative Strength Index (RSI) strategy buys when the RSI is oversold and sells when it's overbought.",
        'MACD': "The Moving Average Convergence Divergence (MACD) strategy generates signals based on the crossover of the MACD line and the signal line.",
        'Bollinger Bands': "This strategy uses Bollinger Bands to identify overbought and oversold conditions.",
        'Mean Reversion': "The Mean Reversion strategy assumes that prices and other indicators tend to move back towards their average over time.",
        'Momentum': "The Momentum strategy is based on the idea that trends in stock prices tend to continue for some time.",
        'VWAP': "The VWAP strategy uses the Volume Weighted Average Price. There are no parameters to adjust as it's calculated based on price and volume data.",
        'Stochastic': "The Stochastic Oscillator strategy uses overbought and oversold levels to generate trading signals."
    }
    st.write(descriptions.get(strategy, "No explanation available for this strategy."))

# Custom CSS to improve the app's appearance
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

st.title('Advanced Stock Trading Strategy Backtester')

# Sidebar for user inputs
logo_url = "little-john-logo.png"
st.sidebar.image(logo_url, use_column_width=True)

with st.sidebar:
    st.header('📊 Stock Selection')
    ticker = st.text_input('Enter stock ticker', value='AAPL')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
    with col2:
        end_date = st.date_input('End Date', value=pd.to_datetime('2020-01-01'))

    st.header('🧮 Strategy Selection')
    strategy_option = st.selectbox('Select Strategy', [
        'Buy and Hold', 'SMA Cross', 'RSI', 'MACD', 'Bollinger Bands', 'Mean Reversion', 'Momentum', 'VWAP', 'Stochastic'
    ])

    st.header('💰 Backtest Settings')
    cash = st.number_input('Starting Cash', min_value=1000, max_value=100000, value=10000)
    commission = st.slider('Commission (%)', min_value=0.0, max_value=0.05, value=0.002, step=0.001)


# Function to shape metric containers
def display_metric(label, value, delta=None, delta_color="normal", show_arrow=False):
    arrow = ""
    if show_arrow and delta is not None:
        delta_value = float(delta.replace('%', ''))
        if delta_value >= 0:
            arrow = "&#9650;"  # Up arrow
            color = "green"
        else:
            arrow = "&#9660;"  # Down arrow
            color = "red"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} <span style="color: {color};">{arrow}</span></div>
        </div>
        """, unsafe_allow_html=True)
    elif delta:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color: {'red' if delta_color == 'inverse' else 'green'};">{delta}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)





# Main content area
ticker_data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if ticker_data is not None and not ticker_data.empty:
    # Map strategy names to classes
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

    try:
        selected_strategy = strategy_map[strategy_option]
        bt = Backtest(ticker_data, selected_strategy, cash=cash, commission=commission)
        output = bt.run()

        # First row: Strategy Parameters and Visualization
        row1_col1, row1_col2 = st.columns([1, 1])

        with row1_col1:
            st.subheader("Strategy Parameters")
            strategy_description(strategy_option)
            strategy_params_and_viz(strategy_option)

        with row1_col2:
            st.subheader("Visualization")
            strategy_viz(strategy_option)

        # Second row: Performance Metrics
        st.subheader('Performance Metrics')
        key_metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
                        'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
                        'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
                        'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
                        'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
                        'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]']

        metrics = output.drop(['_strategy', '_equity_curve', '_trades'])
        selected_metrics = {k: metrics[k] for k in key_metrics if k in metrics}
        df_metrics = pd.DataFrame(selected_metrics, index=['Value']).T

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            display_metric("Total Return", f"{df_metrics.loc['Return [%]', 'Value']:.2f}%")
        with col2:
            display_metric("Sharpe Ratio", f"{df_metrics.loc['Sharpe Ratio', 'Value']:.2f}")
        with col3:
            display_metric("Max Drawdown", f"{df_metrics.loc['Max. Drawdown [%]', 'Value']:.2f}%")
        with col4:
            strategy_return = df_metrics.loc['Return [%]', 'Value']
            bh_return = df_metrics.loc['Buy & Hold Return [%]', 'Value']
            outperformance = strategy_return - bh_return
            display_metric("Strategy vs. Buy & Hold", f"{outperformance:.2f}%", delta=f"{outperformance:.2f}%", delta_color="inverse", show_arrow=True)
        with col5:
            display_metric("Win Rate", f"{df_metrics.loc['Win Rate [%]', 'Value']:.2f}%")


        # Third row: equity curve, comparison graph & strategy performance radar
        row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1])

        with row3_col1:
            st.subheader('Equity Curve')
            fig_equity = go.Figure(data=[go.Scatter(x=output['_equity_curve'].index, y=output['_equity_curve']['Equity'], mode='lines')])
            fig_equity.update_layout(title=f'{ticker} Equity Curve', xaxis_title='Date', yaxis_title='Equity', height=350, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
            st.plotly_chart(fig_equity, use_container_width=True)

        with row3_col2:
            st.subheader('Comparison Graph')
            fig_return_comparison = go.Figure(data=[
                go.Bar(name='Strategy', x=['Return'], y=[strategy_return]),
                go.Bar(name='Buy & Hold', x=['Return'], y=[bh_return])
            ])
            fig_return_comparison.update_layout(title='Strategy vs. Buy & Hold Return Comparison', height=350, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30))
            st.plotly_chart(fig_return_comparison, use_container_width=True)

        with row3_col3:
            st.subheader('Strategy Performance')
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
                height=350,margin=dict(l=30, r=30),plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                title='Strategy Performance Radar'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Fourth row: trade log as pop-up (same optic as "view all metrics")
        with st.expander("View Trade Log"):
            st.dataframe(output['_trades'], use_container_width=True, height=300)

        # Display all metrics in an expandable section
        with st.expander("View All Metrics"):
            st.dataframe(df_metrics, use_container_width=True)

    except KeyError:
        st.error(f"Strategy '{strategy_option}' not implemented. Please select another strategy.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


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
