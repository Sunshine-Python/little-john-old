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
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    sixty_days_ago = datetime.now() - timedelta(days=60)
    if start_date < sixty_days_ago:
        print(f"Warning: Adjusted start date to {sixty_days_ago.strftime('%Y-%m-%d')} due to YFinance limitations.")
        start_date = sixty_days_ago
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if data.empty:
        return None
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Drop rows with zero volume (non-trading days)
    data = data[data['Volume'] > 0]
    
    data = data.reset_index()
    
    return data



def buy_and_hold_viz(data):
    st.write("The Buy and Hold strategy simply buys the stock at the beginning of the period and holds it until the end. There are no parameters to adjust.")
    
    fig = go.Figure(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Price'))
    fig.update_layout(title='Buy and Hold Visualization', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

def sma_cross_viz(data):
    st.subheader('SMA Cross Parameters')
    n1 = st.slider('Short Window (n1)', key='sma_n1', min_value=5, max_value=50, value=10)
    n2 = st.slider('Long Window (n2)', key='sma_n2', min_value=20, max_value=100, value=20)
    stop_loss = st.slider('Stop Loss (%)', key='sma_stop_loss', min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    take_profit = st.slider('Take Profit (%)', key='sma_take_profit', min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    enable_shorting = st.checkbox('Enable Shorting', key='sma_enable_shorting', value=True)
    
    short_sma = data['Close'].rolling(window=n1).mean()
    long_sma = data['Close'].rolling(window=n2).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=short_sma, mode='lines', name=f'SMA({n1})'))
    fig.add_trace(go.Scatter(x=data['Date'], y=long_sma, mode='lines', name=f'SMA({n2})'))
    fig.update_layout(title='SMA Cross Visualization', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{Short SMA} = \frac{1}{n1} \sum_{i=0}^{n1-1} P_{t-i}')
    st.latex(r'\text{Long SMA} = \frac{1}{n2} \sum_{i=0}^{n2-1} P_{t-i}')
    st.write(f"Stop Loss: {stop_loss}%")
    st.write(f"Take Profit: {take_profit}%")
    st.write(f"Shorting {'Enabled' if enable_shorting else 'Disabled'}")

def macd_viz(data):
    st.subheader('MACD Parameters')
    fast = st.slider('Fast Length', key='macd_fast', min_value=5, max_value=50, value=12)
    slow = st.slider('Slow Length', key='macd_slow', min_value=20, max_value=100, value=26)
    signal = st.slider('Signal Length', key='macd_signal', min_value=5, max_value=50, value=9)
    stop_loss = st.slider('Stop Loss (%)', key='macd_stop_loss', min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    take_profit = st.slider('Take Profit (%)', key='macd_take_profit', min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    enable_shorting = st.checkbox('Enable Shorting', key='macd_enable_shorting', value=True)
    
    macd = ta.trend.MACD(data['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=macd.macd(), mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=macd.macd_signal(), mode='lines', name='Signal'))
    fig.add_bar(x=data['Date'], y=macd.macd_diff(), name='Histogram')
    fig.update_layout(title='MACD Visualization', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)
    
    st.latex(r'\text{MACD Line} = \text{EMA}(fast) - \text{EMA}(slow)')
    st.latex(r'\text{Signal Line} = \text{EMA}(\text{MACD Line}, signal)')
    st.write(f"Stop Loss: {stop_loss}%")
    st.write(f"Take Profit: {take_profit}%")
    st.write(f"Shorting {'Enabled' if enable_shorting else 'Disabled'}")

def bollinger_bands_viz(data):
    st.subheader('Bollinger Bands Parameters')
    length = st.slider('Length', key='bb_length', min_value=5, max_value=50, value=20)
    std_dev = st.slider('Number of Standard Deviations', key='bb_std_dev', min_value=1, max_value=3, value=2)
    
    bb = ta.volatility.BollingerBands(data['Close'], window=length, window_dev=std_dev)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=bb.bollinger_hband(), mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=data['Date'], y=bb.bollinger_mavg(), mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=data['Date'], y=bb.bollinger_lband(), mode='lines', name='Lower Band'))
    fig.update_layout(title='Bollinger Bands Visualization', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.latex(r'\text{Middle Band} = \text{SMA}(length)')
    st.latex(r'\text{Upper Band} = \text{Middle Band} + \text{std\_dev} \times \sigma')
    st.latex(r'\text{Lower Band} = \text{Middle Band} - \text{std\_dev} \times \sigma')

class BuyAndHoldStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

class MacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9

    def init(self):
        self.macd = self.I(ta.trend.MACD, self.data.Close, 
                           window_slow=self.slow, window_fast=self.fast, window_sign=self.signal)

    def next(self):
        if crossover(self.macd.macd(), self.macd.signal()):
            self.buy()
        elif crossover(self.macd.signal(), self.macd.macd()):
            self.sell()

class BollingerBandsStrategy(Strategy):
    window = 20
    window_dev = 2

    def init(self):
        self.bb = self.I(ta.volatility.BollingerBands, self.data.Close, 
                         window=self.window, window_dev=self.window_dev)

    def next(self):
        if self.data.Close[-1] < self.bb.lower[-1]:
            self.buy()
        elif self.data.Close[-1] > self.bb.upper[-1]:
            self.sell()





def strategy_params_and_viz(strategy, ticker_data):
    if strategy == 'Buy and Hold':
        buy_and_hold_viz(ticker_data)
    elif strategy == 'SMA Cross':
        sma_cross_viz(ticker_data)
    elif strategy == 'MACD':
        macd_viz(ticker_data)
    elif strategy == 'Bollinger Bands':
        bollinger_bands_viz(ticker_data)
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
        'Buy and Hold', 'SMA Cross', 'MACD', 'Bollinger Bands'
    ])

    st.header('💰 Backtest Settings')
    cash = st.number_input('Starting Cash', min_value=1000, max_value=100000, value=10000)
    commission = st.slider('Commission (%)', min_value=0.0, max_value=0.05, value=0.002, step=0.001)

strategy_map = {
    'Buy and Hold': BuyAndHoldStrategy,
    'SMA Cross': SmaCross,
    'MACD': MacdStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
}

def run_backtest_app():
    ticker_data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if ticker_data is not None and not ticker_data.empty:
        plot_stock_price_and_volume(ticker_data)

        col_strategy, col_results = st.columns([1, 3])

        with col_strategy:
            st.subheader("Strategy Parameters and Visualization")
            strategy_params_and_viz(strategy_option, ticker_data)

        with col_results:
            try:
                selected_strategy = strategy_map[strategy_option]
                
                # Create a new instance of the strategy with updated parameters
                if strategy_option == 'SMA Cross':
                    strategy_instance = selected_strategy(
                        n1=st.session_state.get('sma_n1', 10),
                        n2=st.session_state.get('sma_n2', 20)
                    )
                elif strategy_option == 'MACD':
                    strategy_instance = selected_strategy(
                        fast=st.session_state.get('macd_fast', 12),
                        slow=st.session_state.get('macd_slow', 26),
                        signal=st.session_state.get('macd_signal', 9)
                    )
                elif strategy_option == 'Bollinger Bands':
                    strategy_instance = selected_strategy(
                        window=st.session_state.get('bb_length', 20),
                        window_dev=st.session_state.get('bb_std_dev', 2)
                    )
                else:  # Buy and Hold
                    strategy_instance = selected_strategy()

                bt = Backtest(ticker_data.set_index('Date'), strategy_instance, cash=cash, commission=commission)
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

def plot_stock_price_and_volume(data):
    st.subheader('Stock Price and Volume')
    
    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name='Price',
        line=dict(color='blue')
    ))

    # Add volume trace
    fig.add_trace(go.Bar(
        x=data['Date'],
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color='rgba(0,0,0,0.2)')
    ))

    # Set layout
    fig.update_layout(
        title='Stock Price and Trading Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_backtest_app()

st.markdown("---")
st.subheader("📚 Strategy Explanation")
strategy_explanations = {
    'Buy and Hold': "This strategy simply buys the stock at the beginning and holds it until the end of the period.",
    'SMA Cross': "This strategy uses two Simple Moving Averages (SMA) and generates buy/sell signals when they cross.",
    'MACD': "The Moving Average Convergence Divergence (MACD) strategy generates signals based on the crossover of the MACD line and the signal line.",
    'Bollinger Bands': "This strategy uses Bollinger Bands to identify overbought and oversold conditions.",
}
st.write(strategy_explanations.get(strategy_option, "No explanation available for this strategy."))

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Always do your own research before making investment decisions.")