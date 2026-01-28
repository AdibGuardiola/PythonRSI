import plotly.graph_objects as go
import streamlit as st
import time
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gold RSI Bot", page_icon="🤖", layout="wide")

st.title("🤖 Bot RSI Oro (XAU/USD) - M15")
st.caption("Estrategia: Cruce de Medias + RSI (30/70) en gráficos de 15 minutos.")

# ==========================================
# CREDENTIALS
# ==========================================
# Hardcoded credentials as requested by user
TOKEN = "6075391597:AAFi28sadDJmqOrgvKGlbMnMK5hk8A1JFQY"
CHAT_ID = "909954663"

if not TOKEN or not CHAT_ID:
    st.error("Credenciales no configuradas.")
    st.stop()

# ==========================================
# TRADING LOGIC FUNCTIONS
# ==========================================
import yfinance as yf

# ... [imports remain the same] ...

# ==========================================
# TRADING LOGIC FUNCTIONS
# ==========================================
def get_klines(symbol="GC=F", interval="15m", period="5d"):
    """Fetch candlestick data from Yahoo Finance (Gold Futures)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
            
        # Reset index to get 'Date'/'Datetime' as a column
        df = df.reset_index()
        
        # Standardize column names depending on YF version (Datetime/Date -> time)
        col_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_map)
        
        # Ensure we have time and ohlc
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'time'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
            
        # Ensure UTC-naive for compatibility if needed, or keep aware
        # Streamlit/Plotly handles aware, but for simplicity:
        df['time'] = pd.to_datetime(df['time'])
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return pd.DataFrame()
                
        return df
    except Exception as e:
        print(f"Error getting klines from YF: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate RSI and EMAs."""
    if df.empty:
        return df
        
    # RSI (14)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    return df

def send_telegram(token, chat_id, msg):
    """Send message to Telegram."""
    if not token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending telegram: {e}")
        return False

def check_market(token, chat_id, last_alert_state):
    """
    Checks market data and calls alerting logic.
    Returns: (new_alert_state, status_message, dataframe_for_plot)
    """
    df = get_klines()
    if df.empty:
        return last_alert_state, "Error fetching data from Yahoo Finance", pd.DataFrame()
        
    df = calculate_metrics(df)
    last_row = df.iloc[-1]
    
    rsi_value = last_row['RSI']
    ema9 = last_row['EMA_9']
    ema21 = last_row['EMA_21']
    price = last_row['close']
    
    status_msg = f"RSI: {rsi_value:.2f} | EMA9: {ema9:.2f} | EMA21: {ema21:.2f}"
    
    # Alert Logic
    alert_triggered = False
    alert_text = ""
    
    # RSI Condition
    if rsi_value > 70:
        alert_text += f"\n🚨 SOBRECOMPRA (RSI > 70): {rsi_value:.2f}"
        alert_triggered = True
    elif rsi_value < 30:
        alert_text += f"\n🚨 SOBREVENTA (RSI < 30): {rsi_value:.2f}"
        alert_triggered = True
        
    # Validation for new msg
    if alert_triggered and not last_alert_state:
        full_msg = f"🔥 ALERTA ORO (M15)\nPrecio: {price}\n{alert_text}"
        sent = send_telegram(token, chat_id, full_msg)
        return True, f"Alert Sent! {status_msg}", df
            
    if not alert_triggered and last_alert_state:
        # Reset if conditions normalize
        return False, f"Conditions Normalized. {status_msg}", df
        
    return last_alert_state, status_msg, df

# ==========================================
# MAIN APP EXECUTION
# ==========================================

# 1. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Panel de Control")
    if st.button("🧪 Probar Alerta Telegram"):
        if send_telegram(TOKEN, CHAT_ID, "🧪 Prueba de conexión desde Streamlit Bot"):
            st.success("✅ Mensaje enviado correctamente.")
        else:
            st.error("❌ Error al enviar mensaje. Revisa credenciales.")
    
    st.divider()
    st.info("El bot se ejecuta automáticamente cada 5 minutos.")

# 2. State Initialization
if "last_alert" not in st.session_state:
    st.session_state.last_alert = False

if "logs" not in st.session_state:
    st.session_state.logs = []

# 3. UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Estado")
    status_metric = st.empty()
    alert_status = st.empty()

with col2:
    st.subheader("Logs (Últimos 20)")
    log_area = st.empty()

st.divider()
chart_placeholder = st.empty()

st.divider()
st.info("Bot is running... Do not close this tab if running locally.")

# 4. Main Loop
while True:
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        # Check Market
        new_alert, msg, df = check_market(TOKEN, CHAT_ID, st.session_state.last_alert)
        
        # Update State
        st.session_state.last_alert = new_alert
        
        # Update Log
        log_entry = f"[{timestamp}] {msg}"
        st.session_state.logs.insert(0, log_entry)
        if len(st.session_state.logs) > 20:
            st.session_state.logs.pop()
            
        # UI Updates
        status_metric.markdown(f"**Hora:** {timestamp}")
        alert_status.markdown(f"**Estado Alerta:** {'🔴 ACTIVA' if new_alert else '🟢 Normal'}")
        
        # Visualization
        if not df.empty:
            with chart_placeholder:
                # 1. Price Chart (Candlestick)
                fig_price = go.Figure()
                
                # Candlestick
                fig_price.add_trace(go.Candlestick(
                    x=df['time'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'],
                    name='Precio'
                ))
                
                # EMAs
                fig_price.add_trace(go.Scatter(x=df['time'], y=df['EMA_9'], name='EMA 9', line=dict(color='cyan', width=1.5)))
                fig_price.add_trace(go.Scatter(x=df['time'], y=df['EMA_21'], name='EMA 21', line=dict(color='magenta', width=1.5)))
                
                fig_price.update_layout(
                    title="Oro (PAXG/USDT) - M15",
                    template="plotly_dark",
                    height=500,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_price, use_container_width=True, key=f"price_{timestamp}")

                # 2. RSI Chart (Improved)
                fig_rsi = go.Figure()
                
                # Background Band (30-70)
                fig_rsi.add_shape(type="rect",
                    x0=df['time'].iloc[0], x1=df['time'].iloc[-1],
                    y0=30, y1=70,
                    fillcolor="rgba(128, 0, 128, 0.2)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # RSI Line
                fig_rsi.add_trace(go.Scatter(
                    x=df['time'], y=df['RSI'],
                    name='RSI',
                    line=dict(color='#FFA500', width=2)
                ))
                
                # Threshold Lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobreventa (30)")
                
                fig_rsi.update_layout(
                    title="RSI (14)",
                    template="plotly_dark",
                    height=250,
                    yaxis_range=[0, 100],
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_rsi, use_container_width=True, key=f"rsi_{timestamp}")

        log_text = "\n".join(st.session_state.logs)
        log_area.text_area("Logs", log_text, height=200)
        
    except Exception as e:
        st.error(f"Error: {e}")
        time.sleep(60)
        continue

    # Wait for next cycle
    time.sleep(300)
