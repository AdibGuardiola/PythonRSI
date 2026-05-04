import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(page_title="RSI Cross Triangles (MT5)", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_series.fillna(0)


def download_fx_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, multi_level_index=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna().copy()


def build_indicator(df: pd.DataFrame, rsi_period: int, fast_ma: int, slow_ma: int, trend_fast_ma: int, trend_slow_ma: int, vol_ma: int, adx_period: int = 14, use_trend_filter: bool = True, fix_wicks: bool = False) -> pd.DataFrame:
    data = df.copy()
    data["RSI"] = rsi(data["Close"], rsi_period)
    data["ADX"] = adx(data["High"], data["Low"], data["Close"], adx_period)
    
    if fast_ma > 1:
        data["RSI_FAST"] = data["RSI"].rolling(fast_ma).mean()
    else:
        data["RSI_FAST"] = data["RSI"]
        
    data["RSI_SLOW"] = data["RSI"].ewm(span=slow_ma, adjust=False).mean()
    data["TREND_FAST_MA"] = data["Close"].rolling(trend_fast_ma).mean()
    data["TREND_SLOW_MA"] = data["Close"].rolling(trend_slow_ma).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()
    
    # Volume calculation
    # Forex volume on Yahoo is tick volume, but still useful for relative activity
    if "Volume" not in data.columns or data["Volume"].sum() == 0:
        volatilidad = data["High"] - data["Low"]
        data["Volume"] = volatilidad * 100000 
    
    if fix_wicks:
        # Corrección del "efecto peine" en Yahoo Finance Cross FX (como AUDCHF)
        body = (data["Close"] - data["Open"]).abs()
        hl = data["High"] - data["Low"]
        excess = (hl - body).clip(lower=0)
        spread_noise = excess.rolling(window=12, min_periods=1).median() * 0.8
        
        upper = data[["Open", "Close"]].max(axis=1)
        lower = data[["Open", "Close"]].min(axis=1)
        
        data["High"] = np.maximum(data["High"] - spread_noise * 0.5, upper)
        data["Low"] = np.minimum(data["Low"] + spread_noise * 0.5, lower)

    data["VOL_MA"] = data["Volume"].rolling(vol_ma).mean()

    # Calculation of ATR
    tr1_calc = data["High"] - data["Low"]
    tr2_calc = (data["High"] - data["Close"].shift(1)).abs()
    tr3_calc = (data["Low"] - data["Close"].shift(1)).abs()
    tr_calc = pd.concat([tr1_calc, tr2_calc, tr3_calc], axis=1).max(axis=1)
    data["ATR"] = tr_calc.ewm(alpha=1/adx_period, adjust=False).mean()

    # Cruces clásicos
    buy_cross = (data["RSI_FAST"] > data["RSI_SLOW"]) & (data["RSI_FAST"].shift(1) <= data["RSI_SLOW"].shift(1))
    sell_cross = (data["RSI_FAST"] < data["RSI_SLOW"]) & (data["RSI_FAST"].shift(1) >= data["RSI_SLOW"].shift(1))

    # El filtro de tendencia ahora mira la relación entre las dos Medias del Precio (como en el RSI)
    trend_up = data["TREND_FAST_MA"].shift(1) > data["TREND_SLOW_MA"].shift(1)
    trend_down = data["TREND_FAST_MA"].shift(1) < data["TREND_SLOW_MA"].shift(1)
    
    if use_trend_filter:
        data["FINAL_BUY"] = buy_cross & trend_up
        data["FINAL_SELL"] = sell_cross & trend_down
        data["ZONE_BUY"] = trend_up & (data["RSI_FAST"] > data["RSI_SLOW"])
        data["ZONE_SELL"] = trend_down & (data["RSI_FAST"] < data["RSI_SLOW"])
    else:
        data["FINAL_BUY"] = buy_cross
        data["FINAL_SELL"] = sell_cross
        data["ZONE_BUY"] = (data["RSI_FAST"] > data["RSI_SLOW"])
        data["ZONE_SELL"] = (data["RSI_FAST"] < data["RSI_SLOW"])

    data["CandleColor"] = np.where(data["Close"] >= data["Open"], "#8CFF4D", "#FF79D1")
    return data.dropna().copy()


def make_chart(data: pd.DataFrame, interval: str, adx_threshold: int, show_labels: bool = True) -> go.Figure:
    data = data.copy()
    
    # Transformar a strings para forzar un eje categórico y eliminar el 100% de huecos vacíos
    if "d" in interval.lower() or "mo" in interval.lower() or "wk" in interval.lower():
        fmt = "%d-%b<br>%Y"
    else:
        fmt = "%H:%M<br>%d-%b"
    data.index = data.index.strftime(fmt)

        
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.50, 0.15, 0.15, 0.10, 0.10],
    )

    # 1. Price candles
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="#8CFF4D",
            increasing_fillcolor="#8CFF4D",
            decreasing_line_color="#FF79D1",
            decreasing_fillcolor="#FF79D1",
            name="Precio",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["TREND_FAST_MA"],
            mode="lines",
            line=dict(color="#FF4FC9", width=2),
            name="Trend Fast MA",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["TREND_SLOW_MA"],
            mode="lines",
            line=dict(color="#39FF14", width=3), # Verde fluorescente, un poco más gruesa
            name="Trend Slow MA",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["EMA200"],
            mode="lines",
            line=dict(color="#00FFFF", width=2, dash="dash"),
            name="EMA 200",
        ),
        row=1,
        col=1,
    )

    # Signal Labels (Triangles moved to RSI Subplot)
    buys = data[data["FINAL_BUY"]]
    sells = data[data["FINAL_SELL"]]

    if show_labels:
        for idx, row in buys.iterrows():
            fig.add_annotation(
                x=idx, y=row["RSI_FAST"],
                text="▲ COMPRA",
                showarrow=True, arrowhead=2, arrowcolor="#39FF14",
                arrowsize=1.5, ax=0, ay=20,
                bgcolor="#005500", bordercolor="#39FF14",
                font=dict(color="white", size=10, family="Arial Black"),
                borderpad=2, row=2, col=1
            )
        for idx, row in sells.iterrows():
            fig.add_annotation(
                x=idx, y=row["RSI_FAST"],
                text="▼ VENTA",
                showarrow=True, arrowhead=2, arrowcolor="#FF1493",
                arrowsize=1.5, ax=0, ay=-20,
                bgcolor="#550033", bordercolor="#FF1493",
                font=dict(color="white", size=10, family="Arial Black"),
                borderpad=2, row=2, col=1
            )

    # Paint zones
    def add_zones(condition, color):
        blocks = []
        start = None
        for i, (idx, val) in enumerate(condition.items()):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                blocks.append((start, data.index[i-1]))
                start = None
        if start is not None:
            blocks.append((start, data.index[-1]))
             
        for s, e in blocks:
            fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=0.15, layer="below", line_width=0, row=1, col=1)

    add_zones(data["ZONE_BUY"], "#39FF14")  # Green zone
    add_zones(data["ZONE_SELL"], "#FF1493") # Pink zone


    # 2. RSI panel sin relleno
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI_FAST"],
            mode="lines",
            line=dict(color="#00BFFF", width=1.5), # Azul celeste
            name="RSI Fast (Azul)",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI_SLOW"],
            mode="lines",
            line=dict(color="#FFD700", width=1.5), # Amarillo como en la imagen
            name="RSI Slow (Amarilla)",
        ),
        row=2,
        col=1,
    )

    # Marcaciones del RSI en la gráfica real de 0 a 100
    fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="#00BFFF", opacity=0.8, row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
    fig.add_hline(y=60, line_width=1.5, line_color="red", opacity=0.8, row=2, col=1)
    fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
    fig.add_hline(y=40, line_width=1.5, line_color="red", opacity=0.8, row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
    fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="#00BFFF", opacity=0.8, row=2, col=1)

    # 3. Volume Panel
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            marker_color=data["CandleColor"],
            name="Volume",
        ),
        row=3,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["VOL_MA"],
            mode="lines",
            line=dict(color="rgba(0, 0, 0, 0.7)", width=2),
            name="Vol MA",
        ),
        row=3,
        col=1,
    )

    # 4. ADX Panel
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["ADX"],
            mode="lines",
            line=dict(color="#00FFFF", width=2),
            name="ADX",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="#FF4B4B", opacity=0.8, row=4, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#FFA500", opacity=0.8, row=4, col=1)

    # 5. ATR Panel
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["ATR"],
            mode="lines",
            line=dict(color="#FF8C00", width=2),
            name="ATR",
        ),
        row=5,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=1250,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    # Configurar el eje principal (row=1) como categórico y calcular espaciado de etiquetas
    num_ticks = 8
    step = max(len(data) // num_ticks, 1)
    
    fig.update_xaxes(
        showticklabels=True,
        tickmode="linear",
        dtick=step,
        tickangle=0,
        type="category",
        row=1, col=1
    )
    # Hacer categóricos ocultando etiquetas para las filas inferiores
    fig.update_xaxes(showticklabels=False, type="category", row=2, col=1)
    fig.update_xaxes(showticklabels=False, type="category", row=3, col=1)
    fig.update_xaxes(showticklabels=False, type="category", row=4, col=1)
    fig.update_xaxes(showticklabels=False, type="category", row=5, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI osc", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_yaxes(title_text="ADX", row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)
    return fig


# -----------------------------
# UI
# -----------------------------
if st_autorefresh:
    st_autorefresh(interval=15 * 1000, key="dataframerefresh")
st.title("EA RSI Cross Triangles (Lógica MT5)")
st.caption("Versión 3: Filtro estricto de tendencia `UseTrendFilter` idéntico a EA_RSI_Cross_Triangles_MT5.")

with st.sidebar:
    st.header("Configuración")
    
    opciones = ["GC=F", "HG=F", "JPY=X", "AUDCHF=X", "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "YM=F", "US10Y", "Otro..."]
    sel = st.selectbox("Símbolo Yahoo Finance", opciones, index=0)
    if sel == "Otro...":
        symbol = st.text_input("Escribe el símbolo exacto de Yahoo", value="GC=F")
    else:
        symbol = sel
        
    period = st.selectbox("Periodo", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Intervalo", ["5m", "15m", "30m", "1h"], index=1)

    st.subheader("Parámetros del indicador")
    rsi_period = st.slider("Periodo RSI principal (Rápida / Azul)", 5, 50, 20)
    fast_ma = st.slider("Suavizado extra RSI (Dejar en 1)", 1, 100, 1)
    slow_ma = st.slider("Suavizado Lento RSI (EMA Amarilla)", 4, 100, 50)
    trend_fast_ma = st.slider("Media de tendencia precio (Rápida)", 1, 100, 3)
    trend_slow_ma = st.slider("Media de tendencia precio (Lenta)", 1, 200, 9)
    vol_ma = st.slider("Media de Volumen", 5, 100, 20)
    
    adx_period = st.slider("Periodo ADX", 5, 50, 14)
    adx_threshold = st.slider("Umbral Tendencia ADX", 10, 50, 20)
    
    use_trend_filter = st.checkbox("Usar Filtro de Tendencia (MT5)", value=True, help="Si se activa, exige que la Media Rápida esté por encima/debajo de la Lenta. Si no, dibuja todo.")
    show_labels = st.checkbox("Dibujar Señales (Triángulos)", value=True)
    is_audchf = (symbol == "AUDCHF=X")
    fix_wicks = st.checkbox("Filtrar mechas largas (Solución datos FX cruzados)", value=is_audchf)

    st.markdown("---")
    st.write("Parámetros clásicos del Asesor MT5:")
    st.code("RSI=14 | Fast=2 | Slow=20 | P_Fast=10 | P_Slow=35")

try:
    us10y_data = download_fx_data("^TNX", period, interval)
    if not us10y_data.empty:
        us10y_last = float(us10y_data['Close'].iloc[-1])
        us10y_prev = float(us10y_data['Close'].iloc[-2]) if len(us10y_data) > 1 else us10y_last
        st.metric("US 10Y (Bono Tesoro 10 años)", f"{us10y_last:.3f}%", f"{us10y_last - us10y_prev:+.3f}%")
        if "d" in interval.lower() or "mo" in interval.lower() or "wk" in interval.lower():
            fmt_us10y = "%d-%b<br>%Y"
        else:
            fmt_us10y = "%H:%M<br>%d-%b"
        us10y_slice = us10y_data.tail(250)
        us10y_index_str = us10y_slice.index.strftime(fmt_us10y)
        us10y_chart = go.Figure(go.Scatter(x=us10y_index_str, y=us10y_slice["Close"], mode='lines', line=dict(color='#FF4500', width=2)))
        us10y_chart.update_layout(template="plotly_dark", height=220, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
        num_ticks_us10y = 8
        step_us10y = max(len(us10y_slice) // num_ticks_us10y, 1)
        us10y_chart.update_xaxes(type="category", showticklabels=True, tickmode="linear", dtick=step_us10y, tickangle=0)
        st.plotly_chart(us10y_chart, use_container_width=True)
except Exception:
    pass

try:
    dxy_data = download_fx_data("DX-Y.NYB", period, interval)
    if not dxy_data.empty:
        dxy_last = float(dxy_data['Close'].iloc[-1])
        st.metric("DXY (US Dollar Index)", f"{dxy_last:.2f}")
        if "d" in interval.lower() or "mo" in interval.lower() or "wk" in interval.lower():
            fmt = "%d-%b<br>%Y"
        else:
            fmt = "%H:%M<br>%d-%b"
            
        dxy_slice = dxy_data.tail(250)
        dxy_index_str = dxy_slice.index.strftime(fmt)
        
        dxy_chart = go.Figure(go.Scatter(x=dxy_index_str, y=dxy_slice["Close"], mode='lines', line=dict(color='#00BFFF', width=2)))
        dxy_chart.update_layout(template="plotly_dark", height=220, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
        
        num_ticks = 8
        step = max(len(dxy_slice) // num_ticks, 1)
        dxy_chart.update_xaxes(type="category", showticklabels=True, tickmode="linear", dtick=step, tickangle=0)
        
        st.plotly_chart(dxy_chart, use_container_width=True)
except Exception:
    pass

try:
    us30_data = download_fx_data("YM=F", period, interval)
    if not us30_data.empty:
        us30_last = float(us30_data['Close'].iloc[-1])
        us30_prev = float(us30_data['Close'].iloc[-2]) if len(us30_data) > 1 else us30_last
        st.metric("US 30 (Dow Jones Futures)", f"{us30_last:,.0f}", f"{us30_last - us30_prev:+,.0f}")
        if "d" in interval.lower() or "mo" in interval.lower() or "wk" in interval.lower():
            fmt_us30 = "%d-%b<br>%Y"
        else:
            fmt_us30 = "%H:%M<br>%d-%b"
        us30_slice = us30_data.tail(250)
        us30_index_str = us30_slice.index.strftime(fmt_us30)
        us30_chart = go.Figure(go.Scatter(x=us30_index_str, y=us30_slice["Close"], mode='lines', line=dict(color='#FFD700', width=2)))
        us30_chart.update_layout(template="plotly_dark", height=220, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
        num_ticks_us30 = 8
        step_us30 = max(len(us30_slice) // num_ticks_us30, 1)
        us30_chart.update_xaxes(type="category", showticklabels=True, tickmode="linear", dtick=step_us30, tickangle=0)
        st.plotly_chart(us30_chart, use_container_width=True)
except Exception:
    pass

try:
    raw = download_fx_data(symbol, period, interval)
    if raw.empty:
        st.error("No se pudieron descargar datos. Prueba otro periodo o intervalo.")
        st.stop()

    data = build_indicator(raw, rsi_period, fast_ma, slow_ma, trend_fast_ma, trend_slow_ma, vol_ma, adx_period, use_trend_filter, fix_wicks)
    if data.empty:
        st.error("No hay suficientes datos tras calcular el indicador.")
        st.stop()

    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last

    if last["FINAL_BUY"]:
        signal_txt = "BUY ▲"
    elif last["FINAL_SELL"]:
        signal_txt = "SELL ▼"
    elif last["RSI_FAST"] > last["RSI_SLOW"]:
        signal_txt = "Mom. Alcista"
    else:
        signal_txt = "Mom. Bajista"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precio actual", f"{last['Close']:.3f}", f"{last['Close'] - prev['Close']:.3f}")
    c2.metric("RSI", f"{last['RSI']:.2f}")
    c3.metric("Volumen", f"{last['Volume']:,.0f}", f"{last['Volume'] - prev['Volume']:,.0f}")
    
    adx_state = "TENDENCIA" if last['ADX'] >= adx_threshold else "BALANCE"
    c4.metric(f"ADX ({last['ADX']:.1f})", adx_state)
    c5.metric("Estado MT5", signal_txt)

    with st.sidebar:
        st.markdown("---")
        st.subheader("Estado de Mercado (ADX)")
        if last['ADX'] >= adx_threshold:
            st.success(f"**TENDENCIA** (ADX = {last['ADX']:.1f})")
        else:
            st.warning(f"**BALANCE** (ADX = {last['ADX']:.1f})")

    fig = make_chart(data.tail(250), interval, adx_threshold=adx_threshold, show_labels=show_labels)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Últimas señales de Entrada / Salida"):
        signals = data.loc[data["FINAL_BUY"] | data["FINAL_SELL"]].copy()
        signals["Signal"] = np.where(signals["FINAL_BUY"], "BUY ▲", "SELL ▼")
        display_cols = ["Signal", "Close", "RSI"]
        signals = signals[display_cols].tail(20).sort_index(ascending=False)
        st.dataframe(signals, use_container_width=True)

    with st.expander("Lógica interna (Robot MT5)"):
        st.markdown(
            """
            **Señales (Triángulos exactos al Robot EA MT5):**
            - **BUY ▲**: El RSI Rápido (3) cruza por encima del RSI Lento (10). Si `UseTrendFilter` está activado, la vela anterior TIENE que haber cerrado por encima de la MA de Tendencia.
            - **SELL ▼**: El RSI Rápido (3) cruza por debajo del RSI Lento (10). Si `UseTrendFilter` está activado, la vela anterior TIENE que haber cerrado por debajo de la MA de Tendencia.
            - **Filtro de Volumen:** Pasa a ser un panel meramente visual. El robot MT5 solo tradea por dirección de tendencia.
            - **Triángulos de la gráfica:** Coinciden exactamente con los comandos ObjectCreate del experto.
            """
        )

except Exception as e:
    st.error(f"Error al ejecutar la app: {e}")
    st.info("Instala dependencias: streamlit, yfinance, pandas, numpy, plotly")
