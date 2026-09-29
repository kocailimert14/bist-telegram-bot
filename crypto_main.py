import os
import sys
import time
import numpy as np
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("HATA: Telegram Token veya Chat ID bulunamadı!")
    sys.exit(1)

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BCHUSDT", "LTCUSDT", 
    "NEARUSDT", "APTUSDT", "DOTUSDT", "TAOUSDT", "AAVEUSDT", 
    "RENDERUSDT", "INJUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", 
    "HBARUSDT", "OPUSDT", "ARBUSDT", "UNIUSDT", "RUNEUSDT", 
    "ONDOUSDT", "POLUSDT", "SNXUSDT", "THETAUSDT", "MANAUSDT", 
    "SANDUSDT", "ZECUSDT", "HYPEUSDT", "GRAMUSDT", "SUSDT"
]

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            print("Telegram bildirimi iletildi.")
            return True
        elif res.status_code == 429:
            retry_after = res.json().get("parameters", {}).get("retry_after", 30)
            print(f"Telegram Flood Uyarısı: {retry_after} saniye beklenmeli!")
            return False
        else:
            print(f"Telegram hatası ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")
        return False

def get_binance_klines(symbol: str, interval: str) -> pd.DataFrame:
    url_binance = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit=200"
    try:
        res = requests.get(url_binance, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 5:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bb_int = interval_map.get(interval, "15")
    url_bybit = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bb_int}&limit=200"
    try:
        res = requests.get(url_bybit, timeout=8)
        if res.status_code == 200:
            raw_list = res.json().get('result', {}).get('list', [])
            if raw_list and len(raw_list) >= 5:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['High', 'Low', 'Close'])

def wwma(series: pd.Series, length: int) -> pd.Series:
    vals = series.fillna(0.0).values
    res = np.zeros(len(vals))
    for i in range(len(vals)):
        prev = res[i-1] if i > 0 else 0.0
        res[i] = (prev * (length - 1) + vals[i]) / length
    return pd.Series(res, index=series.index)

def calculate_slingshot(df: pd.DataFrame, idx: int = -1):
res, index=series.index)

def calculate_slingshot(df: pd.DataFrame, idx: int = -1):
    if df is None or len(df) < 15:
        return "Belirsiz", "Belirsiz"
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        ma1 = close.ewm(span=13, adjust=False).mean()
        ma2 = close.ewm(span=21, adjust=False).mean()
        ma3 = close.ewm(span=34, adjust=False).mean()
        ma = close.ewm(span=89, adjust=False).mean()
        rangema = tr.ewm(span=89, adjust=False).mean()
        upper = ma + rangema * 0.5
        lower = ma - rangema * 0.5

        v_ma1 = float(ma1.iloc[idx])
        v_ma2 = float(ma2.iloc[idx])
        v_ma3 = float(ma3.iloc[idx])
        v_upper = float(upper.iloc[idx])
        v_lower = float(lower.iloc[idx])

        if v_ma1 > v_upper and v_ma2 > v_upper and v_ma3 > v_upper:
            kanal_renk = "🟢 Yeşil"
        elif v_ma1 < v_lower and v_ma2 < v_lower and v_ma3 < v_lower:
            kanal_renk = "🔴 Kırmızı"
        else:
            kanal_renk = "🔵 Mavi"

        if v_ma1 > v_ma2 and v_ma2 > v_ma3:
            nokta_renk = "🟢 Yeşil"
        elif v_ma2 < v_ma3:
            nokta_renk = "🔴 Kırmızı"
        else:
            nokta_renk = "🟡 Sarı"

        return kanal_renk, nokta_renk
    except Exception:
        return "Belirsiz", "Belirsiz"

def calculate_woodie_pivots(df_daily: pd.DataFrame) -> dict:
    """TradingView Pivot Points Standard (Auto Anchor: 1D) Woodie hesaplaması."""
    try:
        df = clean_df(df_daily)
        if df.empty or len(df) < 2:
            return {}

        prev_high = float(df['High'].iloc[-2])
        prev_low = float(df['Low'].iloc[-2])
        curr_open = float(df['Open'].iloc[-1])

        p = (prev_high + prev_low + 2.0 * curr_open) / 4.0
        r1 = 2.0 * p - prev_low
        s1 = 2.0 * p - prev_high
        r2 = p + (prev_high - prev_low)
        s2 = p - (prev_high - prev_low)
        r3 = prev_high + 2.0 * (p - prev_low)
        s3 = prev_low - 2.0 * (prev_high - p)

        return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}
    except Exception:
        return {}

def detect_diagonal_trendline_and_initiation(df: pd.DataFrame, lookback: int = 45):
    if df is None or len(df) < 20:
        return "Standart Hareket", "Yeni Trend Yok"
    try:
        window = min(lookback, len(df))
        sub_h = df['High'].values[-window:]
        sub_l = df['Low'].values[-window:]
        sub_c = df['Close'].values[-window:]
        curr_c = sub_c[-1]
        curr_h = sub_h[-1]

        peak_indices = [
            i for i in range(2, window - 2)
            if all(sub_h[i] >= sub_h[i-j] for j in range(1, 3)) and all(sub_h[i] >= sub_h[i+j] for j in range(1, 3))
        ]

        dusen_kirilim_durumu = "Düşen Trend Tespit Edilmedi"
        trend_baslatma_durumu = "Yeni Trend Başlamadı"

        if len(peak_indices) >= 2:
            sorted_peaks = sorted(peak_indices, key=lambda idx: sub_h[idx], reverse=True)
            p1 = sorted_peaks[0]
            candidates = [p for p in peak_indices if p > p1 and sub_h[p] < sub_h[p1]]
            if candidates:
                p2 = candidates[0]
                slope = (sub_h[p2] - sub_h[p1]) / (p2 - p1)
                intercept = sub_h[p1] - slope * p1
                line_val_now = slope * (window - 1) + intercept
                line_val_prev = slope * (window - 2) + intercept

                if curr_c > line_val_now:
                    if sub_c[-2] <= line_val_prev or (curr_c - line_val_now) / line_val_now <= 0.04:
                        dusen_kirilim_durumu = f"✅ Düşen Trend Çizgisi Yukarı Kırıldı! (Trend: ${line_val_now:,.2f}) (Boğa)"
                    else:
                        dusen_kirilim_durumu = f"🚀 Düşen Trend Üzerinde Seyrediyor (Trend: ${line_val_now:,.2f}) (Boğa)"
                elif curr_h >= line_val_now and curr_c <= line_val_now:
                    dusen_kirilim_durumu = f"⚠️ Düşen Trend Çizgisi Test Ediliyor (Direnç: ${line_val_now:,.2f}) (Nötr)"
                else:
                    dusen_kirilim_durumu = f"Düşen Trend Altında (Direnç: ${line_val_now:,.2f})"

        lowest_idx = np.argmin(sub_l[:-2])
        lowest_val = sub_l[lowest_idx]

        if lowest_idx < window - 4:
            after_troughs = [
                i for i in range(lowest_idx + 2, window - 1)
                if sub_l[i] <= sub_l[i-1] and sub_l[i] <= sub_l[i+1]
            ]
            if after_troughs:
                sec_idx = after_troughs[-1]
                sec_val = sub_l[sec_idx]
                if sec_val > lowest_val:
                    up_slope = (sec_val - lowest_val) / (sec_idx - lowest_idx)
                    up_line = up_slope * (window - 1) + (lowest_val - up_slope * lowest_idx)
                    if curr_c >= up_line:
                        trend_baslatma_durumu = f"📈 Yeni Yükselen Trend Başlattı! (Dipten Destek: ${up_line:,.2f}, Yükselen Dip Onaylı) (Boğa)"
                    else:
                        trend_baslatma_durumu = "Yükselen Trend Desteği Altında (Nötr)"
                else:
                    trend_baslatma_durumu = "Düşük Dipler Devam Ediyor"
            elif curr_c > lowest_val * 1.03:
                trend_baslatma_durumu = "⚡ Dipten Hızlı Toparlanma (V Dönüş Başlangıcı) (Boğa)"

        return dusen_kirilim_durumu, trend_baslatma_durumu
    except Exception:
        return "Standart Hareket", "Yeni Trend Yok"

def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    if df is None or len(df) < 5:
        return "Standart Mum"
    try:
        sub = df.iloc[-5:]
        o, h, l, c = sub['Open'].values, sub['High'].values, sub['Low'].values, sub['Close'].values
        body = np.abs(c - o)
        cr = np.where((h - l) == 0, 1e-10, h - l)
        is_bull, is_bear = c > o, c < o
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        # 1. ÇOKLU MUM FORMASYONLARI (4-5 MUM)
        if is_bear[1] and is_bear[2] and is_bear[3] and is_bull[4] and c[3] < c[2] < c[1] and o[4] <= c[3] and c[4] >= o[1]:
            return "⚔️ Bullish Three-Line Strike (Yükseliş Dönüş /
    
