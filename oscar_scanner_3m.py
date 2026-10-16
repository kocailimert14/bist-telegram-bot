import os
import sys
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# TELEGRAM AYARLARI
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:

# ==========================================
# QUANTFURY KRİPTO LİSTESİ (Binance USDT Pariteleri)
# ==========================================
QUANTFURY_COINS = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BCHUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT",
    "ETCUSDT", "ETHUSDT", "FILUSDT", "GRAMUSDT", "HBARUSDT",
    "HYPEUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT",
    "ONDOUSDT", "OPUSDT", "POLUSDT", "RENDERUSDT", "RUNEUSDT",
    "SUSDT", "SANDUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT",
    "THETAUSDT", "UNIUSDT", "VIRTUALUSDT", "XLMUSDT", "XRPUSDT",
    "ZECUSDT"
]

# ==========================================
# İNDİKATÖR PARAMETRELERİ (Görsellerdeki Ayarlar)
# ==========================================
OSCAR_LEN = 8                  # Oscar Candles
OSCAR_SMOOTHING = "SMA"        # Oscar Smoothing: SMA
SLOW_OSCAR_LEN = 16            # Slow Oscar Candles
SLOW_OSCAR_SMOOTHING = "WMA"   # Slow Oscar Smoothing: WMA
CROSS_SENSITIVITY = 1.5        # Cross Signal Sensitivity
BOTTOM_SIGNAL_LINE = 35.0      # Bottom Signal Line
TOP_SIGNAL_LINE = 65.0         # Top Signal Line

# Tekrarlayan sinyalleri engellemek için hafıza
SENT_SIGNALS = set()

def send_telegram(message: str) -> bool:
    """Telegram kanalına HTML formatında mesaj gönderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Mesajı (Simülasyon)]:\n", message)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=12)
        if res.status_code == 200:
            return True
        elif res.status_code == 429:
            retry_after = res.json().get("parameters", {}).get("retry_after", 30)
            return False
        else:
            return False
    except Exception as e:
        return False

def get_binance_klines(symbol: str, interval: str = "3m", limit: int = 100) -> pd.DataFrame:
    """Binance Public API üzerinden mum verilerini çeker."""
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 20:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                # Türkiye saati (UTC+3)
                return df
    except Exception:
        pass

    # Bybit Spot Fallback
    try:
        bybit_int = "3" if interval == "3m" else "5"
        url_bybit = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bybit_int}&limit={limit}"
        res = requests.get(url_bybit, timeout=8)
        if res.status_code == 200:
            raw_list = res.json().get('result', {}).get('list', [])
            if raw_list and len(raw_list) >= 20:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()

def detect_candlestick_patterns(df: pd.DataFrame, idx: int = -1) -> list:
    patterns = []
    if df is None or len(df) < abs(idx) + 2:
        return patterns

    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]
    prev2 = df.iloc[idx - 2]

    c_open, c_high, c_low, c_close = curr['Open'], curr['High'], curr['Low'], curr['Close']
    p_open, p_high, p_low, p_close = prev['Open'], prev['High'], prev['Low'], prev['Close']
    p2_open, p2_high, p2_low, p2_close = prev2['Open'], prev2['High'], prev2['Low'], prev2['Close']

    c_body = abs(c_close - c_open)
    c_range = c_high - c_low
    p_body = abs(p_close - p_open)
    p_range = p_high - p_low

    if c_range == 0 or p_range == 0:
        return patterns

    c_upper_wick = c_high - max(c_open, c_close)
    c_lower_wick = min(c_open, c_close) - c_low

    # 1. Bullish Engulfing (Boğa Yutan)
    if p_close < p_open and c_close > c_open and c_close >= p_open and c_open <= p_close:

    # 2. Bearish Engulfing (Ayı Yutan)
    if p_close > p_open and c_close < c_open and c_close <= p_open and c_open >= p_close:

    # 3. Hammer (Çekiç)
    if c_lower_wick >= 2 * c_body and c_upper_wick <= c_body * 0.5 and c_body / c_range > 0.1:

    # 4. Shooting Star (Ters Çekiç / Kayan Yıldız)
    if c_upper_wick >= 2 * c_body and c_lower_wick <= c_body * 0.5 and c_body / c_range > 0.1:

    # 5. Doji
    if c_body / c_range <= 0.1:

    # 6. Morning Star (Sabah Yıldızı)
    if (p2_close < p2_open and
        abs(p_close - p_open) / (p_high - p_low + 1e-10) < 0.3 and
        c_close > c_open and c_close > (p2_open + p2_close) / 2):

    # 7. Evening Star (Akşam Yıldızı)
    if (p2_close > p2_open and
        abs(p_close - p_open) / (p_high - p_low + 1e-10) < 0.3 and
        c_close < c_open and c_close < (p2_open + p2_close) / 2):

    return patterns

def calculate_oscar(df: pd.DataFrame):
    A = highest(close, len)
    B = lowest(close, len)
    OscarRough = (close - B) / (A - B) * 100
    Oscar1 = (OscarRough[1] / 3) * 2
    OscarThird = OscarRough / 3
    Oscar = Oscar1 + OscarThird
    smoothedOscarRough = sma(OscarRough, len)
    smoothedOscar = sma(Oscar, len)
    """
    if df is None or len(df) < 25:
        return None

    close = df['Close']
    a = close.rolling(window=OSCAR_LEN).max()
    b = close.rolling(window=OSCAR_LEN).min()

    denom = (a - b).replace(0, 1e-10)
    oscar_rough = (close - b) / denom * 100.0

    oscar_rough_prev = oscar_rough.shift(1)
    oscar = (oscar_rough_prev / 3.0) * 2.0 + (oscar_rough / 3.0)

    # Oscar Smoothing: SMA (Girdiler tabındaki ayar)
    smoothed_oscar = oscar.rolling(window=OSCAR_LEN).mean()

    cross_sens = (smoothed_rough - smoothed_oscar).abs()

    crossover = (smoothed_rough > smoothed_oscar) & (smoothed_rough.shift(1) <= smoothed_oscar.shift(1))
    crossunder = (smoothed_rough < smoothed_oscar) & (smoothed_rough.shift(1) >= smoothed_oscar.shift(1))

    # 4 Ayrı Sinyal Şartı
    sig_black_buy = crossover & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough < BOTTOM_SIGNAL_LINE)

    # 2. Yeşil Ok (Alım Teyidi / Devam)

    sig_black_sell = crossunder & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > TOP_SIGNAL_LINE)

    # 4. Kırmızı Ters Ok (Satım Teyidi / Devam)
    sig_red_sell = crossunder.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_oscar > smoothed_rough) & (smoothed_oscar > TOP_SIGNAL_LINE)
    return {
        'smoothed_rough': smoothed_rough,
        'smoothed_oscar': smoothed_oscar,
        'cross_sens': cross_sens,
        'black_buy': sig_black_buy,
        'green_buy': sig_green_buy,
        'black_sell': sig_black_sell,
        'red_sell': sig_red_sell
    }

def scan_single_coin(symbol: str):
    try:
        df = get_binance_klines(symbol, interval="3m", limit=80)
        if df.empty or len(df) < 25:
            return None

        res = calculate_oscar(df)
        if res is None:
            return None

        # Son tamamlanan muma bakılır (veya son anlık mum: index -1)
        target_idx = -1
        candle_time = df.index[target_idx]
        candle_price = float(df['Close'].iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')

        b_buy = bool(res['black_buy'].iloc[target_idx])
        g_buy = bool(res['green_buy'].iloc[target_idx])
        b_sell = bool(res['black_sell'].iloc[target_idx])
        r_sell = bool(res['red_sell'].iloc[target_idx])

        detected_signal = None
        signal_title = ""
        signal_icon = ""

        if b_buy:
            detected_signal = "BLACK_BUY"
        elif g_buy:
            detected_signal = "GREEN_BUY"
        elif b_sell:
            detected_signal = "BLACK_SELL"
        elif r_sell:
            detected_signal = "RED_SELL"

        if not detected_signal:
            return None

        # Aynı mum için tekrar sinyal atmayı engelle
        sig_key = (symbol, str(candle_time), detected_signal)
        if sig_key in SENT_SIGNALS:
            return None
        SENT_SIGNALS.add(sig_key)

        coin_name = symbol.replace("USDT", "")

        rough_val = float(res['smoothed_rough'].iloc[target_idx])
        oscar_val = float(res['smoothed_oscar'].iloc[target_idx])
        sens_val = float(res['cross_sens'].iloc[target_idx])

        candle_patterns = detect_candlestick_patterns(df, idx=target_idx)



        message = (

            f"🎯 <b>Sinyal:</b> {signal_title}\n"


            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⏱ <b>Periyot:</b> 3 Dakika (3m)\n"

            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n\n"

            f"{pattern_str}"


        )
        return message

    except Exception as e:
        return None

def run_scanner():
    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    found_signals = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in QUANTFURY_COINS}
        for future in as_completed(future_to_coin):
            msg = future.result()
            if msg:
                send_telegram(msg)
                found_signals += 1

    elapsed = time.time() - start_time


if __name__ == "__main__":
    args = parser.parse_args()

    if args.loop or args.duration > 0:
        start_loop = time.time()
        while True:
            run_scanner()
            if args.duration > 0 and (time.time() - start_loop) >= args.duration * 60:
                break
            time.sleep(180)  # 3 dakika bekle
    else:
        run_scanner()
