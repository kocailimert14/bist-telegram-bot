# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# TELEGRAM AYARLARI
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("UYARI: TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID ortam degiskeni tanimli degil!")

# ==========================================
# QUANTFURY KRIPTO LISTESI (36 Koin)
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
# INDIKATOR PARAMETRELERI (Koncorde 2.0 + OSCAR v5)
# ==========================================
# OSCAR Parametreleri
OSCAR_LEN = 8                  # len = 8
SLOW_OSCAR_LEN = 16            # slowLen = 16
OSCAR_SMOOTHING = "RMA"        # smoothing = "RMA"
SLOW_OSCAR_SMOOTHING = "WMA"   # slowOscarSmoothing = "WMA"
CROSS_SENSITIVITY = 0.5        # crossSignalSensitivity = 0.5
BOTTOM_SIGNAL_LINE = 35.0      # bottomSignalLine = 35
TOP_SIGNAL_LINE = 65.0         # topSignalLine = 65

# KONCORDE 2.0 Parametreleri
KONCORDE_M = 15                # m = 15 (EMA periyodu)
KONCORDE_LOOKBACK = 90         # pvim/nvim lookback (90 mum)

# Tekrarlayan sinyalleri engellemek icin hafiza
SENT_SIGNALS = set()


def send_telegram(message: str) -> bool:
    """Telegram kanalina HTML formatinda mesaj gonderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Mesaji (Simulasyon)]:\n", message)
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
            print(f"Telegram Flood Uyarisi: {retry_after} sn beklenmeli!")
            return False
        else:
            print(f"Telegram hatasi ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Telegram baglanti hatasi: {e}")
        return False


def get_binance_klines(symbol: str, interval: str = "3m", limit: int = 150) -> pd.DataFrame:
    """Binance Public API uzerinden mum verilerini ceker (Yedek Bybit)."""
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 35:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                # Turkiye saati (UTC+3)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
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
            if raw_list and len(raw_list) >= 35:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()


def pine_ma(smoothing: str, series: pd.Series, length: int) -> pd.Series:
    """Pine Script ma() fonksiyonu: RMA, SMA, EMA, WMA, NONE."""
    s_type = smoothing.upper()
    if s_type == "SMA":
        return series.rolling(window=length).mean()
    elif s_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif s_type == "WMA":
        weights = np.arange(1, length + 1, dtype=float)
        weight_sum = weights.sum()
        return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / weight_sum, raw=True)
    elif s_type == "RMA":
        vals = series.values.astype(float)
        out = np.full(len(vals), np.nan)
        alpha = 1.0 / length
        valid_indices = np.where(~np.isnan(vals))[0]
        if len(valid_indices) >= length:
            start_idx = valid_indices[length - 1]
            out[start_idx] = np.mean(vals[valid_indices[:length]])
            for i in range(start_idx + 1, len(vals)):
                if np.isnan(vals[i]):
                    out[i] = out[i - 1]
                else:
                    out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
        return pd.Series(out, index=series.index)
    else:
        return series


def calculate_koncorde(df: pd.DataFrame):
    """
    Pine Script Koncorde 2.0 hesaplamasi:
    - azul: Kurumsal / Balina Para Giris-Cikisi (NVI bazli)
    - verde: Bireysel / Kucuk Yatirimci (PVI + Marron bazli)
    - marron: RSI + MFI + Bollinger + Stokastik sentezi
    - media: EMA(marron, 15) sinyal cizgisi
    """
    if df is None or len(df) < 35:
        return None

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_ = df['Open'].values
    volume = df['Volume'].values
    n = len(close)

    # 1. PVI & NVI Hesaplamasi
    pvi = np.zeros(n)
    nvi = np.zeros(n)
    pvi[0] = 1000.0
    nvi[0] = 1000.0

    for i in range(1, n):
        prev_c = close[i - 1]
        c_ret = (close[i] - prev_c) / prev_c if prev_c > 0 else 0.0

        if volume[i] > volume[i - 1]:
            pvi[i] = pvi[i - 1] + c_ret * pvi[i - 1]
        else:
            pvi[i] = pvi[i - 1]

        if volume[i] < volume[i - 1]:
            nvi[i] = nvi[i - 1] + c_ret * nvi[i - 1]
        else:
            nvi[i] = nvi[i - 1]

    pvi_s = pd.Series(pvi, index=df.index)
    nvi_s = pd.Series(nvi, index=df.index)

    m = KONCORDE_M
    pvim = pvi_s.ewm(span=m, adjust=False).mean()
    pvimax = pvim.rolling(window=KONCORDE_LOOKBACK, min_periods=1).max()
    pvimin = pvim.rolling(window=KONCORDE_LOOKBACK, min_periods=1).min()
    pvi_diff = pvimax - pvimin
    oscp = np.where(pvi_diff != 0, (pvi_s - pvim) * 100.0 / pvi_diff, 0.0)

    nvim = nvi_s.ewm(span=m, adjust=False).mean()
    nvimax = nvim.rolling(window=KONCORDE_LOOKBACK, min_periods=1).max()
    nvimin = nvim.rolling(window=KONCORDE_LOOKBACK, min_periods=1).min()
    nvi_diff = nvimax - nvimin
    azul = np.where(nvi_diff != 0, (nvi_s - nvim) * 100.0 / nvi_diff, 0.0)

    # 2. Money Flow Index (MFI 14)
    hlc3 = (high + low + close) / 3.0
    hlc3_s = pd.Series(hlc3, index=df.index)
    chg = hlc3_s.diff()
    up_mfi = np.where(chg > 0, volume * hlc3, 0.0)
    dn_mfi = np.where(chg < 0, volume * hlc3, 0.0)
    up_sum = pd.Series(up_mfi, index=df.index).rolling(window=14, min_periods=1).sum()
    dn_sum = pd.Series(dn_mfi, index=df.index).rolling(window=14, min_periods=1).sum()
    sum_mfi = up_sum + dn_sum
    xmf = np.where(sum_mfi != 0, 100.0 * up_sum / sum_mfi, 50.0)

    # 3. Bollinger Oscillator
    tprice = (open_ + high + low + close) / 4.0
    tp_s = pd.Series(tprice, index=df.index)
    basis = tp_s.rolling(window=25, min_periods=1).mean()
    dev = 2.0 * tp_s.rolling(window=25, min_periods=1).std(ddof=0)
    upperBB = basis + dev
    lowerBB = basis - dev
    OB1 = (upperBB + lowerBB) / 2.0
    OB2 = upperBB - lowerBB
    BollOsc = np.where(OB2 != 0, ((tp_s - OB1) / OB2) * 100.0, 0.0)

    # 4. RSI (14) on tprice
    delta = tp_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    xrsi = 100.0 - (100.0 / (1.0 + rs))

    # 5. Stochastic (21, 3)
    low_s = pd.Series(low, index=df.index)
    high_s = pd.Series(high, index=df.index)
    ll_stoc = low_s.rolling(window=21, min_periods=1).min()
    hh_stoc = high_s.rolling(window=21, min_periods=1).max()
    stoc_diff = hh_stoc - ll_stoc
    k_stoc = np.where(stoc_diff != 0, 100.0 * (tp_s - ll_stoc) / stoc_diff, 0.0)
    stoc = pd.Series(k_stoc, index=df.index).rolling(window=3, min_periods=1).mean()

    # 6. Marron, Verde ve Media
    marron = (xrsi + xmf + BollOsc + (stoc / 3.0)) / 2.0
    verde = marron + oscp
    media = pd.Series(marron, index=df.index).ewm(span=m, adjust=False).mean()

    return {
        'azul': pd.Series(azul, index=df.index),
        'verde': pd.Series(verde, index=df.index),
        'marron': pd.Series(marron, index=df.index),
        'media': media
    }


def calculate_oscar(df: pd.DataFrame):
    """OSCAR v5 Pine Script hesaplamasi ve 4 sinyal sartlari."""
    if df is None or len(df) < 35:
        return None

    close = df['Close']
    a = close.rolling(window=OSCAR_LEN).max()
    b = close.rolling(window=OSCAR_LEN).min()
    denom = (a - b).replace(0, 1e-10)
    oscar_rough = (close - b) / denom * 100.0

    oscar_1 = (oscar_rough.shift(1).fillna(50.0) / 3.0) * 2.0
    oscar_third = oscar_rough / 3.0
    oscar = oscar_1 + oscar_third

    smoothed_rough = pine_ma(OSCAR_SMOOTHING, oscar_rough, OSCAR_LEN)
    smoothed_oscar = pine_ma(OSCAR_SMOOTHING, oscar, OSCAR_LEN)

    cross_sens = np.maximum(smoothed_rough, smoothed_oscar) - np.minimum(smoothed_rough, smoothed_oscar)

    crossover = (smoothed_rough > smoothed_oscar) & (smoothed_rough.shift(1) <= smoothed_oscar.shift(1))
    crossunder = (smoothed_rough < smoothed_oscar) & (smoothed_rough.shift(1) >= smoothed_oscar.shift(1))

    # 4 Sinyal Kosulu
    sig_buy_tri = crossover & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough < BOTTOM_SIGNAL_LINE)
    sig_buy_arr = crossover.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > smoothed_oscar) & (smoothed_oscar < BOTTOM_SIGNAL_LINE)
    sig_sell_tri = crossunder & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > TOP_SIGNAL_LINE)
    sig_sell_arr = crossunder.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_oscar > smoothed_rough) & (smoothed_oscar > TOP_SIGNAL_LINE)

    return {
        'smoothed_rough': smoothed_rough,
        'smoothed_oscar': smoothed_oscar,
        'cross_sens': cross_sens,
        'buy_tri': sig_buy_tri,
        'buy_arr': sig_buy_arr,
        'sell_tri': sig_sell_tri,
        'sell_arr': sig_sell_arr
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    """En yuksek istatistiksel basari oranina sahip Boga ve Ayi mum formasyonlari."""
    if df is None or len(df) < 5:
        return "Standart Mum"
    try:
        sub = df.iloc[-5:]
        o = sub['Open'].values
        h = sub['High'].values
        l = sub['Low'].values
        c = sub['Close'].values
        body = np.abs(c - o)
        candle_range = h - l
        is_bull = c > o
        is_bear = c < o
        eps = 1e-10
        cr = np.where(candle_range == 0, eps, candle_range)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        # BOGA FORMASYONLARI
        if (is_bear[1] and is_bear[2] and is_bear[3] and is_bull[4] and 
            c[3] < c[2] < c[1] and o[4] <= c[3] and c[4] >= o[1]):
            return "\u2694\ufe0f Bullish Three-Line Strike (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%84 Ba\u015far\u0131)"

        if (is_bull[2] and is_bull[3] and is_bull[4] and 
            c[4] > c[3] > c[2] and o[3] > o[2] and o[4] > o[3] and
            upper_wick[2]/cr[2] < 0.25 and upper_wick[3]/cr[3] < 0.25 and upper_wick[4]/cr[4] < 0.25):
            return "\U0001f6e1\ufe0f Three White Soldiers - \u00dc\u00e7 Beyaz Asker (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%82 Ba\u015far\u0131)"

        if (is_bull[0] and body[0]/cr[0] > 0.4 and is_bull[4] and c[4] > h[0] and min(l[1], l[2], l[3]) >= l[0]):
            return "\U0001f4c8 Rising Three Methods - Y\u00fckselen \u00dc\u00e7 Y\u00f6ntem (Y\u00fckseli\u015f Devam / Bo\u011fa) (%78 Ba\u015far\u0131)"

        if (is_bear[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bull[4] and c[4] > (o[2] + c[2])/2):
            return "\u2b50 Morning Doji Star - Sabah Y\u0131ld\u0131z\u0131 (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%76 Ba\u015far\u0131)"

        if (is_bear[2] and body[3]/cr[3] < 0.15 and is_bull[4] and h[3] < l[2] and l[4] > h[3] and c[4] > (o[2] + c[2])/2):
            return "\U0001f476 Bullish Abandoned Baby - Terk Edilmi\u015f Bebek (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%75 Ba\u015far\u0131)"

        if (is_bear[2] and is_bull[3] and o[3] <= c[2] and c[3] >= o[2] and is_bull[4] and c[4] > c[3]):
            return "\U0001f7e2 Three Outside Up (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%72 Ba\u015far\u0131)"

        if (is_bear[3] and is_bull[4] and (body[3]/cr[3] > 0.35) and (body[4]/cr[4] > 0.35) and 
            abs(o[4] - o[3]) / o[3] <= 0.003 and c[4] > h[3]):
            return "\u26a1 Bullish Separating Lines - Ayr\u0131lan \u00c7izgiler (Y\u00fckseli\u015f Devam / Bo\u011fa) (%68 Ba\u015far\u0131)"

        if (is_bear[3] and is_bull[4] and c[4] >= o[3] and o[4] <= c[3]):
            return "\U0001f7e2 Bullish Engulfing - Yutan Bo\u011fa (D\u00f6n\u00fc\u015f / Bo\u011fa) (%63 Ba\u015far\u0131)"

        if (lower_wick[4] >= 2 * body[4] and upper_wick[4] <= 0.15 * cr[4]):
            return "\U0001f528 Hammer - \u00c7eki\u00e7 (Y\u00fckseli\u015f D\u00f6n\u00fc\u015f / Bo\u011fa) (%60 Ba\u015far\u0131)"

        # AYI FORMASYONLARI
        if (is_bear[2] and is_bear[3] and is_bear[4] and 
            c[4] < c[3] < c[2] and o[3] < o[2] and o[4] < o[3] and
            lower_wick[2]/cr[2] < 0.25 and lower_wick[3]/cr[3] < 0.25 and lower_wick[4]/cr[4] < 0.25):
            return "\U0001f985 Three Black Crows - \u00dc\u00e7 Kara Karga (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%79 Ba\u015far\u0131)"

        if (is_bull[2] and body[3]/cr[3] < 0.15 and is_bear[4] and l[3] > h[2] and h[4] < l[3] and c[4] < (o[2] + c[2])/2):
            return "\U0001f476 Bearish Abandoned Baby - Terk Edilmi\u015f Bebek (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%75 Ba\u015far\u0131)"

        if (is_bull[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bear[4] and c[4] < (o[2] + c[2])/2):
            return "\U0001f319 Evening Doji Star - Ak\u015fam Y\u0131ld\u0131z\u0131 (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%72 Ba\u015far\u0131)"

        if (is_bull[2] and is_bear[3] and o[3] >= c[2] and c[3] <= o[2] and is_bear[4] and c[4] < c[3]):
            return "\U0001f534 Three Outside Down (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%72 Ba\u015far\u0131)"

        if (is_bear[0] and body[0]/cr[0] > 0.4 and is_bear[4] and c[4] < l[0] and max(h[1], h[2], h[3]) <= h[0]):
            return "\U0001f4c9 Falling Three Methods - D\u00fc\u015fen \u00dc\u00e7 Y\u00f6ntem (D\u00fc\u015f\u00fc\u015f Devam / Ay\u0131) (%71 Ba\u015far\u0131)"

        if (is_bear[3] and is_bear[4] and h[3] < l[2] and c[4] < c[3] and o[4] <= o[3]):
            return "\U0001f570\ufe0f Two Black Gapping - Bo\u015fluklu \u0130ki K\u0131rm\u0131z\u0131 (D\u00fc\u015f\u00fc\u015f Devam / Ay\u0131) (%68 Ba\u015far\u0131)"

        if (is_bull[1] and is_bull[2] and is_bull[3] and is_bear[4] and 
            c[3] > c[2] > c[1] and o[4] >= c[3] and c[4] <= o[1]):
            return "\u2694\ufe0f Bearish Three-Line Strike (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%65 Ba\u015far\u0131)"

        if (is_bull[3] and is_bear[4] and o[4] >= c[3] and c[4] <= o[3]):
            return "\U0001f534 Bearish Engulfing - Yutan Ay\u0131 (D\u00f6n\u00fc\u015f / Ay\u0131) (%63 Ba\u015far\u0131)"

        if (upper_wick[4] >= 2 * body[4] and lower_wick[4] <= 0.15 * cr[4]):
            return "\U0001f320 Shooting Star - Kayan Y\u0131ld\u0131z (D\u00fc\u015f\u00fc\u015f D\u00f6n\u00fc\u015f / Ay\u0131) (%60 Ba\u015far\u0131)"

        return "Standart Mum"
    except Exception:
        return "Standart Mum"


def scan_single_coin(symbol: str):
    """
    Tek bir koini 3 dakikalik periyotta Koncorde ve OSCAR ile analiz eder.
    Hem anlik canli mumu (-1) hem de yeni kapanmis son mumu (-2) kontrol ederek
    hicbir sinyalin kacirilmasina izin vermez.
    """
    try:
        df = get_binance_klines(symbol, interval="3m", limit=120)
        if df.empty or len(df) < 40:
            return []

        # 1. OSCAR Hesaplamasi
        res_osc = calculate_oscar(df)
        if res_osc is None:
            return []

        # 2. KONCORDE Hesaplamasi
        res_kon = calculate_koncorde(df)
        if res_kon is None:
            return []

        messages_to_send = []

        # Hem anlik devam eden mumu (-1) hem de son kapanmis mumu (-2) tara
        for target_idx in [-1, -2]:
            candle_time = df.index[target_idx]
            candle_price = float(df['Close'].iloc[target_idx])
            time_str = candle_time.strftime('%H:%M')

            b_tri = bool(res_osc['buy_tri'].iloc[target_idx])
            b_arr = bool(res_osc['buy_arr'].iloc[target_idx])
            s_tri = bool(res_osc['sell_tri'].iloc[target_idx])
            s_arr = bool(res_osc['sell_arr'].iloc[target_idx])

            detected_signal = None
            signal_title = ""
            is_buy_signal = False

            if b_tri:
                detected_signal = "BUY_TRIANGLE"
                signal_title = "\u2b1b Siyah \u00dc\u00e7gen (Dip Kesi\u015fimi AL)"
                is_buy_signal = True
            elif b_arr:
                detected_signal = "BUY_ARROW"
                signal_title = "\U0001f7e9 Ye\u015fil Ok (Al\u0131m Teyidi / Devam)"
                is_buy_signal = True
            elif s_tri:
                detected_signal = "SELL_TRIANGLE"
                signal_title = "\u2b1b Siyah Ters \u00dc\u00e7gen (Tepe K\u0131r\u0131l\u0131m\u0131 SAT)"
                is_buy_signal = False
            elif s_arr:
                detected_signal = "SELL_ARROW"
                signal_title = "\U0001f7e5 K\u0131rm\u0131z\u0131 Ok (Sat\u0131m Teyidi / Devam)"
                is_buy_signal = False

            if not detected_signal:
                continue

            # Ayni mum ve ayni sinyal daha once gonderilmisse tekrar atma
            sig_key = (symbol, str(candle_time), detected_signal)
            if sig_key in SENT_SIGNALS:
                continue
            SENT_SIGNALS.add(sig_key)

            # KONCORDE PARAMETRELERI
            azul_val = float(res_kon['azul'].iloc[target_idx])
            verde_val = float(res_kon['verde'].iloc[target_idx])
            media_val = float(res_kon['media'].iloc[target_idx])
            marron_val = float(res_kon['marron'].iloc[target_idx])

            # 1. Kurumsal Para (Azul / Balina Etiketi)
            if azul_val > 5.0:
                whale_tag = f"\U0001f40b \u00c7ok G\u00fc\u00e7l\u00fc Balina Al\u0131m\u0131 (Azul: +{azul_val:.1f})"
            elif azul_val > 0.0:
                whale_tag = f"\U0001f40b Balina Giri\u015fi Var (Azul: +{azul_val:.1f})"
            elif azul_val < -5.0:
                whale_tag = f"\u26a0\ufe0f Yo\u011fun Balina Sat\u0131\u015f\u0131 (Azul: {azul_val:.1f})"
            else:
                whale_tag = f"\u26a0\ufe0f Balina \u00c7\u0131k\u0131\u015f\u0131 (Azul: {azul_val:.1f})"

            # 2. Trend Yonu (Verde / Media Etiketi)
            if verde_val > media_val:
                trend_tag = f"\U0001f7e2 Bo\u011fa B\u00f6lgesi (Verde: {verde_val:.1f} > Media: {media_val:.1f})"
            else:
                trend_tag = f"\U0001f534 Ay\u0131 B\u00f6lgesi (Verde: {verde_val:.1f} < Media: {media_val:.1f})"

            # 3. Marron (Fiyat Egilimi)
            if marron_val > 0:
                marron_tag = f"\U0001f7e2 Pozitif B\u00f6lge (+{marron_val:.1f})"
            else:
                marron_tag = f"\U0001f534 Negatif B\u00f6lge ({marron_val:.1f})"

            # 4. Hacim Gucu (RVOL 20 Mum)
            vol = df['Volume']
            curr_vol = float(vol.iloc[target_idx])
            avg_vol = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
            rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0

            if rvol >= 1.5:
                hacim_metni = f"\U0001f680 \u00c7ok G\u00fc\u00e7l\u00fc (Ortalaman\u0131n {rvol:.1f}x Kat\u0131)"
            elif rvol >= 1.1:
                hacim_metni = f"\U0001f7e2 G\u00fc\u00e7l\u00fc (Ortalaman\u0131n {rvol:.1f}x Kat\u0131)"
            elif rvol >= 0.8:
                hacim_metni = f"\u26aa Normal (Ortalaman\u0131n {rvol:.1f}x Kat\u0131)"
            else:
                hacim_metni = f"\u26a0\ufe0f Zay\u0131f (Ortalaman\u0131n {rvol:.1f}x Kat\u0131)"

            # 5. Mum Formasyonu
            sub_df = df.iloc[:target_idx+1] if target_idx != -1 else df
            candle_pat = detect_candlestick_patterns(sub_df)

            # 6. Guven Skoru (1 - 5 Yildiz)
            puan = 1
            if is_buy_signal:
                if azul_val > 0:
                    puan += 1
                if verde_val > media_val:
                    puan += 1
                if rvol >= 1.1:
                    puan += 1
                if ("Bo\u011fa" in candle_pat) or ("Y\u00fckseli\u015f" in candle_pat):
                    puan += 1
            else:
                if azul_val < 0:
                    puan += 1
                if verde_val < media_val:
                    puan += 1
                if rvol >= 1.1:
                    puan += 1
                if ("Ay\u0131" in candle_pat) or ("D\u00fc\u015f\u00fc\u015f" in candle_pat):
                    puan += 1

            puan = min(puan, 5)
            yildizlar = "\u2b50" * puan
            skor_metni = f"{yildizlar} ({puan}/5)"

            coin_name = symbol.replace("USDT", "")
            tv_link = f"https://tr.tradingview.com/chart/?symbol=BINANCE:{coin_name}USDT"

            tag = "\U0001f7e2 <b>KR\u0130PTO AL S\u0130NYAL\u0130</b> <b>(Koncorde + OSCAR)</b>" if is_buy_signal else "\U0001f534 <b>KR\u0130PTO SAT S\u0130NYAL\u0130</b> <b>(Koncorde + OSCAR)</b>"

            # Tam ve eksiksiz etiket duzeni (crypto_main_clean.py formatinda)
            message = (
                f"{tag}\n\n"
                f"\U0001fa99 <b>Koin:</b> <a href=\"{tv_link}\"><b>#{coin_name}/USDT</b></a> <i>(<a href=\"{tv_link}\">Grafi\u011fi A\u00e7 \u2197</a>)</i>\n"
                f"\U0001f3af <b>Sinyal T\u00fcr\u00fc:</b> {signal_title}\n"
                f"\u23f1 <b>Zaman Dilimi:</b> 3 Dakika (3m)\n"
                f"\U0001f552 <b>Mum Saati:</b> <code>{time_str}</code> (TS\u0130)\n"
                f"\U0001f4b5 <b>Fiyat:</b> ${candle_price:,.4f}\n\n"
                f"<b>\u2b50 Sinyal G\u00fcven Puan\u0131:</b> {skor_metni}\n"
                f"<b>\U0001f4ca Hacim G\u00fcc\u00fc:</b> {hacim_metni}\n\n"
                f"<b>\U0001f56f\ufe0f Formasyon & Trend Teyitleri:</b>\n"
                f"\u25ab\ufe0f <b>Mum Formasyonu:</b> {candle_pat}\n\n"
                f"<b>\U0001f30a Koncorde 2.0 Teyitleri:</b>\n"
                f"\u25ab\ufe0f <b>Kurumsal Para (Azul / Balina):</b> {whale_tag}\n"
                f"\u25ab\ufe0f <b>Piyasa Trendi (Verde / Media):</b> {trend_tag}\n"
                f"\u25ab\ufe0f <b>Fiyat E\u011filimi (Marron):</b> {marron_tag}\n"
            )
            messages_to_send.append(message)

        return messages_to_send

    except Exception as e:
        return []


def run_scanner():
    """Tum Quantfury koinlerini paralel olarak tarar."""
    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- [{now_str}] KONCORDE + OSCAR 3m Tarama Baslatildi ({len(QUANTFURY_COINS)} Koin) ---")

    found_signals = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in QUANTFURY_COINS}
        for future in as_completed(future_to_coin):
            try:
                msgs = future.result()
                if msgs:
                    for msg in msgs:
                        send_telegram(msg)
                        found_signals += 1
            except Exception as e:
                print(f"Tarama alt islem hatasi: {e}")

    elapsed = time.time() - start_time
    print(f"Tarama tamamlandi! Bulunan Sinyal: {found_signals} | Sure: {elapsed:.2f} saniye")


if __name__ == "__main__":
    is_loop = "--loop" in sys.argv
    duration_minutes = None

    if "--duration" in sys.argv:
        try:
            d_idx = sys.argv.index("--duration")
            duration_minutes = float(sys.argv[d_idx + 1])
        except Exception:
            pass

    if is_loop:
        dur_msg = f"({duration_minutes} dakika boyunca)" if duration_minutes else "(suresiz)"
        print(f"KONCORDE + OSCAR 3m Tarayici dongu modunda calisiyor {dur_msg} - Her 60 saniyede bir kontrol...")
        start_ts = time.time()
        while True:
            run_scanner()
            if duration_minutes and (time.time() - start_ts) >= (duration_minutes * 60 - 10):
                print(f"Belirlenen calisma suresi ({duration_minutes} dakika) tamamlandi. Gorev basariyla sonlandiriliyor.")
                break
            time.sleep(60)  # Canli mumu aninda yakalamak icin her 60 saniyede bir kontrol
    else:
        run_scanner()
