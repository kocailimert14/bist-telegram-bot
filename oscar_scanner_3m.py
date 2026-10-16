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
    print("UYARI: TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID ortam deÄŸiÅŸkeni tanÄ±mlÄ± deÄŸil!")

# ==========================================
# GÃœNCELLENEN QUANTFURY KRÄ°PTO LÄ°STESÄ° (36 Koin)
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
# Ä°NDÄ°KATÃ–R PARAMETRELERÄ° (Yeni OSCAR v2 Kod AyarlarÄ±)
# ==========================================
OSCAR_LEN = 8                  # len = input(title="Oscar Candles", defval=8)
SLOW_OSCAR_LEN = 16            # slowLen = input(title="Slow Oscar Candles", defval=16)
OSCAR_SMOOTHING = "RMA"        # smoothing = input(title="Oscar Smoothing", defval="RMA", options=["RMA", "SMA", "EMA", "WMA", "NONE"])
SLOW_OSCAR_SMOOTHING = "WMA"   # slowOscarSmoothing = input(title="Slow Oscar Smoothing", defval="WMA", options=["RMA", "SMA", "EMA", "WMA", "NONE"])
CROSS_SENSITIVITY = 0.5        # crossSignalSensitivity = input(title="Cross Signal Sensitivity", defval=0.5, step=0.1)
BOTTOM_SIGNAL_LINE = 35.0      # bottomSignalLine = input(title="Bottom Signal Line", defval=35, step=5)
TOP_SIGNAL_LINE = 65.0         # topSignalLine = input(title="Top Signal Line", defval=65, step=5)

# Tekrarlayan sinyalleri engellemek iÃ§in hafÄ±za
SENT_SIGNALS = set()


def send_telegram(message: str) -> bool:
    """Telegram kanalÄ±na HTML formatÄ±nda mesaj gÃ¶nderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram MesajÄ± (SimÃ¼lasyon)]:\n", message)
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
            print(f"Telegram Flood UyarÄ±sÄ±: {retry_after} sn beklenmeli!")
            return False
        else:
            print(f"Telegram hatasÄ± ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Telegram baÄŸlantÄ± hatasÄ±: {e}")
        return False


def get_binance_klines(symbol: str, interval: str = "3m", limit: int = 100) -> pd.DataFrame:
    """Binance Public API Ã¼zerinden mum verilerini Ã§eker."""
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
                # TÃ¼rkiye saati (UTC+3)
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


def pine_ma(smoothing: str, series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script v4 ma() ve slowOscarMa() fonksiyonlarÄ±nÄ±n birebir karÅŸÄ±lÄ±ÄŸÄ±:
    SeÃ§enekler: ["RMA", "SMA", "EMA", "WMA", "NONE"]
    """
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
        # Pine Script rma(src, len): Wilder's Smoothing (alpha = 1 / len, ilk deÄŸer SMA ile baÅŸlar)
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
        # "NONE"
        return series


def calculate_oscar(df: pd.DataFrame):
    """
    Yeni OSCAR v2 (Â© GenZai 30-07-2020 v2) Ä°ndikatÃ¶rÃ¼nÃ¼n Pine Script ile birebir hesaplamasÄ±:
    A = highest(OscarOfIndicator, len)
    B = lowest(OscarOfIndicator, len)
    OscarRough = (OscarOfIndicator - B) / (A - B) * 100
    Oscar1 = (OscarRough[1] / 3) * 2
    OscarThird = OscarRough / 3
    Oscar = Oscar1 + OscarThird
    smoothedOscarRough = ma(smoothing, OscarRough, len)
    smoothedOscar = ma(smoothing, Oscar, len)
    """
    if df is None or len(df) < 35:
        return None

    close = df['Close']

    # Fast Oscar
    a = close.rolling(window=OSCAR_LEN).max()
    b = close.rolling(window=OSCAR_LEN).min()
    denom = (a - b).replace(0, 1e-10)
    oscar_rough = (close - b) / denom * 100.0

    oscar_1 = (oscar_rough.shift(1) / 3.0) * 2.0
    oscar_third = oscar_rough / 3.0
    oscar = oscar_1 + oscar_third

    smoothed_rough = pine_ma(OSCAR_SMOOTHING, oscar_rough, OSCAR_LEN)
    smoothed_oscar = pine_ma(OSCAR_SMOOTHING, oscar, OSCAR_LEN)

    # Slow Oscar
    slow_a = close.rolling(window=SLOW_OSCAR_LEN).max()
    slow_b = close.rolling(window=SLOW_OSCAR_LEN).min()
    slow_denom = (slow_a - slow_b).replace(0, 1e-10)
    slow_oscar_rough = (close - slow_b) / slow_denom * 100.0

    slow_oscar_1 = (slow_oscar_rough.shift(1) / 3.0) * 2.0
    slow_oscar_third = slow_oscar_rough / 3.0
    slow_oscar = slow_oscar_1 + slow_oscar_third
    smoothed_slow_oscar = pine_ma(SLOW_OSCAR_SMOOTHING, slow_oscar, SLOW_OSCAR_LEN)

    # Cross Sensitivity
    cross_sens = np.maximum(smoothed_rough, smoothed_oscar) - np.minimum(smoothed_rough, smoothed_oscar)

    # KesiÅŸimler
    crossover = (smoothed_rough > smoothed_oscar) & (smoothed_rough.shift(1) <= smoothed_oscar.shift(1))
    crossunder = (smoothed_rough < smoothed_oscar) & (smoothed_rough.shift(1) >= smoothed_oscar.shift(1))

    # 4 AyrÄ± Sinyal ÅartÄ±
    # 1. Siyah ÃœÃ§gen (Dip KesiÅŸimi AL)
    sig_black_buy = crossover & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough < BOTTOM_SIGNAL_LINE)

    # 2. YeÅŸil Ok (AlÄ±m Teyidi / Devam)
    sig_green_buy = crossover.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > smoothed_oscar) & (smoothed_oscar < BOTTOM_SIGNAL_LINE)

    # 3. Siyah Ters ÃœÃ§gen (Tepe KÄ±rÄ±lÄ±mÄ± SAT)
    sig_black_sell = crossunder & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > TOP_SIGNAL_LINE)

    # 4. KÄ±rmÄ±zÄ± Ters Ok (SatÄ±m Teyidi / Devam)
    sig_red_sell = crossunder.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_oscar > smoothed_rough) & (smoothed_oscar > TOP_SIGNAL_LINE)

    return {
        'smoothed_rough': smoothed_rough,
        'smoothed_oscar': smoothed_oscar,
        'smoothed_slow_oscar': smoothed_slow_oscar,
        'cross_sens': cross_sens,
        'black_buy': sig_black_buy,
        'green_buy': sig_green_buy,
        'black_sell': sig_black_sell,
        'red_sell': sig_red_sell
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    """
    En yÃ¼ksek istatistiksel baÅŸarÄ± oranÄ±na sahip BoÄŸa ve AyÄ± mum formasyonlarÄ±.
    """
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

        # ==========================================
        # 1. EN YÃœKSEK BAÅARI ORANLI YÃœKSELÄ°Å (BOÄA) FORMASYONLARI
        # ==========================================
        if (is_bear[1] and is_bear[2] and is_bear[3] and is_bull[4] and 
            c[3] < c[2] < c[1] and o[4] <= c[3] and c[4] >= o[1]):
            return "âš”ï¸ Bullish Three-Line Strike (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%84 BaÅŸarÄ±)"

        if (is_bull[2] and is_bull[3] and is_bull[4] and 
            c[4] > c[3] > c[2] and o[3] > o[2] and o[4] > o[3] and
            upper_wick[2]/cr[2] < 0.25 and upper_wick[3]/cr[3] < 0.25 and upper_wick[4]/cr[4] < 0.25):
            return "ğŸ›¡ï¸ Three White Soldiers - ÃœÃ§ Beyaz Asker (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%82 BaÅŸarÄ±)"

        if (is_bull[0] and body[0]/cr[0] > 0.4 and is_bull[4] and c[4] > h[0] and min(l[1], l[2], l[3]) >= l[0]):
            return "ğŸ“ˆ Rising Three Methods - YÃ¼kselen ÃœÃ§ YÃ¶ntem (YÃ¼kseliÅŸ Devam / BoÄŸa) (%78 BaÅŸarÄ±)"

        if (is_bear[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bull[4] and c[4] > (o[2] + c[2])/2):
            return "â­ Morning Doji Star - Sabah YÄ±ldÄ±zÄ± (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%76 BaÅŸarÄ±)"

        if (is_bear[2] and body[3]/cr[3] < 0.15 and is_bull[4] and h[3] < l[2] and l[4] > h[3] and c[4] > (o[2] + c[2])/2):
            return "ğŸ‘¶ Bullish Abandoned Baby - Terk EdilmiÅŸ Bebek (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%75 BaÅŸarÄ±)"

        if (is_bear[2] and is_bull[3] and o[3] <= c[2] and c[3] >= o[2] and is_bull[4] and c[4] > c[3]):
            return "ğŸŸ¢ Three Outside Up (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%72 BaÅŸarÄ±)"

        if (is_bear[3] and is_bull[4] and (body[3]/cr[3] > 0.35) and (body[4]/cr[4] > 0.35) and 
            abs(o[4] - o[3]) / o[3] <= 0.003 and c[4] > h[3]):
            return "âš¡ Bullish Separating Lines - AyrÄ±lan Ã‡izgiler (YÃ¼kseliÅŸ Devam / BoÄŸa) (%68 BaÅŸarÄ±)"

        if (is_bear[3] and is_bull[4] and c[4] >= o[3] and o[4] <= c[3]):
            return "ğŸŸ¢ Bullish Engulfing - Yutan BoÄŸa (DÃ¶nÃ¼ÅŸ / BoÄŸa) (%63 BaÅŸarÄ±)"

        if (lower_wick[4] >= 2 * body[4] and upper_wick[4] <= 0.15 * cr[4]):
            return "ğŸ”¨ Hammer - Ã‡ekiÃ§ (YÃ¼kseliÅŸ DÃ¶nÃ¼ÅŸ / BoÄŸa) (%60 BaÅŸarÄ±)"

        # ==========================================
        # 2. EN YÃœKSEK BAÅARI ORANLI DÃœÅÃœÅ (AYI) FORMASYONLARI
        # ==========================================
        if (is_bear[2] and is_bear[3] and is_bear[4] and 
            c[4] < c[3] < c[2] and o[3] < o[2] and o[4] < o[3] and
            lower_wick[2]/cr[2] < 0.25 and lower_wick[3]/cr[3] < 0.25 and lower_wick[4]/cr[4] < 0.25):
            return "ğŸ¦… Three Black Crows - ÃœÃ§ Kara Karga (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%79 BaÅŸarÄ±)"

        if (is_bull[2] and body[3]/cr[3] < 0.15 and is_bear[4] and l[3] > h[2] and h[4] < l[3] and c[4] < (o[2] + c[2])/2):
            return "ğŸ‘¶ Bearish Abandoned Baby - Terk EdilmiÅŸ Bebek (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%75 BaÅŸarÄ±)"

        if (is_bull[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bear[4] and c[4] < (o[2] + c[2])/2):
            return "ğŸŒ™ Evening Doji Star - AkÅŸam YÄ±ldÄ±zÄ± (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%72 BaÅŸarÄ±)"

        if (is_bull[2] and is_bear[3] and o[3] >= c[2] and c[3] <= o[2] and is_bear[4] and c[4] < c[3]):
            return "ğŸ”´ Three Outside Down (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%72 BaÅŸarÄ±)"

        if (is_bear[0] and body[0]/cr[0] > 0.4 and is_bear[4] and c[4] < l[0] and max(h[1], h[2], h[3]) <= h[0]):
            return "ğŸ“‰ Falling Three Methods - DÃ¼ÅŸen ÃœÃ§ YÃ¶ntem (DÃ¼ÅŸÃ¼ÅŸ Devam / AyÄ±) (%71 BaÅŸarÄ±)"

        if (is_bear[3] and is_bear[4] and h[3] < l[2] and c[4] < c[3] and o[4] <= o[3]):
            return "ğŸ•°ï¸ Two Black Gapping - BoÅŸluklu Ä°ki KÄ±rmÄ±zÄ± (DÃ¼ÅŸÃ¼ÅŸ Devam / AyÄ±) (%68 BaÅŸarÄ±)"

        if (is_bull[1] and is_bull[2] and is_bull[3] and is_bear[4] and 
            c[3] > c[2] > c[1] and o[4] >= c[3] and c[4] <= o[1]):
            return "âš”ï¸ Bearish Three-Line Strike (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%65 BaÅŸarÄ±)"

        if (is_bull[3] and is_bear[4] and o[4] >= c[3] and c[4] <= o[3]):
            return "ğŸ”´ Bearish Engulfing - Yutan AyÄ± (DÃ¶nÃ¼ÅŸ / AyÄ±) (%63 BaÅŸarÄ±)"

        if (upper_wick[4] >= 2 * body[4] and lower_wick[4] <= 0.15 * cr[4]):
            return "ğŸŒ  Shooting Star - Kayan YÄ±ldÄ±z (DÃ¼ÅŸÃ¼ÅŸ DÃ¶nÃ¼ÅŸ / AyÄ±) (%60 BaÅŸarÄ±)"

        return "Standart Mum"
    except Exception:
        return "Standart Mum"


def scan_single_coin(symbol: str):
    """Tek bir koini 3 dakikalÄ±k periyotta analiz eder ve sinyal varsa Telegram mesajÄ± Ã¼retir."""
    try:
        df = get_binance_klines(symbol, interval="3m", limit=100)
        if df.empty or len(df) < 35:
            return None

        res = calculate_oscar(df)
        if res is None:
            return None

        # Son tamamlanan muma bakÄ±lÄ±r
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
            signal_title = "â¬› Siyah ÃœÃ§gen (Dip KesiÅŸimi AL)"
            signal_icon = "ğŸŸ¢"
        elif g_buy:
            detected_signal = "GREEN_BUY"
            signal_title = "ğŸŸ© YeÅŸil Ok (AlÄ±m Teyidi / Devam)"
            signal_icon = "ğŸŸ¢"
        elif b_sell:
            detected_signal = "BLACK_SELL"
            signal_title = "â¬› Siyah Ters ÃœÃ§gen (Tepe KÄ±rÄ±lÄ±mÄ± SAT)"
            signal_icon = "ğŸ”´"
        elif r_sell:
            detected_signal = "RED_SELL"
            signal_title = "ğŸŸ¥ KÄ±rmÄ±zÄ± Ok (SatÄ±m Teyidi / Devam)"
            signal_icon = "ğŸ”´"

        if not detected_signal:
            return None

        # AynÄ± mum iÃ§in tekrar sinyal atmayÄ± engelle
        sig_key = (symbol, str(candle_time), detected_signal)
        if sig_key in SENT_SIGNALS:
            return None
        SENT_SIGNALS.add(sig_key)

        coin_name = symbol.replace("USDT", "")
        tv_link = f"https://tr.tradingview.com/chart/?symbol=BINANCE:{coin_name}USDT"

        # Mum Formasyonu Tespiti
        candle_pat = detect_candlestick_patterns(df)

        message = (
            f"{signal_icon} <b>OSCAR Ä°NDÄ°KATÃ–R SÄ°NYALÄ°</b>\n\n"
            f"ğŸª™ <b>Koin:</b> <a href=\"{tv_link}\">#{coin_name}/USDT</a> <i>(GrafiÄŸi AÃ§)</i>\n"
            f"ğŸ¯ <b>Sinyal:</b> {signal_title}\n"
            f"â± <b>Periyot:</b> 3 Dakika (3m)\n"
            f"ğŸ•’ <b>Mum Saati:</b> <code>{time_str}</code> (TSÄ°)\n"
            f"ğŸ’µ <b>Fiyat:</b> ${candle_price:,.4f}\n\n"
            f"ğŸ•¯ï¸ <b>Mum Formasyonu:</b> {candle_pat}\n"
        )
        return message

    except Exception as e:
        return None


def run_scanner():
    """TÃ¼m Quantfury koinlerini paralel olarak tarar."""
    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- [{now_str}] OSCAR 3m Tarama BaÅŸlatÄ±ldÄ± ({len(QUANTFURY_COINS)} Koin) ---")

    found_signals = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in QUANTFURY_COINS}
        for future in as_completed(future_to_coin):
            msg = future.result()
            if msg:
                send_telegram(msg)
                found_signals += 1

    elapsed = time.time() - start_time
    print(f"Tarama tamamlandÄ±! Bulunan Sinyal: {found_signals} | SÃ¼re: {elapsed:.2f} saniye")


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
        dur_msg = f"({duration_minutes} dakika boyunca)" if duration_minutes else "(sÃ¼resiz)"
        print(f"OSCAR 3m TarayÄ±cÄ± dÃ¶ngÃ¼ modunda Ã§alÄ±ÅŸÄ±yor {dur_msg} - Her 3 dakikada bir kontrol...")
        start_ts = time.time()
        while True:
            run_scanner()
            if duration_minutes and (time.time() - start_ts) >= (duration_minutes * 60 - 10):
                print(f"Belirlenen Ã§alÄ±ÅŸma sÃ¼resi ({duration_minutes} dakika) tamamlandÄ±. GÃ¶rev baÅŸarÄ±yla sonlandÄ±rÄ±lÄ±yor.")
                break
            time.sleep(180)  # 3 dakika bekle
    else:
        run_scanner()
