import os
import sys
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests

# ==========================================
# TELEGRAM CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# QUANTFURY CRYPTO LIST (Binance USDT Pairs)
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
# INDICATOR PARAMETERS
# ==========================================
# OSCAR Settings
OSCAR_LEN = 8
SLOW_OSCAR_LEN = 16
CROSS_SENSITIVITY = 1.5
BOTTOM_SIGNAL_LINE = 35.0
TOP_SIGNAL_LINE = 65.0

# Koncorde Settings
KONCORDE_MFI_LEN = 14
KONCORDE_RSI_LEN = 14
KONCORDE_BOLLINGER_LEN = 25
KONCORDE_BOLLINGER_DEV = 2.0
KONCORDE_MEDIA_LEN = 15

# RVOL Settings
RVOL_LEN = 20

# Memory to avoid duplicate alerts
SENT_SIGNALS = set()

# Candlestick Pattern Win Rates (Historical averages)
PATTERN_WIN_RATES = {
    "Bullish Engulfing": 63,
    "Bearish Engulfing": 62,
    "Hammer": 60,
    "Shooting Star": 59,
    "Morning Star": 67,
    "Evening Star": 65,
    "Doji": 52
}


def send_telegram(message: str) -> bool:
    """Sends HTML formatted message to Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Message (Simulation)]:\n", message)
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
            print(f"Rate limited. Retry after {retry_after}s")
            return False
        else:
            return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def get_binance_klines(symbol: str, interval: str = "3m", limit: int = 100) -> pd.DataFrame:
    """Fetches kline data from Binance API with Bybit fallback."""
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 30:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
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
            if raw_list and len(raw_list) >= 30:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()


def calculate_oscar(df: pd.DataFrame):
    """Calculates OSCAR indicator lines and signal conditions."""
    if df is None or len(df) < 25:
        return None

    close = df['Close']
    a = close.rolling(window=OSCAR_LEN).max()
    b = close.rolling(window=OSCAR_LEN).min()
    denom = (a - b).replace(0, 1e-10)
    oscar_rough = (close - b) / denom * 100.0
    oscar_rough_prev = oscar_rough.shift(1)
    oscar = (oscar_rough_prev / 3.0) * 2.0 + (oscar_rough / 3.0)

    smoothed_rough = oscar_rough.rolling(window=OSCAR_LEN).mean()
    smoothed_oscar = oscar.rolling(window=OSCAR_LEN).mean()
    cross_sens = (smoothed_rough - smoothed_oscar).abs()

    crossover = (smoothed_rough > smoothed_oscar) & (smoothed_rough.shift(1) <= smoothed_oscar.shift(1))
    crossunder = (smoothed_rough < smoothed_oscar) & (smoothed_rough.shift(1) >= smoothed_oscar.shift(1))

    sig_black_buy = crossover & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough < BOTTOM_SIGNAL_LINE)
    sig_green_buy = crossover.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_oscar < smoothed_rough) & (smoothed_oscar < BOTTOM_SIGNAL_LINE)
    sig_black_sell = crossunder & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > TOP_SIGNAL_LINE)
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


def calculate_koncorde(df: pd.DataFrame):
    """Calculates Koncorde 2.0 components: Azul (Whales), Verde (Retail), and Media."""
    if df is None or len(df) < 30:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # 1. Money Flow Index (MFI)
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0.0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0.0), index=df.index)
    pos_mf = pos_flow.rolling(window=KONCORDE_MFI_LEN).sum()
    neg_mf = neg_flow.rolling(window=KONCORDE_MFI_LEN).sum().replace(0, 1e-10)
    mfi = 100.0 - (100.0 / (1.0 + (pos_mf / neg_mf)))

    # 2. Relative Strength Index (RSI)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=KONCORDE_RSI_LEN).mean()
    avg_loss = loss.rolling(window=KONCORDE_RSI_LEN).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 3. Bollinger Band Width %
    bb_sma = close.rolling(window=KONCORDE_BOLLINGER_LEN).mean()
    bb_std = close.rolling(window=KONCORDE_BOLLINGER_LEN).std()
    bb_upper = bb_sma + (KONCORDE_BOLLINGER_DEV * bb_std)
    bb_lower = bb_sma - (KONCORDE_BOLLINGER_DEV * bb_std)
    bbw = ((bb_upper - bb_lower) / bb_sma.replace(0, 1e-10)) * 100.0

    # 4. Koncorde Components
    # Verde (Retail Trend) = (RSI + MFI) / 2
    verde = (rsi + mfi) / 2.0
    # Azul (Whale Activity) = Volume Oscillation / Trend Normalized
    vol_sma = volume.rolling(window=20).mean().replace(0, 1e-10)
    rvol = volume / vol_sma
    azul = (mfi - rsi) * rvol
    # Media (Signal Line) = EMA/SMA of Verde
    media = verde.rolling(window=KONCORDE_MEDIA_LEN).mean()

    return {
        'verde': verde,
        'azul': azul,
        'media': media,
        'rvol': rvol
    }


def detect_candlestick_patterns(df: pd.DataFrame, idx: int = -1) -> list:
    """Detects candlestick patterns and attaches historical win rates."""
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

    # 1. Bullish Engulfing
    if p_close < p_open and c_close > c_open and c_close >= p_open and c_open <= p_close:
        patterns.append(("Bullish Engulfing", PATTERN_WIN_RATES["Bullish Engulfing"]))
    # 2. Bearish Engulfing
    if p_close > p_open and c_close < c_open and c_close <= p_open and c_open >= p_close:
        patterns.append(("Bearish Engulfing", PATTERN_WIN_RATES["Bearish Engulfing"]))
    # 3. Hammer
    if c_lower_wick >= 2 * c_body and c_upper_wick <= c_body * 0.5 and c_body / c_range > 0.1:
        patterns.append(("Hammer", PATTERN_WIN_RATES["Hammer"]))
    # 4. Shooting Star
    if c_upper_wick >= 2 * c_body and c_lower_wick <= c_body * 0.5 and c_body / c_range > 0.1:
        patterns.append(("Shooting Star", PATTERN_WIN_RATES["Shooting Star"]))
    # 5. Doji
    if c_body / c_range <= 0.1:
        patterns.append(("Doji", PATTERN_WIN_RATES["Doji"]))
    # 6. Morning Star
    if (p2_close < p2_open and
        abs(p_close - p_open) / (p_high - p_low + 1e-10) < 0.3 and
        c_close > c_open and c_close > (p2_open + p2_close) / 2):
        patterns.append(("Morning Star", PATTERN_WIN_RATES["Morning Star"]))
    # 7. Evening Star
    if (p2_close > p2_open and
        abs(p_close - p_open) / (p_high - p_low + 1e-10) < 0.3 and
        c_close < c_open and c_close < (p2_open + p2_close) / 2):
        patterns.append(("Evening Star", PATTERN_WIN_RATES["Evening Star"]))

    return patterns


def calculate_confidence_score(signal_type: str, koncorde: dict, rvol: float, patterns: list, target_idx: int = -1) -> tuple:
    """Calculates a 1-5 confidence score tag based on indicator alignment."""
    score = 1  # Base score for OSCAR trigger

    verde_val = float(koncorde['verde'].iloc[target_idx])
    media_val = float(koncorde['media'].iloc[target_idx])
    azul_val = float(koncorde['azul'].iloc[target_idx])

    is_buy = "BUY" in signal_type
    is_sell = "SELL" in signal_type

    # +1 for Koncorde Verde vs Media alignment
    if is_buy and verde_val > media_val:
        score += 1
    elif is_sell and verde_val < media_val:
        score += 1

    # +1 for Koncorde Azul (Whale activity alignment)
    if is_buy and azul_val > 0:
        score += 1
    elif is_sell and azul_val < 0:
        score += 1

    # +1 for High Volume Force (RVOL >= 1.5)
    if rvol >= 1.5:
        score += 1

    # +1 for Supporting Candlestick Pattern
    if patterns:
        score += 1

    score = min(score, 5)

    # Confidence tags
    tag_map = {
        1: "⭐ Low Confidence (1/5)",
        2: "⭐⭐ Moderate Confidence (2/5)",
        3: "⭐⭐⭐ High Confidence (3/5)",
        4: "⭐⭐⭐⭐ Very High Confidence (4/5)",
        5: "⭐⭐⭐⭐⭐ ULTRA CONFIDENCE - WHALE & TREND ALIGNED (5/5)"
    }
    return score, tag_map[score]



def scan_single_coin(symbol: str):
    """Scans a single pair for OSCAR + Koncorde 2.0 signals."""
    try:
        df = get_binance_klines(symbol, interval="3m", limit=80)
        if df.empty or len(df) < 30:
            return None

        oscar_res = calculate_oscar(df)
        koncorde_res = calculate_koncorde(df)
        if oscar_res is None or koncorde_res is None:
            return None

        # Check both real-time live candle (-1) and newly closed candle (-2)
        for target_idx in [-1, -2]:
            candle_time = df.index[target_idx]
            candle_price = float(df['Close'].iloc[target_idx])
            time_str = candle_time.strftime('%H:%M')

            b_buy = bool(oscar_res['black_buy'].iloc[target_idx])
            g_buy = bool(oscar_res['green_buy'].iloc[target_idx])
            b_sell = bool(oscar_res['black_sell'].iloc[target_idx])
            r_sell = bool(oscar_res['red_sell'].iloc[target_idx])

            detected_signal = None
            signal_title = ""
            signal_icon = ""

            if b_buy:
                detected_signal = "BLACK_BUY"
                signal_title = "Siyah Ok - Dip Alım Sinyali"
                signal_icon = "🟢"
            elif g_buy:
                detected_signal = "GREEN_BUY"
                signal_title = "Yeşil Ok - Alım Teyidi"
                signal_icon = "✅"
            elif b_sell:
                detected_signal = "BLACK_SELL"
                signal_title = "Siyah Ok - Tepe Satım Sinyali"
                signal_icon = "🔴"
            elif r_sell:
                detected_signal = "RED_SELL"
                signal_title = "Kırmızı Ok - Satım Teyidi"
                signal_icon = "❌"

            if not detected_signal:
                continue

            sig_key = (symbol, str(candle_time), detected_signal)
            if sig_key in SENT_SIGNALS:
                continue
            SENT_SIGNALS.add(sig_key)

            # Koncorde Values
            verde = float(koncorde_res['verde'].iloc[target_idx])
            media = float(koncorde_res['media'].iloc[target_idx])
            azul = float(koncorde_res['azul'].iloc[target_idx])
            rvol = float(koncorde_res['rvol'].iloc[target_idx])

            # Whale Status (Koncorde Azul)
            if azul > 10:
                whale_status = "🐋 <b>Balina Pozisyonu:</b> Strong Buying (+ Azul)"
            elif azul > 0:
                whale_status = "🐬 <b>Balina Pozisyonu:</b> Mild Buying"
            elif azul < -10:
                whale_status = "🚨 <b>Balina Pozisyonu:</b> Heavy Selling (- Azul)"
            else:
                whale_status = "🐟 <b>Balina Pozisyonu:</b> Neutral / Retail Dominant"

            # Trend Status (Verde vs Media)
            trend_status = "📈 <b>Trend Status:</b> Bullish (Verde > Media)" if verde > media else "📉 <b>Trend Status:</b> Bearish (Verde < Media)"

            # RVOL Force
            rvol_status = f"📊 <b>Volume Force (RVOL):</b> {rvol:.2f}x average" + (" 🔥" if rvol >= 1.5 else "")

            # Candlestick Patterns with Win Rate
            candle_patterns = detect_candlestick_patterns(df, idx=target_idx)
            if candle_patterns:
                pat_list = [f"{p[0]} (Win Rate: {p[1]}%)" for p in candle_patterns]
                pattern_str = f"🕯 <b>Formasyon:</b> {', '.join(pat_list)}\n"
            else:
                pattern_str = ""

            # Confidence Score (1-5)
            score, score_tag = calculate_confidence_score(detected_signal, koncorde_res, rvol, candle_patterns, target_idx)

            coin_name = symbol.replace("USDT", "")
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"

            message = (
                f"{signal_icon} <a href=\"{tv_url}\"><b>{coin_name}/USDT</b></a>\n"
                f"🎯 <b>Sinyal:</b> {signal_title}\n"
                f"🏷 <b>Güven Derecesi:</b> {score_tag}\n"
                f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
                f"⏱ <b>Periyot:</b> 3 Dakika (3m)\n"
                f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
                f"------------------------------\n"
                f"🔹 <b>Koncorde 2.0 Analizi:</b>\n"
                f"   • {whale_status}\n"
                f"   • {trend_status}\n"
                f"   • {rvol_status}\n"
                f"{pattern_str}"
            )
            return message
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return None
    """Scans a single pair for OSCAR + Koncorde 2.0 signals."""
    try:
        df = get_binance_klines(symbol, interval="3m", limit=80)
        if df.empty or len(df) < 30:
            return None

        oscar_res = calculate_oscar(df)
        koncorde_res = calculate_koncorde(df)
        if oscar_res is None or koncorde_res is None:
            return None

        target_idx = -1
        candle_time = df.index[target_idx]
        candle_price = float(df['Close'].iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')

        b_buy = bool(oscar_res['black_buy'].iloc[target_idx])
        g_buy = bool(oscar_res['green_buy'].iloc[target_idx])
        b_sell = bool(oscar_res['black_sell'].iloc[target_idx])
        r_sell = bool(oscar_res['red_sell'].iloc[target_idx])

        detected_signal = None
        signal_title = ""
        signal_icon = ""

        if b_buy:
            detected_signal = "BLACK_BUY"
            signal_title = "Siyah Ok - Dip Alım Sinyali"
            signal_icon = "🟢"
        elif g_buy:
            detected_signal = "GREEN_BUY"
            signal_title = "Yeşil Ok - Alım Teyidi"
            signal_icon = "✅"
        elif b_sell:
            detected_signal = "BLACK_SELL"
            signal_title = "Siyah Ok - Tepe Satım Sinyali"
            signal_icon = "🔴"
        elif r_sell:
            detected_signal = "RED_SELL"
            signal_title = "Kırmızı Ok - Satım Teyidi"
            signal_icon = "❌"

        if not detected_signal:
            return None

        sig_key = (symbol, str(candle_time), detected_signal)
        if sig_key in SENT_SIGNALS:
            return None
        SENT_SIGNALS.add(sig_key)

        # Koncorde Values
        verde = float(koncorde_res['verde'].iloc[target_idx])
        media = float(koncorde_res['media'].iloc[target_idx])
        azul = float(koncorde_res['azul'].iloc[target_idx])
        rvol = float(koncorde_res['rvol'].iloc[target_idx])

        # Whale Status (Koncorde Azul)
        if azul > 10:
            whale_status = "🐋 <b>Balina Pozisyonu:</b> Strong Buying (+ Azul)"
        elif azul > 0:
            whale_status = "🐬 <b>Balina Pozisyonu:</b> Mild Buying"
        elif azul < -10:
            whale_status = "🚨 <b>Balina Pozisyonu:</b> Heavy Selling (- Azul)"
        else:
            whale_status = "🐟 <b>Balina Pozisyonu:</b> Neutral / Retail Dominant"

        # Trend Status (Verde vs Media)
        trend_status = "📈 <b>Trend Status:</b> Bullish (Verde > Media)" if verde > media else "📉 <b>Trend Status:</b> Bearish (Verde < Media)"

        # RVOL Force
        rvol_status = f"📊 <b>Volume Force (RVOL):</b> {rvol:.2f}x average" + (" 🔥" if rvol >= 1.5 else "")

        # Candlestick Patterns with Win Rate
        candle_patterns = detect_candlestick_patterns(df, idx=target_idx)
        if candle_patterns:
            pat_list = [f"{p[0]} (Win Rate: {p[1]}%)" for p in candle_patterns]
            pattern_str = f"🕯 <b>Formasyon:</b> {', '.join(pat_list)}\n"
        else:
            pattern_str = ""

        # Confidence Score (1-5)
        score, score_tag = calculate_confidence_score(detected_signal, koncorde_res, rvol, candle_patterns, target_idx)

        coin_name = symbol.replace("USDT", "")
        tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"

        message = (
            f"{signal_icon} <a href=\"{tv_url}\"><b>{coin_name}/USDT</b></a>\n"
            f"🎯 <b>Sinyal:</b> {signal_title}\n"
            f"🏷 <b>Güven Derecesi:</b> {score_tag}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⏱ <b>Periyot:</b> 3 Dakika (3m)\n"
            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
            f"------------------------------\n"
            f"🔹 <b>Koncorde 2.0 Analizi:</b>\n"
            f"   • {whale_status}\n"
            f"   • {trend_status}\n"
            f"   • {rvol_status}\n"
            f"{pattern_str}"
        )
        return message
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return None


def run_scanner():
    """Executes a single scan cycle over all coins."""
    start_time = time.time()
    found_signals = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in QUANTFURY_COINS}
        for future in as_completed(future_to_coin):
            coin = future_to_coin[future]
            try:
                msg = future.result()
                if msg:
                    send_telegram(msg)
                    found_signals += 1
            except Exception as exc:
                print(f"{coin} generated an exception: {exc}")

    elapsed = time.time() - start_time
    print(f"Scan finished in {elapsed:.2f}s. Signals found: {found_signals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantfury 3m Unified Koncorde 2.0 + OSCAR Scanner")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--duration", type=int, default=0, help="Loop duration in minutes")
    args = parser.parse_args()

    if args.loop or args.duration > 0:
        start_loop = time.time()
        while True:
            run_scanner()
            if args.duration > 0 and (time.time() - start_loop) >= args.duration * 60:
                break

            time.sleep(60)  # 60 seconds real-time live check
    else:
        run_scanner()
