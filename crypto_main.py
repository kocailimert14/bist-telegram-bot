import os
import sys
import numpy as np
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("HATA: Telegram Token veya Chat ID bulunamadı!")
    sys.exit(1)

# İzleme Listenizdeki 35 Koin
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BCHUSDT", "LTCUSDT", 
    "NEARUSDT", "APTUSDT", "DOTUSDT", "TAOUSDT", "AAVEUSDT", 
    "RENDERUSDT", "INJUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", 
    "HBARUSDT", "OPUSDT", "ARBUSDT", "UNIUSDT", "RUNEUSDT", 
    "ONDOUSDT", "POLUSDT", "SNXUSDT", "THETAUSDT", "MANAUSDT", 
    "SANDUSDT", "ZECUSDT", "HYPEUSDT", "GRAMUSDT", "SUSDT"
]

TIMEFRAMES = [
    ("15m", "15 Dakika (15m)", "15"),
    ("1h",  "1 Saat (1h)",      "60"),
    ("4h",  "4 Saat (4h)",      "240"),
    ("1d",  "Günlük (1D)",      "D")
]

def send_telegram(message: str):
    """Telegram'a bildirim gönderir."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            print("Telegram bildirimi iletildi.")
        else:
            print(f"Telegram hatası: {res.text}")
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")

def get_crypto_klines(symbol: str, interval_code: str) -> pd.DataFrame:
    """TradingView ile birebir aynı kripto mumlarını engelsiz API üzerinden çeker."""
    # 1. Bybit Spot üzerinden dene
    url_spot = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval_code}&limit=100"
    try:
        res = requests.get(url_spot, timeout=8)
        if res.status_code == 200:
            raw_list = res.json().get('result', {}).get('list', [])
            if raw_list and len(raw_list) >= 20:
                raw_list = raw_list[::-1]  # Zamanı eskidikten yeniye sırala
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                df['Open'] = df['Open'].astype(float)
                df['High'] = df['High'].astype(float)
                df['Low'] = df['Low'].astype(float)
                df['Close'] = df['Close'].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    # 2. Bybit Linear (Vadeli) üzerinden dene (Spot'ta olmayan yeni koinler için)
    url_linear = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval_code}&limit=100"
    try:
        res = requests.get(url_linear, timeout=8)
        if res.status_code == 200:
            raw_list = res.json().get('result', {}).get('list', [])
            if raw_list and len(raw_list) >= 20:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                df['Open'] = df['Open'].astype(float)
                df['High'] = df['High'].astype(float)
                df['Low'] = df['Low'].astype(float)
                df['Close'] = df['Close'].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()

def wwma(series: pd.Series, length: int) -> pd.Series:
    """Pine Script: wwma(l,p) => (nz(wwma) * (l - 1) + p) / l"""
    vals = series.fillna(0.0).values
    res = np.zeros(len(vals))
    for i in range(len(vals)):
        prev = res[i-1] if i > 0 else 0.0
        res[i] = (prev * (length - 1) + vals[i]) / length
    return pd.Series(res, index=series.index)

def evaluate_eco_crypto(df: pd.DataFrame, symbol: str, tf_label: str):
    """TradingView Evan Cabral Oscillators (ECO) formülü."""
    if df.empty or len(df) < 20:
        return None

    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    hiDiff = high - high.shift(1)
    loDiff = low.shift(1) - low

    plusDM = pd.Series(np.where((hiDiff > loDiff) & (hiDiff > 0), hiDiff, 0.0), index=df.index)
    minusDM = pd.Series(np.where((loDiff > hiDiff) & (loDiff > 0), loDiff, 0.0), index=df.index)

    DMIlength = 10
    Stolength = 3

    ATR = wwma(tr, DMIlength)
    PlusDI = 100 * wwma(plusDM, DMIlength) / ATR.replace(0, 1e-10)
    MinusDI = 100 * wwma(minusDM, DMIlength) / ATR.replace(0, 1e-10)
    osc = PlusDI - MinusDI

    hi = osc.rolling(window=Stolength).max()
    lo = osc.rolling(window=Stolength).min()

    sum_osc_lo = (osc - lo).rolling(window=Stolength).sum()
    sum_hi_lo = (hi - lo).rolling(window=Stolength).sum()

    denom = sum_hi_lo.replace(0, 1e-10)
    stoch = (sum_osc_lo / denom) * 100
    stoch = stoch.clip(lower=0, upper=100).ffill().fillna(50.0)

    # Pine script kesişim şartları
    cross_up = (stoch.shift(1) < 10) & (stoch > 10)
    cross_down = (stoch.shift(1) > 90) & (stoch < 90)

    target_idx = None
    sig_type = None

    if cross_up.iloc[-1]:
        target_idx = -1
        sig_type = "BUY"
    elif cross_down.iloc[-1]:
        target_idx = -1
        sig_type = "SELL"
    elif cross_up.iloc[-2]:
        target_idx = -2
        sig_type = "BUY"
    elif cross_down.iloc[-2]:
        target_idx = -2
        sig_type = "SELL"

    if sig_type is not None:
        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])
        c_stoch = float(stoch.iloc[target_idx])
        p_stoch = float(stoch.iloc[target_idx - 1])

        time_str = candle_time.strftime('%d.%m.%Y') if "Günlük" in tf_label else candle_time.strftime('%H:%M')
        coin_name = symbol.replace("USDT", "")

        tag = "🟢 *KRİPTO AL SİNYALİ*" if sig_type == "BUY" else "🔴 *KRİPTO SAT SİNYALİ*"
        trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

        return (
            f"{tag} *(Evan Cabral - ECO)*\n\n"
            f"🪙 *Koin:* #{coin_name}/USDT\n"
            f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
            f"🕒 *Mum Saati:* `{time_str}` (TSİ)\n"
            f"💵 *Sinyal Fiyatı:* ${candle_price:,.4f}\n"
            f"📊 *DMI-Stoch:* {c_stoch:.1f} (Önceki: {p_stoch:.1f})\n"
            f"🎯 *Tetikleyici:* DMI-Stoch {trigger}."
        )

    return None

def scan_single_coin(symbol: str):
    """Tek bir koin için 4 periyodu tarar."""
    found_signals = []
    for tf_key, label, code in TIMEFRAMES:
        df = get_crypto_klines(symbol, code)
        sig = evaluate_eco_crypto(df, symbol, label)
        if sig:
            found_signals.append(sig)
    return found_signals

def main():
    print(f"Kripto Evan Cabral (ECO) Taraması Başlıyor ({len(COINS)} Koin; 15m, 1h, 4h, 1D)...")
    toplam = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_coin, coin): coin for coin in COINS}
        for future in as_completed(futures):
            results = future.result()
            if results:
                for msg in results:
                    send_telegram(msg)
                    toplam += 1

    print(f"Kripto taraması bitti! Bulunan toplam sinyal: {toplam}")

if __name__ == "__main__":
    main()
