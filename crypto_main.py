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

# Sadece 1 Saatlik Mumlar
TIMEFRAMES = [
    ("1h", "1 Saat (1h)")
]

def send_telegram(message: str):
    """Telegram'a HTML formatında bildirim gönderir."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML"
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            print("Telegram bildirimi iletildi.")
        else:
            print(f"Telegram hatası ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")

def get_binance_klines(symbol: str, interval: str) -> pd.DataFrame:
    """Binance resmi engelsiz sunucusu üzerinden TradingView ile birebir mumları çeker."""
    url_binance = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit=200"
    try:
        res = requests.get(url_binance, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 20:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                df['Open'] = df['Open'].astype(float)
                df['High'] = df['High'].astype(float)
                df['Low'] = df['Low'].astype(float)
                df['Close'] = df['Close'].astype(float)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    # Yedek: Bybit
    interval_map = {"1h": "60"}
    bb_int = interval_map.get(interval, "60")
    url_bybit = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bb_int}&limit=200"
    try:
        res = requests.get(url_bybit, timeout=8)
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

def calculate_slingshot(df: pd.DataFrame, idx: int):
    """Sling Shot System: Düz Kanal Rengi ve Noktasal Trend Rengi hesabı."""
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

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

    # Düz Trend Kanalı Rengi
    if (v_ma1 > v_upper) and (v_ma2 > v_upper) and (v_ma3 > v_upper):
        kanal_renk = "🟢 Yeşil"
    elif (v_ma1 < v_lower) and (v_ma2 < v_lower) and (v_ma3 < v_lower):
        kanal_renk = "🔴 Kırmızı"
    else:
        kanal_renk = "🔵 Mavi"

    # Noktasal Trend Çizgileri Rengi
    if (v_ma1 > v_ma2) and (v_ma2 > v_ma3):
        nokta_renk = "🟢 Yeşil"
    elif (v_ma1 < v_ma2) and (v_ma2 < v_lower if False else v_ma2 < v_ma3):
        nokta_renk = "🔴 Kırmızı"
    else:
        nokta_renk = "🟡 Sarı"

    return kanal_renk, nokta_renk

def evaluate_eco_crypto(df: pd.DataFrame, symbol: str, tf_label: str):
    """TradingView Evan Cabral Oscillators (ECO) ve SlingShot teyidi."""
    if df is None or df.empty or len(df) < 20:
        return []

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

    signals = []
    coin_name = symbol.replace("USDT", "")
    now = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=3)
    candle_duration = pd.Timedelta(hours=1)

    for idx in [-1, -2]:
        sig_type = None
        if cross_up.iloc[idx]:
            sig_type = "BUY"
        elif cross_down.iloc[idx]:
            sig_type = "SELL"

        if sig_type is not None:
            candle_time = df.index[idx]
            if getattr(candle_time, 'tzinfo', None) is not None:
                candle_time = candle_time.tz_convert('+03:00').tz_localize(None)

            if idx == -2:
                # Kapanmış saatlik mumun üzerinden 20 dakikadan fazla geçmişse ESKİDİR, gönderme!
                candle_close_time = candle_time + candle_duration
                minutes_since_close = (now - candle_close_time).total_seconds() / 60.0
                if minutes_since_close > 20.0:
                    continue
                durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"
            else:
                durum_metni = "⚠️ CANLI MUM (Kapanış Beklenmedi / Anlık)"

            candle_price = float(close.iloc[idx])
            c_st = float(stoch.iloc[idx])
            p_st = float(stoch.iloc[idx - 1])
            time_str = candle_time.strftime('%H:%M')

            # Sling Shot Trend Teyitleri
            kanal_renk, nokta_renk = calculate_slingshot(df, idx)

            tag = "🟢 <b>KRİPTO AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>KRİPTO SAT SİNYALİ</b>"
            trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

            msg = (
                f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
                f"🪙 <b>Koin:</b> #{coin_name}/USDT\n"
                f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
                f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
                f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
                f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
                f"📊 <b>DMI-Stoch:</b> {c_st:.1f} (Önceki: {p_st:.1f})\n"
                f"🎯 <b>Tetikleyici:</b> DMI-Stoch {trigger}\n\n"
                f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
                f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
                f"▫️ <b>Noktasal Trend:</b> {nokta_renk}"
            )
            signals.append(msg)
            break

    return signals

def scan_single_coin(symbol: str):
    """Tek bir koin için sadece 1h periyodunu tarar."""
    found_signals = []
    try:
        for tf_key, label in TIMEFRAMES:
            df = get_binance_klines(symbol, tf_key)
            sigs = evaluate_eco_crypto(df, symbol, label)
            if sigs:
                found_signals.extend(sigs)
    except Exception as e:
        print(f"{symbol} analiz hatası: {e}")
    return found_signals

def main():
    print(f"Kripto Evan Cabral (ECO) Taraması Başlıyor ({len(COINS)} Koin; Sadece 1 Saatlik)...")
    toplam = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_coin, coin): coin for coin in COINS}
        for future in as_completed(futures):
            try:
                sigs = future.result()
                if sigs:
                    for msg in sigs:
                        send_telegram(msg)
                        toplam += 1
            except Exception as e:
                print(f"İş parçacığı hatası: {e}")

    print(f"Tarama bitti! Üretilen yeni sinyal sayısı: {toplam}")

if __name__ == "__main__":
    main()
