import os
import sys
import numpy as np
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("HATA: Telegram Token veya Chat ID bulunamadı!")
    sys.exit(1)

def get_all_bist_tickers():
    """BIST'teki tüm aktif hisseleri çeker."""
    url = "https://scanner.tradingview.com/turkey/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "options": {"lang": "tr"},
        "symbols": {"query": {"types": []}},
        "columns": ["name"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 1000]
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        data = res.json()
        tickers = [item["d"][0] + ".IS" for item in data.get("data", []) if "d" in item and len(item["d"]) > 0]
        if len(tickers) > 100:
            return tickers
    except Exception as e:
        print(f"Dinamik liste hatası: {e}")

    # Yedek liste
    return [
        "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS", 
        "GARAN.IS", "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", 
        "SISE.IS",  "SAHOL.IS", "FROTO.IS", "TOASO.IS", "ENKAI.IS"
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
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"Telegram hatası: {e}")

def wwma(series: pd.Series, length: int) -> pd.Series:
    """Pine Script: wwma(l,p) => (nz(wwma) * (l - 1) + p) / l"""
    vals = series.fillna(0.0).values
    res = np.zeros(len(vals))
    for i in range(len(vals)):
        prev = res[i-1] if i > 0 else 0.0
        res[i] = (prev * (length - 1) + vals[i]) / length
    return pd.Series(res, index=series.index)

def rsi(series: pd.Series, period: int) -> pd.Series:
    """Pine Script standart rsi hesabı."""
    delta = series.diff().fillna(0.0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wwma(gain, period)
    avg_loss = wwma(loss, period)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def stoch_rsi(series: pd.Series, len_rsi: int, len_stoch: int, smooth_k: int, smooth_d: int):
    """Pine Script k ve d hesaplaması."""
    r = rsi(series, len_rsi)
    low_r = r.rolling(len_stoch).min()
    high_r = r.rolling(len_stoch).max()
    stoch = 100 * (r - low_r) / (high_r - low_r).replace(0, 1e-10)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def evaluate_eco(df: pd.DataFrame, symbol: str, tf_label: str):
    """Pine Script kodundaki matematiksel formülü birebir hesaplar."""
    if df.empty or len(df) < 25:
        return None

    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()

    # True Range (tr)
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

    stoch = (sum_osc_lo / sum_hi_lo.replace(0, np.nan)) * 100

    prev_stoch = float(stoch.iloc[-2])
    curr_stoch = float(stoch.iloc[-1])
    curr_price = float(close.iloc[-1])
    hisse_adi = symbol.replace(".IS", "")

    # Pine Script: crossUp = Stoch < 10 and Stoch > 10 ? 1 : 0
    is_buy = (prev_stoch < 10) and (curr_stoch > 10)

    # Pine Script: crossDown = Stoch > 90 and Stoch < 90 ? 1 : 0
    is_sell = (prev_stoch > 90) and (curr_stoch < 90)

    if is_buy:
        return (
            f"🟢 *BIST AL SİNYALİ (Evan Cabral - ECO)*\n\n"
            f"📌 *Hisse:* #{hisse_adi}\n"
            f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
            f"💵 *Fiyat:* {curr_price:.2f} TL\n"
            f"📊 *DMI-Stoch:* {curr_stoch:.2f} (Önceki: {prev_stoch:.2f})\n"
            f"🎯 *Tetikleyici:* DMI-Stoch 10 seviyesini yukarı kesti ('B')."
        )
    elif is_sell:
        return (
            f"🔴 *BIST SAT SİNYALİ (Evan Cabral - ECO)*\n\n"
            f"📌 *Hisse:* #{hisse_adi}\n"
            f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
            f"💵 *Fiyat:* {curr_price:.2f} TL\n"
            f"📊 *DMI-Stoch:* {curr_stoch:.2f} (Önceki: {prev_stoch:.2f})\n"
            f"🎯 *Tetikleyici:* DMI-Stoch 90 seviyesini aşağı kesti ('S')."
        )

    return None

def analyze_ticker(symbol: str):
    """4 periyotta (15m, 1h, 4h, 1d) indikatörü çalıştırır."""
    signals = []
    try:
        # 1. 15 Dakikalık
        df_15m = yf.download(symbol, period="5d", interval="15m", progress=False)
        s15 = evaluate_eco(df_15m, symbol, "15 Dakika (15m)")
        if s15: signals.append(s15)

        # 2. 1 Saatlik
        df_1h = yf.download(symbol, period="1mo", interval="1h", progress=False)
        s1h = evaluate_eco(df_1h, symbol, "1 Saat (1h)")
        if s1h: signals.append(s1h)

        # 3. 4 Saatlik
        if not df_1h.empty and len(df_1h) >= 30:
            df_4h = df_1h.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            s4h = evaluate_eco(df_4h, symbol, "4 Saat (4h)")
            if s4h: signals.append(s4h)

        # 4. Günlük (1D)
        df_1d = yf.download(symbol, period="6mo", interval="1d", progress=False)
        s1d = evaluate_eco(df_1d, symbol, "Günlük (1D)")
        if s1d: signals.append(s1d)

    except Exception:
        pass

    return signals

def main():
    tickers = get_all_bist_tickers()
    print(f"Evan Cabral Oscillators (ECO) Taraması Başlıyor ({len(tickers)} hisse; 15m, 1h, 4h, 1d)...")
    
    toplam_sinyal = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            results = future.result()
            if results:
                for sig in results:
                    send_telegram(sig)
                    toplam_sinyal += 1

    print(f"Tarama tamamlandı! Üretilen sinyal sayısı: {toplam_sinyal}")

if __name__ == "__main__":
    main()
