import os
import sys
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
    """BIST'teki tüm aktif hisseleri dinamik olarak çeker."""
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

def calculate_stoch_rsi(series: pd.Series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI hesaplar."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-10) * 100
    k = stoch.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d

def check_evan_cabral(df: pd.DataFrame, symbol: str, tf_label: str):
    """Verilen periyot verisinde Evan Cabral stratejisini kontrol eder."""
    if df.empty or len(df) < 35:
        return None

    close = df["Close"].squeeze()

    # 1. Stochastic RSI
    k, d = calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)

    # 2. Bollinger Bantları (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)

    curr_price = float(close.iloc[-1])
    prev_k = float(k.iloc[-2])
    curr_k = float(k.iloc[-1])
    prev_d = float(d.iloc[-2])
    curr_d = float(d.iloc[-1])
    curr_lower = float(lower_bb.iloc[-1])
    curr_upper = float(upper_bb.iloc[-1])

    hisse_adi = symbol.replace(".IS", "")

    # AL Sinyali: Aşırı satımdan (<25) yukarı dönüş + Bollinger alt bandına temas/yakınlık
    if (prev_k <= prev_d) and (curr_k > curr_d) and (prev_k < 25) and (curr_price <= curr_lower * 1.015):
        return (
            f"🟢 *BIST AL SİNYALİ (Evan Cabral)*\n\n"
            f"📌 *Hisse:* #{hisse_adi}\n"
            f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
            f"💵 *Fiyat:* {curr_price:.2f} TL\n"
            f"📊 *Stoch RSI (%K):* {curr_k:.1f}\n"
            f"🎯 *Koşul:* Alt bant tepkisi ve aşırı satımdan yukarı dönüş."
        )

    # SAT Sinyali: Aşırı alımdan (>75) aşağı dönüş + Bollinger üst bandına temas/yakınlık
    elif (prev_k >= prev_d) and (curr_k < curr_d) and (prev_k > 75) and (curr_price >= curr_upper * 0.985):
        return (
            f"🔴 *BIST SAT SİNYALİ (Evan Cabral)*\n\n"
            f"📌 *Hisse:* #{hisse_adi}\n"
            f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
            f"💵 *Fiyat:* {curr_price:.2f} TL\n"
            f"📊 *Stoch RSI (%K):* {curr_k:.1f}\n"
            f"🎯 *Koşul:* Üst bant direnci ve aşırı alımdan aşağı dönüş."
        )

    return None

def analyze_ticker(symbol: str):
    """Tek bir hisse için 15m, 1h, 4h ve 1d periyotlarını analiz eder."""
    signals = []
    try:
        # 1. 15 Dakikalık Veri
        df_15m = yf.download(symbol, period="5d", interval="15m", progress=False)
        sig_15m = check_evan_cabral(df_15m, symbol, "15 Dakika (15m)")
        if sig_15m:
            signals.append(sig_15m)

        # 2. 1 Saatlik Veri
        df_1h = yf.download(symbol, period="1mo", interval="1h", progress=False)
        sig_1h = check_evan_cabral(df_1h, symbol, "1 Saat (1h)")
        if sig_1h:
            signals.append(sig_1h)

        # 3. 4 Saatlik Veri (1 Saatlik barlar birleştirilerek oluşturulur)
        if not df_1h.empty and len(df_1h) >= 40:
            df_4h = df_1h.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            sig_4h = check_evan_cabral(df_4h, symbol, "4 Saat (4h)")
            if sig_4h:
                signals.append(sig_4h)

        # 4. 1 Günlük Veri
        df_1d = yf.download(symbol, period="6mo", interval="1d", progress=False)
        sig_1d = check_evan_cabral(df_1d, symbol, "Günlük (1D)")
        if sig_1d:
            signals.append(sig_1d)

    except Exception:
        pass

    return signals

def main():
    tickers = get_all_bist_tickers()
    print(f"BIST Çoklu Zaman Dilimi Taraması Başlıyor ({len(tickers)} hisse; 15m, 1h, 4h, 1d)...")
    
    toplam_sinyal = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            results = future.result()
            if results:
                for sig in results:
                    send_telegram(sig)
                    toplam_sinyal += 1

    print(f"Tarama bitti! Toplam tespit edilen sinyal sayısı: {toplam_sinyal}")

if __name__ == "__main__":
    main()
