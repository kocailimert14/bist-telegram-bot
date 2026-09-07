import os
import sys
import requests
import pandas as pd
import yfinance as yf

# GitHub Secrets'tan Telegram bilgilerini güvenli şekilde al
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("HATA: Telegram Token veya Chat ID bulunamadı! Lütfen Secrets ayarlarını kontrol edin.")
    sys.exit(1)

# Taranacak BIST Hisseleri (İstediğiniz hisseleri ekleyip çıkarabilirsiniz)
BIST_TICKERS = [
    "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS", 
    "GARAN.IS", "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", 
    "SISE.IS",  "SAHOL.IS", "FROTO.IS", "TOASO.IS", "ENKAI.IS",
    "PGSUS.IS", "KOZAL.IS", "PETKM.IS", "EKGYO.IS", "HEKTS.IS"
]

TIMEFRAME = "1h"  # Saatlik mumlar (İsteğe göre: 15m, 1h, 1d)

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
            print("Telegram bildirimi başarıyla iletildi.")
        else:
            print(f"Telegram hatası: {res.text}")
    except Exception as e:
        print(f"Bağlantı hatası: {e}")

def calculate_stoch_rsi(series: pd.Series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI hesaplaması."""
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

def scan_stocks():
    print(f"BIST Taraması Başlıyor ({len(BIST_TICKERS)} Hisse, Periyot: {TIMEFRAME})...")
    sinyal_sayisi = 0

    for symbol in BIST_TICKERS:
        try:
            # Hisse verisini indir
            df = yf.download(symbol, period="1mo", interval=TIMEFRAME, progress=False)
            if df.empty or len(df) < 35:
                continue

            close = df["Close"].squeeze()

            # 1. Stochastic RSI (%K, %D)
            k, d = calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)

            # 2. Bollinger Bantları (20, 2)
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_bb = sma20 + (2 * std20)
            lower_bb = sma20 - (2 * std20)

            # Son iki mumun değerleri
            curr_price = float(close.iloc[-1])
            prev_k = float(k.iloc[-2])
            curr_k = float(k.iloc[-1])
            prev_d = float(d.iloc[-2])
            curr_d = float(d.iloc[-1])
            curr_lower = float(lower_bb.iloc[-1])
            curr_upper = float(upper_bb.iloc[-1])

            hisse_adi = symbol.replace(".IS", "")

            # Evan Cabral AL Koşulu:
            # Stoch RSI aşırı satımdan (<25) yukarı kesişiyor VE Fiyat Bollinger alt bandına yakın
            if (prev_k <= prev_d) and (curr_k > curr_d) and (prev_k < 25) and (curr_price <= curr_lower * 1.015):
                msg = (
                    f"🟢 *BIST AL SİNYALİ (Evan Cabral)*\n\n"
                    f"📌 *Hisse:* #{hisse_adi}\n"
                    f"💵 *Fiyat:* {curr_price:.2f} TL\n"
                    f"⏱ *Periyot:* {TIMEFRAME}\n"
                    f"📊 *Stoch RSI (%K):* {curr_k:.1f}\n"
                    f"🎯 *Koşul:* Alt bant tepkisi ve aşırı satımdan yukarı dönüş."
                )
                print(f"AL Sinyali bulundu: {hisse_adi}")
                send_telegram(msg)
                sinyal_sayisi += 1

            # Evan Cabral SAT Koşulu:
            # Stoch RSI aşırı alımdan (>75) aşağı kesişiyor VE Fiyat Bollinger üst bandına yakın
            elif (prev_k >= prev_d) and (curr_k < curr_d) and (prev_k > 75) and (curr_price >= curr_upper * 0.985):
                msg = (
                    f"🔴 *BIST SAT SİNYALİ (Evan Cabral)*\n\n"
                    f"📌 *Hisse:* #{hisse_adi}\n"
                    f"💵 *Fiyat:* {curr_price:.2f} TL\n"
                    f"⏱ *Periyot:* {TIMEFRAME}\n"
                    f"📊 *Stoch RSI (%K):* {curr_k:.1f}\n"
                    f"🎯 *Koşul:* Üst bant direnci ve aşırı alımdan aşağı dönüş."
                )
                print(f"SAT Sinyali bulundu: {hisse_adi}")
                send_telegram(msg)
                sinyal_sayisi += 1

        except Exception as e:
            print(f"{symbol} taranırken hata: {e}")

    print(f"Tarama bitti. Toplam üretilen sinyal: {sinyal_sayisi}")

if __name__ == "__main__":
    scan_stocks()
