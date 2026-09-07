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
        if len(tickers) > 50:
            return tickers
    except Exception as e:
        print(f"Dinamik liste hatası: {e}")

    # Yedek Geniş Liste
    return [
        "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS", "GARAN.IS", 
        "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", "SISE.IS",  "SAHOL.IS", 
        "FROTO.IS", "TOASO.IS", "ENKAI.IS", "PGSUS.IS", "KOZAL.IS", "PETKM.IS", 
        "EKGYO.IS", "HEKTS.IS", "SASA.IS",  "ASTOR.IS", "ALARK.IS", "ARCLK.IS", 
        "GUBRF.IS", "KRDMD.IS", "ODAS.IS",  "OYAKC.IS", "SOKM.IS",  "TAVHL.IS",
        "PKART.IS", "TKFEN.IS", "TTKOM.IS", "TCELL.IS", "VESTL.IS", "MGROS.IS"
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

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance MultiIndex sütun yapısını ve eksik verileri temizler."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['High', 'Low', 'Close'])
    return df

def wwma(series: pd.Series, length: int) -> pd.Series:
    """Pine Script: wwma(l,p) => (nz(wwma) * (l - 1) + p) / l"""
    vals = series.fillna(0.0).values
    res = np.zeros(len(vals))
    for i in range(len(vals)):
        prev = res[i-1] if i > 0 else 0.0
        res[i] = (prev * (length - 1) + vals[i]) / length
    return pd.Series(res, index=series.index)

def make_bist_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """TradingView BIST 4 saatlik mumlarını (09:00-13:00 ve 13:00-18:10) birebir oluşturur."""
    df = clean_df(df_1h)
    if df.empty or len(df) < 15:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    # TradingView 4H Kırılımı: 09:00 - 13:00 (1. Mum) ve 13:00 - 18:10 (2. Mum)
    df_copy['half'] = np.where(df_copy.index.hour < 13, 1, 2)
    
    df_4h = df_copy.groupby(['date', 'half']).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    
    new_idx = []
    for d, h in df_4h.index:
        hour_str = "09:00:00" if h == 1 else "13:00:00"
        new_idx.append(pd.Timestamp(f"{d} {hour_str}+03:00"))
    df_4h.index = pd.DatetimeIndex(new_idx)
    return df_4h

def evaluate_eco(df: pd.DataFrame, symbol: str, tf_label: str):
    """Pine Script ECO göstergesini hesaplar ve tam mum saatiyle sinyal üretir."""
    df = clean_df(df)
    if df.empty or len(df) < 15:
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

    # En son mum veya bir önceki mumda kesişim oldu mu?
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
        hisse_adi = symbol.replace(".IS", "")

        if sig_type == "BUY":
            return (
                f"🟢 *BIST AL SİNYALİ (Evan Cabral - ECO)*\n\n"
                f"📌 *Hisse:* #{hisse_adi}\n"
                f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
                f"🕒 *Mum Saati:* `{time_str}`\n"
                f"💵 *Sinyal Fiyatı:* {candle_price:.2f} TL\n"
                f"📊 *DMI-Stoch:* {c_stoch:.1f} (Önceki: {p_stoch:.1f})\n"
                f"🎯 *Tetikleyici:* DMI-Stoch 10 seviyesini yukarı kesti ('B')."
            )
        else:
            return (
                f"🔴 *BIST SAT SİNYALİ (Evan Cabral - ECO)*\n\n"
                f"📌 *Hisse:* #{hisse_adi}\n"
                f"⏱ *Zaman Dilimi:* `{tf_label}`\n"
                f"🕒 *Mum Saati:* `{time_str}`\n"
                f"💵 *Sinyal Fiyatı:* {candle_price:.2f} TL\n"
                f"📊 *DMI-Stoch:* {c_stoch:.1f} (Önceki: {p_stoch:.1f})\n"
                f"🎯 *Tetikleyici:* DMI-Stoch 90 seviyesini aşağı kesti ('S')."
            )

    return None

def analyze_ticker(symbol: str):
    """15m, 1h, 4h ve 1d periyotlarını analiz eder."""
    signals = []
    
    # 1. 15 Dakika
    try:
        df_15m = yf.download(symbol, period="5d", interval="15m", progress=False)
        s15 = evaluate_eco(df_15m, symbol, "15 Dakika (15m)")
        if s15: signals.append(s15)
    except Exception:
        pass

    # 2. 1 Saat
    df_1h = None
    try:
        df_1h = yf.download(symbol, period="1mo", interval="1h", progress=False)
        s1h = evaluate_eco(df_1h, symbol, "1 Saat (1h)")
        if s1h: signals.append(s1h)
    except Exception:
        pass

    # 3. 4 Saat (09:00 - 13:00 ve 13:00 - 18:10 TradingView Uyumlu)
    try:
        if df_1h is not None and not df_1h.empty:
            df_4h = make_bist_4h(df_1h)
            s4h = evaluate_eco(df_4h, symbol, "4 Saat (4h)")
            if s4h: signals.append(s4h)
    except Exception:
        pass

    # 4. Günlük (1D)
    try:
        df_1d = yf.download(symbol, period="1y", interval="1d", progress=False)
        s1d = evaluate_eco(df_1d, symbol, "Günlük (1D)")
        if s1d: signals.append(s1d)
    except Exception:
        pass

    return signals

def main():
    tickers = get_all_bist_tickers()
    print(f"Evan Cabral (ECO) Taraması Başlıyor ({len(tickers)} hisse; 15m, 1h, 4h, 1D)...")
    
    toplam_sinyal = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            results = future.result()
            if results:
                for sig in results:
                    send_telegram(sig)
                    toplam_sinyal += 1

    print(f"Tarama tamamlandı! Üretilen toplam sinyal sayısı: {toplam_sinyal}")

if __name__ == "__main__":
    main()
