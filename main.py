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

# TUPRS, TURSG ve BIST'in En Aktif Hisseleri
BIST_TICKERS = [
    "TUPRS.IS", "TURSG.IS", "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "GARAN.IS", 
    "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", "SISE.IS",  "SAHOL.IS", "FROTO.IS", 
    "TOASO.IS", "ENKAI.IS", "PGSUS.IS", "KOZAL.IS", "PETKM.IS", "EKGYO.IS", "HEKTS.IS", 
    "SASA.IS",  "ASTOR.IS", "ALARK.IS", "ARCLK.IS", "GUBRF.IS", "KRDMD.IS", "ODAS.IS", 
    "OYAKC.IS", "SOKM.IS",  "TAVHL.IS", "PKART.IS", "TKFEN.IS", "TTKOM.IS", "TCELL.IS", 
    "VESTL.IS", "MGROS.IS", "VAKBN.IS", "HALKB.IS", "ISGYO.IS", "DOHOL.IS", "KOZAA.IS", 
    "IPEKE.IS", "BERA.IS",  "CIMSA.IS", "AKSA.IS",  "AKSEN.IS", "QUAGR.IS", "CANTE.IS",
    "MIATK.IS", "REEDR.IS", "SDTTR.IS", "KONTR.IS", "EUPWR.IS", "GESAN.IS", "CWENE.IS"
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
    elif (v_ma1 < v_ma2) and (v_ma2 < v_ma3):
        nokta_renk = "🔴 Kırmızı"
    else:
        nokta_renk = "🟡 Sarı"

    return kanal_renk, nokta_renk

def build_bist_hourly(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15 dakikalık barları tam saat başlarına (10:00, 11:00...) hizalar."""
    if df_15m.empty or len(df_15m) < 8:
        return pd.DataFrame()
    df_1h = df_15m.resample('1h', closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    return df_1h

def make_bist_4h_from_15m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15m verisinden tam 09:00 ve 13:00 başlangıçlı kusursuz 4H barları oluşturur."""
    if df_15m.empty or len(df_15m) < 16:
        return pd.DataFrame()
    df_copy = df_15m.copy()
    df_copy['date'] = df_copy.index.date
    # 1. Mum: 10:00 - 12:45 (hour < 13)
    # 2. Mum: 13:00 - 18:00 (hour >= 13)
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
    """Pine Script ECO göstergesini saf mantığıyla hesaplar."""
    df = clean_df(df)
    if df.empty or len(df) < 10:
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

    # Saf ECO değerleri:
    c_curr = float(stoch.iloc[-1])
    c_prev = float(stoch.iloc[-2])
    c_prev2 = float(stoch.iloc[-3]) if len(stoch) >= 3 else c_prev

    target_idx = None
    sig_type = None
    durum_metni = ""

    # 1. CANLI MUMDA KESİŞİM (O an açık olan canlı mum, örn: TUPRS 13:00 4H mumu)
    if (c_prev < 10) and (c_curr > 10):
        target_idx = -1
        sig_type = "BUY"
        durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
    elif (c_prev > 90) and (c_curr < 90):
        target_idx = -1
        sig_type = "SELL"
        durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
    # 2. BİR ÖNCEKİ KAPANMIŞ MUMDA KESİŞİM
    elif (c_prev2 < 10) and (c_prev > 10):
        target_idx = -2
        sig_type = "BUY"
        durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"
    elif (c_prev2 > 90) and (c_prev < 90):
        target_idx = -2
        sig_type = "SELL"
        durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

    if sig_type is not None:
        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])
        c_st = float(stoch.iloc[target_idx])
        p_st = float(stoch.iloc[target_idx - 1])

        time_str = candle_time.strftime('%d.%m.%Y') if "Günlük" in tf_label else candle_time.strftime('%H:%M')
        hisse_adi = symbol.replace(".IS", "")
        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)

        tag = "🟢 <b>BIST AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>BIST SAT SİNYALİ</b>"
        trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

        return (
            f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
            f"📌 <b>Hisse:</b> #{hisse_adi}\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> {candle_price:.2f} TL\n"
            f"📊 <b>DMI-Stoch:</b> {c_st:.1f} (Önceki: {p_st:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch {trigger}\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}"
        )

    return None

def analyze_ticker(symbol: str):
    """15m verisinden 1h ve 4h barları kusursuz türetir, günlük veriyi de tarar."""
    signals = []
    
    # 1. 15m verisi indirilerek tam saatlik (1h) ve tam seanslık (4h) barlar oluşturulur
    try:
        df_15m = yf.download(symbol, period="1mo", interval="15m", progress=False)
        clean_15m = clean_df(df_15m)
        if not clean_15m.empty:
            # 1 Saatlik
            df_1h = build_bist_hourly(clean_15m)
            s1h = evaluate_eco(df_1h, symbol, "1 Saat (1h)")
            if s1h:
                signals.append(s1h)
            
            # 4 Saatlik (Tam 13:00 başlangıçlı)
            df_4h = make_bist_4h_from_15m(clean_15m)
            s4h = evaluate_eco(df_4h, symbol, "4 Saat (4h)")
            if s4h:
                signals.append(s4h)
    except Exception as e:
        pass

    # 2. Günlük (1D)
    try:
        df_1d = yf.download(symbol, period="6mo", interval="1d", progress=False)
        s1d = evaluate_eco(df_1d, symbol, "Günlük (1D)")
        if s1d:
            signals.append(s1d)
    except Exception:
        pass

    return signals

def main():
    print(f"BIST Taraması Başlıyor ({len(BIST_TICKERS)} hisse; 1h, 4h, 1D)...")
    toplam = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, ticker): ticker for ticker in BIST_TICKERS}
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    for sig in results:
                        send_telegram(sig)
                        toplam += 1
            except Exception as e:
                print(f"Hisse analiz hatası: {e}")

    print(f"BIST Taraması bitti! Bulunan toplam sinyal: {toplam}")

if __name__ == "__main__":
    main()
