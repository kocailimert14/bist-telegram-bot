import os
import sys
import numpy as np
import requests
import pandas as pd
import yfinance as yf

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

    # Geniş Yedek Liste
    return [
        "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS", "GARAN.IS", 
        "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", "SISE.IS",  "SAHOL.IS", 
        "FROTO.IS", "TOASO.IS", "ENKAI.IS", "PGSUS.IS", "KOZAL.IS", "PETKM.IS", 
        "EKGYO.IS", "HEKTS.IS", "SASA.IS",  "ASTOR.IS", "ALARK.IS", "ARCLK.IS", 
        "GUBRF.IS", "KRDMD.IS", "ODAS.IS",  "OYAKC.IS", "SOKM.IS",  "TAVHL.IS", 
        "PKART.IS", "TKFEN.IS", "TTKOM.IS", "TCELL.IS", "VESTL.IS", "MGROS.IS"
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

def extract_ticker_df(batch_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Toplu yfinance tablosundan tek bir hisseyi hatasız ayıklar."""
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()
    if not isinstance(batch_df.columns, pd.MultiIndex):
        return batch_df.dropna(subset=['High', 'Low', 'Close'])
    
    try:
        if ticker in batch_df.columns.get_level_values(0):
            sub = batch_df[ticker].copy()
            return sub.dropna(subset=['High', 'Low', 'Close'])
    except Exception:
        pass
        
    try:
        if ticker in batch_df.columns.get_level_values(1):
            sub = batch_df.xs(ticker, axis=1, level=1).copy()
            return sub.dropna(subset=['High', 'Low', 'Close'])
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

    if (v_ma1 > v_upper) and (v_ma2 > v_upper) and (v_ma3 > v_upper):
        kanal_renk = "🟢 Yeşil"
    elif (v_ma1 < v_lower) and (v_ma2 < v_lower) and (v_ma3 < v_lower):
        kanal_renk = "🔴 Kırmızı"
    else:
        kanal_renk = "🔵 Mavi"

    if (v_ma1 > v_ma2) and (v_ma2 > v_ma3):
        nokta_renk = "🟢 Yeşil"
    elif (v_ma1 < v_ma2) and (v_ma2 < v_lower if False else v_ma2 < v_ma3):
        nokta_renk = "🔴 Kırmızı"
    else:
        nokta_renk = "🟡 Sarı"

    return kanal_renk, nokta_renk

def build_bist_hourly(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15 dakikalık barları 10:00, 11:00, 12:00 gibi tam saat başlarına hizalayarak saatlik mum oluşturur."""
    if df_15m.empty or len(df_15m) < 8:
        return pd.DataFrame()
    df_1h = df_15m.resample('1h', closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    return df_1h

def make_bist_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """TradingView BIST 4 saatlik mumlarını (09:00-13:00 ve 13:00-18:10) oluşturur."""
    if df_1h.empty or len(df_1h) < 10:
        return pd.DataFrame()
    
    df_copy = df_1h.copy()
    df_copy['date'] = df_copy.index.date
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

def evaluate_eco_bist(df: pd.DataFrame, symbol: str, tf_label: str, tf_key: str):
    """Pine Script ECO göstergesini saf mantığıyla hesaplar."""
    if df.empty or len(df) < 10:
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

    # Pine script saf kesişim kuralları
    cross_up = (stoch.shift(1) < 10) & (stoch > 10)
    cross_down = (stoch.shift(1) > 90) & (stoch < 90)

    signals = []
    hisse_adi = symbol.replace(".IS", "")
    now = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=3)

    target_idx = None
    sig_type = None
    durum_metni = ""

    # 1. Önce o an açık olan CANLI MUMU kontrol et (iloc[-1])
    if cross_up.iloc[-1]:
        target_idx = -1
        sig_type = "BUY"
        durum_metni = "⚠️ CANLI MUM (Kapanış Beklenmedi / Anlık)"
    elif cross_down.iloc[-1]:
        target_idx = -1
        sig_type = "SELL"
        durum_metni = "⚠️ CANLI MUM (Kapanış Beklenmedi / Anlık)"
    # 2. Eğer canlı mumda kesişim yoksa, bir önceki KAPANMIŞ MUMU kontrol et (iloc[-2])
    elif cross_up.iloc[-2]:
        target_idx = -2
        sig_type = "BUY"
        durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"
    elif cross_down.iloc[-2]:
        target_idx = -2
        sig_type = "SELL"
        durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

    if sig_type is not None:
        candle_time = df.index[target_idx]
        if getattr(candle_time, 'tzinfo', None) is not None:
            candle_time = candle_time.tz_convert('+03:00').tz_localize(None)

        # Kapanmış mum (-2) için mantıklı periyot penceresi
        if target_idx == -2:
            if tf_key == "1h":
                close_time = candle_time + pd.Timedelta(hours=1)
                if (now - close_time).total_seconds() / 60.0 > 60.0:
                    return []
            elif tf_key == "4h":
                close_time = candle_time + pd.Timedelta(hours=4)
                if (now - close_time).total_seconds() / 60.0 > 240.0:
                    return []
            elif tf_key == "1d":
                if (now.date() - candle_time.date()).days > 3:
                    return []

        candle_price = float(close.iloc[target_idx])
        c_st = float(stoch.iloc[target_idx])
        p_st = float(stoch.iloc[target_idx - 1])
        time_str = candle_time.strftime('%d.%m.%Y') if tf_key == "1d" else candle_time.strftime('%H:%M')

        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)

        tag = "🟢 <b>BIST AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>BIST SAT SİNYALİ</b>"
        trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

        msg = (
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
        signals.append(msg)

    return signals

def main():
    tickers = get_all_bist_tickers()
    toplam = 0
    CHUNK_SIZE = 50
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    print(f"BIST Taraması Başlıyor ({len(tickers)} hisse, {len(chunks)} toplu paket)...")

    for i, chunk in enumerate(chunks, 1):
        try:
            # 10 günlük 15m verisi (Tüm barları eksiksiz üretir)
            data_15m = yf.download(chunk, period="10d", interval="15m", group_by="ticker", threads=True, progress=False)
            data_1d = yf.download(chunk, period="6mo", interval="1d", group_by="ticker", threads=True, progress=False)

            for ticker in chunk:
                try:
                    clean_15m = extract_ticker_df(data_15m, ticker)
                    if not clean_15m.empty:
                        df_1h = build_bist_hourly(clean_15m)
                        if not df_1h.empty and len(df_1h) >= 10:
                            # 1 Saatlik Tarama
                            for msg in evaluate_eco_bist(df_1h, ticker, "1 Saat (1h)", "1h"):
                                send_telegram(msg)
                                toplam += 1
                            # 4 Saatlik Tarama
                            df_4h = make_bist_4h(df_1h)
                            for msg in evaluate_eco_bist(df_4h, ticker, "4 Saat (4h)", "4h"):
                                send_telegram(msg)
                                toplam += 1

                    clean_1d = extract_ticker_df(data_1d, ticker)
                    if not clean_1d.empty and len(clean_1d) >= 10:
                        for msg in evaluate_eco_bist(clean_1d, ticker, "Günlük (1D)", "1d"):
                            send_telegram(msg)
                            toplam += 1

                except Exception as err:
                    pass

        except Exception as e:
            print(f"Grup {i} indirme hatası: {e}")

    print(f"BIST Taraması bitti! Üretilen yeni sinyal sayısı: {toplam}")

if __name__ == "__main__":
    main()
