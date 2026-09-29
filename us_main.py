import os
import sys
import time
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

# TradingView Pivot Timeframe Seçeneği:
# "Auto" : 15m için Günlük (1D), 30m için Haftalık (1W) baz alır (TradingView varsayılanı).
# "Daily": Hem 15m hem 30m için Günlük (1D) baz alır.
PIVOT_ANCHOR_MODE = "Auto"

US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO", "ORCL",
    "CRM",  "AMD",  "QCOM", "INTC",  "CSCO", "IBM",  "TXN",  "AMAT", "MU",   "NOW",
    "ADBE", "PLTR", "UBER", "ABNB",  "COIN", "MSTR", "JPM",  "BAC",  "WFC",  "GS",
    "MS",   "V",    "MA",   "AXP",   "PYPL", "BLK",  "LLY",  "JNJ",  "UNH",  "ABBV",
    "MRK",  "PFE",  "ISRG", "WMT",   "COST", "HD",   "PG",   "KO",   "CAT",  "GE", "DIS"
]

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            print("Telegram bildirimi iletildi.")
            return True
        elif res.status_code == 429:
            retry_after = res.json().get("parameters", {}).get("retry_after", 30)
            print(f"Telegram Flood Uyarısı: {retry_after} saniye beklenmeli!")
            return False
        else:
            print(f"Telegram hatası ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")
        return False

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif len(df.columns.levels) > 1 and 'Close' in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['High', 'Low', 'Close'])
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_convert('Europe/Istanbul')
    else:
        df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
    return df

def wwma(series: pd.Series, length: int) -> pd.Series:
    vals = series.fillna(0.0).values
    res = np.zeros(len(vals))
    for i in range(len(vals)):
        prev = res[i-1] if i > 0 else 0.0
        res[i] = (prev * (length - 1) + vals[i]) / length
    return pd.Series(res, index=series.index)

def calculate_slingshot(df: pd.DataFrame, idx: int = -1):
    if df is None or len(df) < 15:
        return "Belirsiz", "Belirsiz"
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

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
        elif v_ma2 < v_ma3:
            nokta_renk = "🔴 Kırmızı"
        else:
            nokta_renk = "🟡 Sarı"

        return kanal_renk, nokta_renk
    except Exception:
        return "Belirsiz", "Belirsiz"

def calculate_woodie_pivots(df_anchor: pd.DataFrame, label: str = "Günlük") -> dict:
    """TradingView Woodie Pivot Noktaları hesaplaması."""
    try:
        df = clean_df(df_anchor)
        if df.empty or len(df) < 2:
            return {}

        prev_high = float(df['High'].iloc[-2])
        prev_low = float(df['Low'].iloc[-2])
        curr_open = float(df['Open'].iloc[-1])

        p = (prev_high + prev_low + 2.0 * curr_open) / 4.0
        r1 = 2.0 * p - prev_low
        s1 = 2.0 * p - prev_high
        r2 = p + (prev_high - prev_low)
        s2 = p - (prev_high - prev_low)
        r3 = prev_high + 2.0 * (p - prev_low)
        s3 = prev_low - 2.0 * (prev_high - p)

        return {
            "P": p, "R1": r1, "S1": s1,
            "R2": r2, "S2": s2, "R3": r3, "S3": s3,
            "label": label
        }
    except Exception:
        return {}

def detect_diagonal_trendline_and_initiation(df: pd.DataFrame, lookback: int = 45):
    if df is None or len(df) < 20:
        return "Standart Hareket", "Yeni Trend Yok"
    try:
        window = min(lookback, len(df))
        sub_h = df['High'].values[-window:]
        sub_l = df['Low'].values[-window:]
        sub_c = df['Close'].values[-window:]
        curr_c = sub_c[-1]
        curr_h = sub_h[-1]

        peak_indices = [
            i for i in range(2, window - 2)
            if all(sub_h[i] >= sub_h[i-j] for j in range(1, 3)) and all(sub_h[i] >= sub_h[i+j] for j in range(1, 3))
        ]

        dusen_kirilim_durumu = "Düşen Trend Tespit Edilmedi"
        trend_baslatma_durumu = "Yeni Trend Başlamadı"

        if len(peak_indices) >= 2:
            sorted_peaks = sorted(peak_indices, key=lambda idx: sub_h[idx], reverse=True)
            p1 = sorted_peaks[0]
            candidates = [p for p in peak_indices if p > p1 and sub_h[p] < sub_h[p1]]
            if candidates:
                p2 = candidates[0]
                slope = (sub_h[p2] - sub_h[p1]) / (p2 - p1)
                intercept = sub_h[p1] - slope * p1

                line_val_now = slope * (window - 1) + intercept
                line_val_prev = slope * (window - 2) + intercept

                if curr_c > line_val_now:
                    if sub_c[-2] <= line_val_prev or (curr_c - line_val_now) / line_val_now <= 0.04:
                        dusen_kirilim_durumu = f"✅ Düşen Trend Çizgisi Yukarı Kırıldı! (Trend: ${line_val_now:,.2f}) (Boğa)"
                    else:
                        dusen_kirilim_durumu = f"🚀 Düşen Trend Üzerinde Seyrediyor (Trend: ${line_val_now:,.2f}) (Boğa)"
                elif curr_h >= line_val_now and curr_c <= line_val_now:
                    dusen_kirilim_durumu = f"⚠️ Düşen Trend Çizgisi Test Ediliyor (Direnç: ${line_val_now:,.2f}) (Nötr)"
                else:
                    dusen_kirilim_durumu = f"Düşen Trend Altında (Direnç: ${line_val_now:,.2f})"

        lowest_idx = np.argmin(sub_l[:-2])
        lowest_val = sub_l[lowest_idx]

        if lowest_idx < window - 4:
            after_troughs = [
                i for i in range(lowest_idx + 2, window - 1)
                if sub_l[i] <= sub_l[i-1] and sub_l[i] <= sub_l[i+1]
            ]

            if after_troughs:
                sec_idx = after_troughs[-1]
                sec_val = sub_l[sec_idx]

                if sec_val > lowest_val:
                    up_slope = (sec_val - lowest_val) / (sec_idx - lowest_idx)
                    up_line = up_slope * (window - 1) + (lowest_val - up_slope * lowest_idx)

                    if curr_c >= up_line:
                        trend_baslatma_durumu = f"📈 Yeni Yükselen Trend Başlattı! (Dipten Destek: ${up_line:,.2f}, Yükselen Dip Onaylı) (Boğa)"
                    else:
                        trend_baslatma_durumu = "Yükselen Trend Desteği Altında (Nötr)"
                else:
                    trend_baslatma_durumu = "Düşük Dipler Devam Ediyor"
            elif curr_c > lowest_val * 1.03:
                trend_baslatma_durumu = "⚡ Dipten Hızlı Toparlanma (V Dönüş Başlangıcı) (Boğa)"

        return dusen_kirilim_durumu, trend_baslatma_durumu
    except Exception:
        return "Standart Hareket", "Yeni Trend Yok"

def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    if df is None or len(df) < 5:
        return "Standart Mum"
    try:
        sub = df.iloc[-5:]
        o, h, l, c = sub['Open'].values, sub['High'].values, sub['Low'].values, sub['Close'].values
        body = np.abs(c - o)
        cr = np.where((h - l) == 0, 1e-10, h - l)
        is_bull, is_bear = c > o, c < o
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        # 1. ÇOKLU MUM FORMASYONLARI (4-5 MUM)
        if is_bear[1] and is_bear[2] and is_bear[3] and is_bull[4] and c[3] < c[2] < c[1] and o[4] <= c[3] and c[4] >= o[1]:
            return "⚔️ Bullish Three-Line Strike (Yükseliş Dönüş / Boğa) (%84 Başarı)"
        
        if is_bull[1] and is_bull[2] and is_bull[3] and is_bear[4] and c[3] > c[2] > c[1] and o[4] >= c[3] and c[4] <= o[1]:
            return "⚔️ Bearish Three-Line Strike (Düşüş Dönüş / Ayı) (%71 Başarı)"

        if is_bull[0] and body[0]/cr[0] > 0.4 and is_bull[4] and c[4] > h[0] and min(l[1:4]) >= l[0] and max(h[1:4]) <= h[4]:
            return "📈 Rising Three Methods - Yükselen Üç Yöntem (Yükseliş Devam / Boğa) (%78 Başarı)"

        if is_bear[0] and body[0]/cr[0] > 0.4 and is_bear[4] and c[4] < l[0] and max(h[1:4]) <= h[0] and min(l[1:4]) >= l[4]:
            return "📉 Falling Three Methods - Düşen Üç Yöntem (Düşüş Devam / Ayı) (%71 Başarı)"

        # 2. ÜÇLÜ MUM FORMASYONLARI (3 MUM)
        if is_bull[2] and is_bull[3] and is_bull[4] and c[4] > c[3] > c[2] and o[3] > o[2] and o[4] > o[3] and (upper_wick[2:] / cr[2:] < 0.25).all():
            return "🛡 Three White Soldiers - Üç Beyaz Asker (Yükseliş Dönüş / Boğa) (%82 Başarı)"

        if is_bear[2] and is_bear[3] and is_bear[4] and c[4] < c[3] < c[2] and o[3] < o[2] and o[4] < o[3] and (lower_wick[2:] / cr[2:] < 0.25).all():
            return "🦅 Three Black Crows - Üç Kara Karga (Düşüş Dönüş / Ayı) (%79 Başarı)"

        if is_bear[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bull[4] and c[4] > (o[2] + c[2])/2:
            return "⭐ Morning Doji Star - Sabah Yıldızı (Yükseliş Dönüş / Boğa) (%76 Başarı)"

        if is_bull[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bear[4] and c[4] < (o[2] + c[2])/2:
            return "🌙 Evening Doji Star - Akşam Yıldızı (Düşüş Dönüş / Ayı) (%72 Başarı)"

        # 3. İKİLİ MUM FORMASYONLARI (2 MUM)
        if is_bull[3] and is_bear[4] and o[4] >= c[3] and c[4] <= o[3] and body[4]/cr[4] > 0.4:
            return "🔴 Bearish Engulfing - Yutan Ayı (Düşüş Dönüş / Ayı) (%79 Başarı)"

        if is_bear[3] and is_bull[4] and o[4] <= c[3] and c[4] >= o[3] and body[4]/cr[4] > 0.4:
            return "🟢 Bullish Engulfing - Yutan Boğa (Yükseliş Dönüş / Boğa) (%63 Başarı)"

        if is_bear[3] and is_bull[4] and body[3]/cr[3] > 0.35 and o[4] <= c[3] and c[4] > (o[3] + c[3])/2 and c[4] < o[3]:
            return "⚡ Piercing Line - Delen Mum (Yükseliş Dönüş / Boğa) (%64 Başarı)"

        if is_bull[3] and is_bear[4] and body[3]/cr[3] > 0.35 and o[4] >= c[3] and c[4] < (o[3] + c[3])/2 and c[4] > o[3]:
            return "☁️ Dark Cloud Cover - Kara Bulut (Düşüş Dönüş / Ayı) (%60 Başarı)"

        if is_bear[3] and is_bull[4] and abs(l[4] - l[3])/cr[4] < 0.06 and lower_wick[3]/cr[3] > 0.25 and lower_wick[4]/cr[4] > 0.25:
            return "🧲 Tweezer Bottom - Cımbız Dip (Yükseliş Dönüş / Boğa) (%60 Başarı)"

        if is_bull[3] and is_bear[4] and abs(h[4] - h[3])/cr[4] < 0.06 and upper_wick[3]/cr[3] > 0.25 and upper_wick[4]/cr[4] > 0.25:
            return "🧲 Tweezer Top - Cımbız Tepe (Düşüş Dönüş / Ayı) (%60 Başarı)"

        # 4. TEK MUM FORMASYONLARI (1 MUM)
        if body[4]/cr[4] < 0.1 and upper_wick[4]/cr[4] > 0.65 and lower_wick[4]/cr[4] < 0.1:
            return "🪦 Gravestone Doji - Mezar Taşı Doji (Düşüş Dönüş / Ayı) (%66 Başarı)"

        if body[4]/cr[4] < 0.1 and lower_wick[4]/cr[4] > 0.65 and upper_wick[4]/cr[4] < 0.1:
            return "🦗 Dragonfly Doji - Yusufçuk Doji (Yükseliş Dönüş / Boğa) (%65 Başarı)"

        if is_bull[4] and upper_wick[4] >= 2.0 * body[4] and lower_wick[4]/cr[4] <= 0.15 and 0.1 <= body[4]/cr[4] <= 0.35 and c[3] <= c[2]:
            return "🔨 Inverted Hammer - Ters Çekiç (Yükseliş Dönüş / Boğa) (%65 Başarı)"

        if lower_wick[4] >= 2.0 * body[4] and upper_wick[4]/cr[4] <= 0.15 and 0.1 <= body[4]/cr[4] <= 0.35:
            return "🔨 Hammer - Çekiç (Yükseliş Dönüş / Boğa) (%60 Başarı)"

        if upper_wick[4] >= 2.0 * body[4] and lower_wick[4]/cr[4] <= 0.15 and 0.1 <= body[4]/cr[4] <= 0.35 and c[3] >= c[2]:
            return "💫 Shooting Star - Kayan Yıldız (Düşüş Dönüş / Ayı) (%59 Başarı)"

        return "Standart Mum"
    except Exception:
        return "Standart Mum"

def detect_ict_smc_models(df: pd.DataFrame) -> str:
    if df is None or len(df) < 25:
        return "Belirsiz"
    try:
        high, low, close, opens = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
        curr_c, curr_h, curr_l = close[-1], high[-1], low[-1]

        swing_low = np.min(low[-16:-2])
        if (low[-2] < swing_low and close[-2] > swing_low) or (curr_l < swing_low and curr_c > swing_low):
            return "🧲 Liquidity Sweep / Turtle Soup (Tuzak Dip Temizlendi) (Boğa)"

        swing_high = np.max(high[-16:-2])
        if (high[-2] > swing_high and close[-2] < swing_high) or (curr_h > swing_high and curr_c < swing_high):
            return "🧲 Buy-Side Liquidity Sweep (Tepe Likiditesi Alındı) (Ayı)"

        if low[-2] > high[-4] and (high[-4] <= curr_l <= low[-2] or high[-4] <= curr_c <= low[-2]):
            return "⚡ Bullish FVG (Dengesizlik Boşluğu Test Ediliyor) (Boğa)"
        if high[-2] < low[-4] and (high[-2] <= curr_h <= low[-4] or high[-2] <= curr_c <= low[-4]):
            return "⚡ Bearish FVG (Satış Dengesizliği Test Ediliyor) (Ayı)"

        for i in range(2, 6):
            if close[-i] > opens[-i] and (close[-i] - opens[-i]) > np.std(np.abs(close - opens)) * 1.5:
                ob_idx = -(i + 1)
                if low[ob_idx] <= curr_l <= high[ob_idx] or low[ob_idx] <= curr_c <= high[ob_idx]:
                    return "🧱 Bullish Order Block (OB Kurumsal Alım Bölgesi) (Boğa)"

        return "Normal Fiyat Yapısı"
    except Exception:
        return "Normal Fiyat Yapısı"

def calculate_score_and_rvol(df: pd.DataFrame, idx: int, sig_type: str, ss_multi: dict, candle_pat: str, dusen_trend: str, yeni_trend: str):
    try:
        vol = df['Volume'].squeeze()
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]

        window = 20
        past_vols = vol.iloc[-window-1:-1] if len(vol) > window + 1 else vol.iloc[:-1]
        valid_past = past_vols[past_vols > 0]
        avg_vol = float(valid_past.mean()) if len(valid_past) > 0 else float(vol[vol > 0].mean()) if len(vol[vol > 0]) > 0 else 1.0

        raw_curr_vol = float(vol.iloc[idx]) if len(vol) > 0 else 0.0
        if (raw_curr_vol <= 0 or np.isnan(raw_curr_vol)) and len(vol) >= 2:
            eval_vol = float(vol.iloc[-2])
        else:
            eval_vol = raw_curr_vol

        if eval_vol > 0 and idx == -1 and raw_curr_vol > 0:
            candle_time = df.index[-1]
            now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
            elapsed_sec = (now_tsi - candle_time).total_seconds()
            tf_sec = max(60.0, (df.index[-1] - df.index[-2]).total_seconds()) if len(df.index) >= 2 else 900.0
            ratio = min(max(elapsed_sec / tf_sec, 0.1), 1.0)
            eval_vol = min(eval_vol / ratio, eval_vol * 4.0)

        rvol = eval_vol / avg_vol if avg_vol > 0 else 1.0

        if rvol >= 1.5:
            hacim_metni = f"🚀 Çok Güçlü (Ortalamanın {rvol:.1f}x Katı)"
        elif rvol >= 1.1:
            hacim_metni = f"🟢 Güçlü (Ortalamanın {rvol:.1f}x Katı)"
        elif rvol >= 0.8:
            hacim_metni = f"⚪ Normal (Ortalamanın {rvol:.1f}x Katı)"
        else:
            hacim_metni = f"⚠️ Zayıf (Ortalamanın {rvol:.1f}x Katı)"

        puan = 1
        k1h, _ = ss_multi.get("1h", ("", ""))
        k15, _ = ss_multi.get("15m", ("", ""))
        if (sig_type == "BUY" and "Yeşil" in k1h) or (sig_type == "SELL" and "Kırmızı" in k1h):
            puan += 1
        if (sig_type == "BUY" and "Yeşil" in k15) or (sig_type == "SELL" and "Kırmızı" in k15):
            puan += 1
        if rvol >= 1.1:
            puan += 1
        if sig_type == "BUY":
            if ("Boğa" in candle_pat) or ("Kırıldı" in dusen_trend) or ("Yeni Yükselen" in yeni_trend):
                puan += 1
        else:
            if ("Ayı" in candle_pat) or ("Aşağı Kırıldı" in dusen_trend) or ("Düşük Dipler" in yeni_trend):
                puan += 1

        puan = min(puan, 5)
        skor_metni = f"{'⭐' * puan} ({puan}/5)"
        return hacim_metni, skor_metni
    except Exception:
        return "⚪ Normal (Ortalamanın 1.0x Katı)", "⭐⭐⭐ (3/5)"

def evaluate_eco(df: pd.DataFrame, symbol: str, tf_label: str, ss_multi: dict, pivots: dict = None):
    df = clean_df(df)
    if df.empty or len(df) < 15:
        return None

    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    hiDiff = high - high.shift(1)
    loDiff = low.shift(1) - low
    plusDM = pd.Series(np.where((hiDiff > loDiff) & (hiDiff > 0), hiDiff, 0.0), index=df.index)
    minusDM = pd.Series(np.where((loDiff > hiDiff) & (loDiff > 0), loDiff, 0.0), index=df.index)

    ATR = wwma(tr, 10)
    PlusDI = 100 * wwma(plusDM, 10) / ATR.replace(0, 1e-10)
    MinusDI = 100 * wwma(minusDM, 10) / ATR.replace(0, 1e-10)
    osc = PlusDI - MinusDI

    hi = osc.rolling(window=3).max()
    lo = osc.rolling(window=3).min()
    stoch = (((osc - lo).rolling(window=3).sum() / (hi - lo).rolling(window=3).sum().replace(0, 1e-10)) * 100).clip(0, 100).ffill().fillna(50.0)

    c_prev = float(stoch.iloc[-2])
    c_curr = float(stoch.iloc[-1])

    if c_prev < 10 and c_curr > 10:
        sig_type = "BUY"
    elif c_prev > 90 and c_curr < 90:
        sig_type = "SELL"
    else:
        return None

    target_idx = -1
    candle_time = df.index[target_idx]

    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    if candle_time.date() != now_tsi.date():
        return None

    age_minutes = (now_tsi - candle_time).total_seconds() / 60.0
    if age_minutes > 120:
        return None

    candle_pat = detect_candlestick_patterns(df)
    
    # Mum formasyonu oluşmamışsa (Standart Mum ise) telegrama bildirim iletme
    if candle_pat == "Standart Mum":
        return None

    candle_price = float(close.iloc[target_idx])
    time_str = candle_time.strftime('%H:%M')
    tv_link = f"https://tr.tradingview.com/chart/?symbol={symbol}"

    dusen_trend, yeni_trend = detect_diagonal_trendline_and_initiation(df)
    smc_model = detect_ict_smc_models(df)

    hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, sig_type, ss_multi, candle_pat, dusen_trend, yeni_trend)

    ss_15m_k, ss_15m_n = ss_multi.get("15m", ("Belirsiz", "Belirsiz"))
    ss_1h_k, ss_1h_n = ss_multi.get("1h", ("Belirsiz", "Belirsiz"))
    ss_4h_k, ss_4h_n = ss_multi.get("4h", ("Belirsiz", "Belirsiz"))

    pivot_metni = ""
    if pivots and "P" in pivots:
        p = pivots.get("P", 0.0)
        p_label = pivots.get("label", "Günlük")
        r1 = pivots.get("R1", 0.0)
        r2 = pivots.get("R2", 0.0)
        r3 = pivots.get("R3", 0.0)
        s1 = pivots.get("S1", 0.0)
        s2 = pivots.get("S2", 0.0)
        s3 = pivots.get("S3", 0.0)
        dist_p = ((candle_price - p) / p) * 100 if p > 0 else 0.0
        durum = "🟢 Pivot Üzerinde (Boğa)" if candle_price >= p else "🔴 Pivot Altında (Ayı)"
        pivot_metni = (
            f"\n\n<b>🎯 Pivot Noktaları Standart (Woodie - {p_label}):</b>\n"
            f"▫️ <b>Konum:</b> {durum} (<code>{dist_p:+.2f}%</code>)\n"
            f"▫️ <b>Pivot (P):</b> ${p:,.2f}\n"
            f"▫️ <b>Dirençler:</b> R1: ${r1:,.2f} | R2: ${r2:,.2f} | R3: ${r3:,.2f}\n"
            f"▫️️ <b>Destekler:</b> S1: ${s1:,.2f} | S2: ${s2:,.2f} | S3: ${s3:,.2f}"
        )

    tag = "🟢 <b>ABD BORSASI AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>ABD BORSASI SAT SİNYALİ</b>"

    return (
        f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
        f"📌 <b>Hisse:</b> <a href='{tv_link}'>#{symbol}</a> <i>(Grafiği Aç)</i>\n"
        f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
        f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
        f"💵 <b>Fiyat:</b> ${candle_price:,.2f}\n\n"
        f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
        f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
        f"<b>🕯️ Formasyon & Trend Teyitleri:</b>\n"
        f"▫️ <b>Düşen Trend Kırılımı:</b> {dusen_trend}\n"
        f"▫️️ <b>Trend Başlatma Durumu:</b> {yeni_trend}\n"
        f"▫️ <b>Mum Formasyonu:</b> {candle_pat}\n"
        f"▫️ <b>ICT / SMC Modeli:</b> {smc_model}\n\n"
        f"<b>📈 Trend Teyitleri (SlingShot Multi-TF):</b>\n"
        f"▫️ <b>15 Dakika (15m):</b> {ss_15m_k} Kanal | {ss_15m_n} Nokta\n"
        f"▫️ <b>1 Saat (1h):</b> {ss_1h_k} Kanal | {ss_1h_n} Nokta\n"
        f"▫️ <b>4 Saat (4h):</b> {ss_4h_k} Kanal | {ss_4h_n} Nokta"
        f"{pivot_metni}"
    )

def scan_ticker(symbol: str):
    signals = []
    try:
        df_15m = yf.download(symbol, period="5d", interval="15m", progress=False)
        clean_15m = clean_df(df_15m)
        df_1h = yf.download(symbol, period="2mo", interval="1h", progress=False)
        clean_1h = clean_df(df_1h)
        df_30m = yf.download(symbol, period="1mo", interval="30m", progress=False)
        clean_30m = clean_df(df_30m)

        df_4h = clean_1h.resample("4h").agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        # Günlük ve Haftalık Veriler (Pivot Hesaplaması İçin)
        df_daily = yf.download(symbol, period="1mo", interval="1d", progress=False)
        clean_daily = clean_df(df_daily)
        if clean_daily.empty or len(clean_daily) < 2:
            clean_daily = clean_1h.resample('1D').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

        df_weekly = yf.download(symbol, period="3mo", interval="1wk", progress=False)
        clean_weekly = clean_df(df_weekly)
        if clean_weekly.empty or len(clean_weekly) < 2:
            clean_weekly = clean_1h.resample('1W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

        ss_multi = {
            "15m": calculate_slingshot(clean_15m, -1),
            "1h": calculate_slingshot(clean_1h, -1),
            "4h": calculate_slingshot(df_4h, -1)
        }

        # TradingView Pivot Points Standard Mantığı:
        pivots_daily = calculate_woodie_pivots(clean_daily, label="Günlük")
        pivots_weekly = calculate_woodie_pivots(clean_weekly, label="Haftalık")

        pivots_15m = pivots_daily
        pivots_30m = pivots_weekly if PIVOT_ANCHOR_MODE == "Auto" else pivots_daily

        s15m = evaluate_eco(clean_15m, symbol, "15 Dakika (15m)", ss_multi, pivots=pivots_15m)
        if s15m: 
            signals.append(s15m)

        s30m = evaluate_eco(clean_30m, symbol, "30 Dakika (30m)", ss_multi, pivots=pivots_30m)
        if s30m: 
            signals.append(s30m)

    except Exception as e:
        print(f"{symbol} analiz hatası: {e}")
    return signals

def main():
    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    # ABD Seansı (16:30 - 23:00 TSİ) kontrolü
    if now_tsi.hour < 16 or (now_tsi.hour == 16 and now_tsi.minute < 20) or now_tsi.hour >= 23:
        print(f"ABD Seansı Kapalı (Saat: {now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    all_signals = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_ticker, ticker): ticker for ticker in US_TICKERS}
        for future in as_completed(futures):
            try:
                results = future.result()
                if results: 
                    all_signals.extend(results)
            except Exception as e:
                print(f"Hisse analiz hatası: {e}")

    for i, sig in enumerate(all_signals):
        send_telegram(sig)
        if i < len(all_signals) - 1:
            time.sleep(1.5)

if __name__ == "__main__":
    main()
