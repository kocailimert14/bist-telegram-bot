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

def calculate_woodie_pivot_htf(df_htf: pd.DataFrame, curr_price: float):
    if df_htf is None or len(df_htf) < 2:
        return None
    try:
        prev_h = float(df_htf['High'].iloc[-2])
        prev_l = float(df_htf['Low'].iloc[-2])
        curr_o = float(df_htf['Open'].iloc[-1])

        p = (prev_h + prev_l + 2.0 * curr_o) / 4.0
        r1 = 2.0 * p - prev_l
        s1 = 2.0 * p - prev_h
        r2 = p + (prev_h - prev_l)
        s2 = p - (prev_h - prev_l)
        r3 = prev_h + 2.0 * (p - prev_l)
        s3 = prev_l - 2.0 * (prev_h - p)

        return {
            "P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3,
            "dist_P": ((p - curr_price) / curr_price) * 100.0
        }
    except Exception:
        return None

def calculate_slingshot(df: pd.DataFrame, idx: int = -1):
    if df is None or len(df) < 15:
        return "Belirsiz", "Belirsiz"
    try:
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
        elif (v_ma1 < v_lower if False else v_ma2 < v_ma3):
            nokta_renk = "🔴 Kırmızı"
        else:
            nokta_renk = "🟡 Sarı"

        return kanal_renk, nokta_renk
    except Exception:
        return "Belirsiz", "Belirsiz"

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

        peak_indices = []
        k = 2
        for i in range(k, window - k):
            if all(sub_h[i] >= sub_h[i-j] for j in range(1, k+1)) and all(sub_h[i] >= sub_h[i+j] for j in range(1, k+1)):
                peak_indices.append(i)

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
            after_troughs = []
            for i in range(lowest_idx + 2, window - 1):
                if sub_l[i] <= sub_l[i-1] and sub_l[i] <= sub_l[i+1]:
                    after_troughs.append(i)

            if after_troughs:
                second_low_idx = after_troughs[-1]
                second_low_val = sub_l[second_low_idx]

                if second_low_val > lowest_val:
                    up_slope = (second_low_val - lowest_val) / (second_low_idx - lowest_idx)
                    up_intercept = lowest_val - up_slope * lowest_idx
                    up_line_now = up_slope * (window - 1) + up_intercept

                    if curr_c >= up_line_now:
                        trend_baslatma_durumu = f"📈 Yeni Yükselen Trend Başlattı! (Dipten Destek: ${up_line_now:,.2f}, Yükselen Dip Onaylı) (Boğa)"
                    else:
                        trend_baslatma_durumu = "Yükselen Trend Desteği Altında (Nötr)"
                else:
                    trend_baslatma_durumu = "Düşük Dipler Devam Ediyor"
            else:
                if curr_c > lowest_val * 1.03:
                    trend_baslatma_durumu = "⚡ Dipten Hızlı Toparlanma (V Dönüş Başlangıcı) (Boğa)"

        return dusen_kirilim_durumu, trend_baslatma_durumu
    except Exception:
        return "Standart Hareket", "Yeni Trend Yok"

def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    if df is None or len(df) < 5:
        return "Standart Mum"
    try:
        sub = df.iloc[-5:]
        o = sub['Open'].values
        h = sub['High'].values
        l = sub['Low'].values
        c = sub['Close'].values
        body = np.abs(c - o)
        candle_range = h - l
        is_bull = c > o
        is_bear = c < o
        eps = 1e-10
        cr = np.where(candle_range == 0, eps, candle_range)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        if (is_bear[1] and is_bear[2] and is_bear[3] and is_bull[4] and 
            c[3] < c[2] < c[1] and o[4] <= c[3] and c[4] >= o[1]):
            return "⚔️ Bullish Three-Line Strike (Yükseliş Dönüş / Boğa) (%84 Başarı)"

        if (is_bull[2] and is_bull[3] and is_bull[4] and 
            c[4] > c[3] > c[2] and o[3] > o[2] and o[4] > o[3] and
            upper_wick[2]/cr[2] < 0.25 and upper_wick[3]/cr[3] < 0.25 and upper_wick[4]/cr[4] < 0.25):
            return "🛡️ Three White Soldiers - Üç Beyaz Asker (Yükseliş Dönüş / Boğa) (%82 Başarı)"

        if (is_bull[0] and body[0]/cr[0] > 0.4 and is_bull[4] and c[4] > h[0] and min(l[1], l[2], l[3]) >= l[0]):
            return "📈 Rising Three Methods - Yükselen Üç Yöntem (Yükseliş Devam / Boğa) (%78 Başarı)"

        if (is_bear[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bull[4] and c[4] > (o[2] + c[2])/2):
            return "⭐ Morning Doji Star - Sabah Yıldızı (Yükseliş Dönüş / Boğa) (%76 Başarı)"

        if (is_bear[2] and body[3]/cr[3] < 0.15 and is_bull[4] and h[3] < l[2] and l[4] > h[3] and c[4] > (o[2] + c[2])/2):
            return "👶 Bullish Abandoned Baby - Terk Edilmiş Bebek (Yükseliş Dönüş / Boğa) (%75 Başarı)"

        if (is_bear[2] and is_bull[3] and o[3] <= c[2] and c[3] >= o[2] and is_bull[4] and c[4] > c[3]):
            return "🟢 Three Outside Up (Yükseliş Dönüş / Boğa) (%72 Başarı)"

        if (is_bear[3] and is_bull[4] and (body[3]/cr[3] > 0.35) and (body[4]/cr[4] > 0.35) and 
            abs(o[4] - o[3]) / o[3] <= 0.003 and c[4] > h[3]):
            return "⚡ Bullish Separating Lines - Ayrılan Çizgiler (Yükseliş Devam / Boğa) (%68 Başarı)"

        if (is_bear[2] and is_bear[3] and is_bear[4] and 
            c[4] < c[3] < c[2] and o[3] < o[2] and o[4] < o[3] and
            lower_wick[2]/cr[2] < 0.25 and lower_wick[3]/cr[3] < 0.25 and lower_wick[4]/cr[4] < 0.25):
            return "🦅 Three Black Crows - Üç Kara Karga (Düşüş Dönüş / Ayı) (%79 Başarı)"

        if (is_bull[2] and body[3]/cr[3] < 0.15 and is_bear[4] and l[3] > h[2] and h[4] < l[3] and c[4] < (o[2] + c[2])/2):
            return "👶 Bearish Abandoned Baby - Terk Edilmiş Bebek (Düşüş Dönüş / Ayı) (%75 Başarı)"

        if (is_bull[2] and body[2]/cr[2] > 0.35 and body[3]/cr[3] < 0.3 and is_bear[4] and c[4] < (o[2] + c[2])/2):
            return "🌙 Evening Doji Star - Akşam Yıldızı (Düşüş Dönüş / Ayı) (%72 Başarı)"

        if (is_bull[2] and is_bear[3] and o[3] >= c[2] and c[3] <= o[2] and is_bear[4] and c[4] < c[3]):
            return "🔴 Three Outside Down (Düşüş Dönüş / Ayı) (%72 Başarı)"

        if (is_bear[0] and body[0]/cr[0] > 0.4 and is_bear[4] and c[4] < l[0] and max(h[1], h[2], h[3]) <= h[0]):
            return "📉 Falling Three Methods - Düşen Üç Yöntem (Düşüş Devam / Ayı) (%71 Başarı)"

        if (is_bear[3] and is_bear[4] and h[3] < l[2] and c[4] < c[3] and o[4] <= o[3]):
            return "🕳️ Two Black Gapping - Boşluklu İki Kırmızı (Düşüş Devam / Ayı) (%68 Başarı)"

        if (is_bull[1] and is_bull[2] and is_bull[3] and is_bear[4] and 
            c[3] > c[2] > c[1] and o[4] >= c[3] and c[4] <= o[1]):
            return "⚔️ Bearish Three-Line Strike (Düşüş Dönüş / Ayı) (%65 Başarı)"

        return "Standart Mum"
    except Exception:
        return "Standart Mum"

def detect_harmonic_patterns(df: pd.DataFrame, deviation_pct: float = 1.8) -> str:
    if df is None or len(df) < 30:
        return "Harmonik Formasyon Yok"
    try:
        highs, lows, closes = df['High'].values, df['Low'].values, df['Close'].values
        trend, last_h, last_l, idx_h, idx_l = 0, highs[0], lows[0], 0, 0
        pivots = []
        for i in range(1, len(df)):
            h, l = highs[i], lows[i]
            if trend <= 0 and h >= last_l * (1 + deviation_pct / 100):
                pivots.append((idx_l, 'L', last_l))
                trend, last_h, idx_h = 1, h, i
            elif trend >= 0 and l <= last_h * (1 - deviation_pct / 100):
                pivots.append((idx_l, 'H', last_h))
                trend, last_l, idx_l = -1, l, i
            else:
                if h > last_h: last_h, idx_h = h, i
                if l < last_l: last_l, idx_l = l, i
        pivots.append((idx_h, 'H', last_h) if trend == 1 else (idx_l, 'L', last_l))

        PATTERNS = [
            ("Bat (Yarasa)", 0.35, 0.55, 0.886, 0.08, "🦇", 85),
            ("Gartley 222", 0.58, 0.65, 0.786, 0.08, "📐", 75),
            ("Butterfly (Kelebek)", 0.72, 0.82, 1.272, 0.15, "🦋", 78),
            ("Crab (Yengeç)", 0.35, 0.65, 1.618, 0.15, "🦀", 80),
            ("Shark (Köpekbalığı)", 0.40, 0.90, 0.886, 0.12, "🦈", 75)
        ]

        def eval_xabcd(pX, pA, pB, pC, d_price):
            is_bull = (pX[1] == 'L' and pA[1] == 'H' and pB[1] == 'L' and pC[1] == 'H')
            is_bear = (pX[1] == 'H' and pA[1] == 'L' and pB[1] == 'H' and pC[1] == 'L')
            if not (is_bull or is_bear): return None
            xa = abs(pA[2] - pX[2])
            if xa == 0: return None
            b_ratio = abs(pA[2] - pB[2]) / xa
            d_ratio = (pA[2] - d_price) / xa if is_bull else (d_price - pA[2]) / xa
            direction = "Yükseliş / Boğa" if is_bull else "Düşüş / Ayı"
            prefix = "Bullish" if is_bull else "Bearish"

            for name, b_min, b_max, d_target, d_tol, icon, rate in PATTERNS:
                if (b_min <= b_ratio <= b_max) and abs(d_ratio - d_target) <= d_tol:
                    return f"{icon} {prefix} {name} ({direction}) (%{rate} Başarı)"
            return None

        if len(pivots) >= 5 and (len(df) - 1 - pivots[-1][0]) <= 6:
            res = eval_xabcd(pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1][2])
            if res: return res
        if len(pivots) >= 4:
            res = eval_xabcd(pivots[-4], pivots[-3], pivots[-2], pivots[-1], closes[-1])
            if res: return res
        return "Harmonik Formasyon Yok"
    except Exception:
        return "Harmonik Formasyon Yok"

def detect_ict_smc_models(df: pd.DataFrame) -> str:
    if df is None or len(df) < 25:
        return "Belirsiz"
    try:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        opens = df['Open'].values
        curr_c = close[-1]
        curr_h = high[-1]
        curr_l = low[-1]

        swing_low = np.min(low[-16:-2])
        if (low[-2] < swing_low and close[-2] > swing_low) or (curr_l < swing_low and curr_c > swing_low):
            return "🧲 Liquidity Sweep / Turtle Soup (Tuzak Dip Temizlendi) (Boğa)"

        swing_high = np.max(high[-16:-2])
        if (high[-2] > swing_high and close[-2] < swing_high) or (curr_h > swing_high and curr_c < swing_high):
            return "🧲 Buy-Side Liquidity Sweep (Tepe Likiditesi Alındı) (Ayı)"

        if low[-2] > high[-4]:
            if high[-4] <= curr_l <= low[-2] or high[-4] <= curr_c <= low[-2]:
                return "⚡ Bullish FVG (Dengesizlik Boşluğu Test Ediliyor) (Boğa)"
        if high[-2] < low[-4]:
            if high[-2] <= curr_h <= low[-4] or high[-2] <= curr_c <= low[-4]:
                return "⚡ Bearish FVG (Satış Dengesizliği Test Ediliyor) (Ayı)"

        for i in range(2, 6):
            if close[-i] > opens[-i] and (close[-i] - opens[-i]) > np.std(np.abs(close - opens)) * 1.5:
                ob_idx = -(i + 1)
                if low[ob_idx] <= curr_l <= high[ob_idx] or low[ob_idx] <= curr_c <= high[ob_idx]:
                    return "🧱 Bullish Order Block (OB Kurumsal Alım Bölgesi) (Boğa)"

        recent_min = np.min(low[-20:])
        recent_max = np.max(high[-20:])
        rng = recent_max - recent_min
        if rng > 0:
            fib_62 = recent_max - rng * 0.62
            fib_79 = recent_max - rng * 0.79
            if fib_79 <= curr_c <= fib_62:
                return "🎯 OTE (%62 - %79 Optimal Giriş Bölgesi) (Boğa)"

        return "Normal Fiyat Yapısı"
    except Exception:
        return "Normal Fiyat Yapısı"

def calculate_strong_sr(df: pd.DataFrame, idx: int = -1, lookback: int = 60, min_dist_pct: float = 0.6):
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        curr_price = float(close.iloc[idx])
        window = min(lookback, len(df))
        sub_high = high.iloc[-window:].values
        sub_low = low.iloc[-window:].values
        res_peaks = []
        sup_troughs = []
        k = 3
        for i in range(k, len(sub_high) - k):
            if all(sub_high[i] >= sub_high[i-j] for j in range(1, k+1)) and all(sub_high[i] >= sub_high[i+j] for j in range(1, k+1)):
                dist = ((sub_high[i] - curr_price) / curr_price) * 100
                if dist >= min_dist_pct:
                    res_peaks.append(sub_high[i])
            if all(sub_low[i] <= sub_low[i-j] for j in range(1, k+1)) and all(sub_low[i] <= sub_low[i+j] for j in range(1, k+1)):
                dist = ((curr_price - sub_low[i]) / curr_price) * 100
                if dist >= min_dist_pct:
                    sup_troughs.append(sub_low[i])

        if res_peaks:
            resistance = float(min(res_peaks))
        else:
            resistance = float(np.max(sub_high))
            if ((resistance - curr_price) / curr_price) * 100 < min_dist_pct:
                resistance = curr_price * (1 + min_dist_pct / 100)

        if sup_troughs:
            support = float(max(sup_troughs))
        else:
            support = float(np.min(sub_low))
            if ((curr_price - support) / curr_price) * 100 < min_dist_pct:
                support = curr_price * (1 - min_dist_pct / 100)

        dist_sup = ((support - curr_price) / curr_price) * 100
        dist_res = ((resistance - curr_price) / curr_price) * 100
        return support, resistance, dist_sup, dist_res
    except Exception:
        return None, None, 0.0, 0.0

def calculate_score_and_rvol(df: pd.DataFrame, idx: int, sig_type: str, ss_multi: dict, d_sup: float, d_res: float, candle_pat: str, dusen_trend: str, yeni_trend: str, harmonic_pat: str = "Harmonik Formasyon Yok", woodie: dict = None):
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
            if len(df.index) >= 2:
                tf_sec = max(60.0, (df.index[-1] - df.index[-2]).total_seconds())
            else:
                tf_sec = 900.0

            ratio = min(max(elapsed_sec / tf_sec, 0.1), 1.0)
            projected_vol = eval_vol / ratio
            eval_vol = min(projected_vol, eval_vol * 4.0)

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
            if ("Boğa" in candle_pat) or ("Kırıldı" in dusen_trend) or ("Yeni Yükselen" in yeni_trend) or ("Boğa" in harmonic_pat):
                puan += 1
        else:
            if ("Ayı" in candle_pat) or ("Aşağı Kırıldı" in dusen_trend) or ("Düşük Dipler" in yeni_trend) or ("Ayı" in harmonic_pat):
                puan += 1

        curr_price = float(df['Close'].iloc[idx])
        if woodie:
            if sig_type == "BUY" and curr_price >= woodie["P"]:
                puan += 1
            elif sig_type == "SELL" and curr_price <= woodie["P"]:
                puan += 1

        puan = min(puan, 5)
        yildizlar = "⭐" * puan
        skor_metni = f"{yildizlar} ({puan}/5)"
        return hacim_metni, skor_metni
    except Exception:
        return "⚪ Normal (Ortalamanın 1.0x Katı)", "⭐⭐⭐ (3/5)"

def evaluate_eco(df: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, tf_label: str, pivot_label: str, ss_multi: dict):
    if df is None or df.empty or len(df) < 15:
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

    c_prev = float(stoch.iloc[-2])
    c_curr = float(stoch.iloc[-1])

    sig_type = None
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

    candle_price = float(close.iloc[target_idx])
    time_str = candle_time.strftime('%H:%M')
    tv_link = f"https://tr.tradingview.com/chart/?symbol={symbol}"

    dusen_trend, yeni_trend = detect_diagonal_trendline_and_initiation(df)
    candle_pat = detect_candlestick_patterns(df)
    harmonic_pat = detect_harmonic_patterns(df)
    smc_model = detect_ict_smc_models(df)
    
    woodie = calculate_woodie_pivot_htf(df_htf, candle_price)

    sup, res, d_sup, d_res = calculate_strong_sr(df, target_idx)
    hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, sig_type, ss_multi, d_sup, d_res, candle_pat, dusen_trend, yeni_trend, harmonic_pat, woodie)

    ss_15m_k, ss_15m_n = ss_multi.get("15m", ("Belirsiz", "Belirsiz"))
    ss_1h_k, ss_1h_n = ss_multi.get("1h", ("Belirsiz", "Belirsiz"))
    ss_4h_k, ss_4h_n = ss_multi.get("4h", ("Belirsiz", "Belirsiz"))

    sr_metni = ""
    if sup is not None and res is not None:
        sr_metni = (
            f"\n\n<b>🎯 Kuvvetli Destek & Direnç:</b>\n"
            f"▫️ <b>Ana Destek:</b> ${sup:,.2f} (<code>{d_sup:+.1f}%</code>)\n"
            f"▫️ <b>Ana Direnç:</b> ${res:,.2f} (<code>{d_res:+.1f}%</code>)"
        )

    woodie_metni = ""
    if woodie:
        p_durum = "Üzerinde (Boğa)" if candle_price >= woodie["P"] else "Altında (Ayı)"
        woodie_metni = (
            f"\n\n<b>📐 Woodie Pivot Seviyeleri ({pivot_label}):</b>\n"
            f"▫️ <b>Pivot (P):</b> ${woodie['P']:,.2f} (<code>{woodie['dist_P']:+.1f}%</code>) - {p_durum}\n"
            f"▫️ <b>Dirençler:</b> R1: ${woodie['R1']:,.2f} | R2: ${woodie['R2']:,.2f}\n"
            f"▫️ <b>Destekler:</b> S1: ${woodie['S1']:,.2f} | S2: ${woodie['S2']:,.2f}"
        )

    tag = "🟢 <b>ABD BORSASI AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>ABD BORSASI SAT SİNYALİ</b>"

    return (
        f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
        f"📌 <b>Hisse:</b> <a href=\"{tv_link}\">#{symbol}</a> <i>(Grafiği Aç)</i>\n"
        f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
        f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
        f"💵 <b>Fiyat:</b> ${candle_price:,.2f}\n\n"
        f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
        f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
        f"<b>🕯️ Formasyon & Trend Teyitleri:</b>\n"
        f"▫️ <b>Düşen Trend Kırılımı:</b> {dusen_trend}\n"
        f"▫️ <b>Trend Başlatma Durumu:</b> {yeni_trend}\n"
        f"▫️ <b>Mum Formasyonu:</b> {candle_pat}\n"
        f"▫️ <b>Harmonik Formasyon:</b> {harmonic_pat}\n"
        f"▫️ <b>ICT / SMC Modeli:</b> {smc_model}\n\n"
        f"<b>📈 Trend Teyitleri (SlingShot Multi-TF):</b>\n"
        f"▫️ <b>15 Dakika (15m):</b> {ss_15m_k} Kanal | {ss_15m_n} Nokta\n"
        f"▫️ <b>1 Saat (1h):</b> {ss_1h_k} Kanal | {ss_1h_n} Nokta\n"
        f"▫️ <b>4 Saat (4h):</b> {ss_4h_k} Kanal | {ss_4h_n} Nokta"
        f"{sr_metni}"
        f"{woodie_metni}"
    )

def scan_ticker(symbol: str):
    signals = []
    try:
        df_15m = clean_df(yf.download(symbol, period="5d", interval="15m", progress=False))
        df_30m = clean_df(yf.download(symbol, period="1mo", interval="30m", progress=False))
        df_1h = clean_df(yf.download(symbol, period="2mo", interval="1h", progress=False))
        
        # TradingView Pivot kaynakları (15m için Günlük 1d, 30m için Haftalık 1wk)
        df_1d = clean_df(yf.download(symbol, period="1mo", interval="1d", progress=False))
        df_1w = clean_df(yf.download(symbol, period="3mo", interval="1wk", progress=False))

        df_4h = df_1h.resample("4h").agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        ss_multi = {
            "15m": calculate_slingshot(df_15m, -1),
            "1h": calculate_slingshot(df_1h, -1),
            "4h": calculate_slingshot(df_4h, -1)
        }

        # 15m taraması -> Günlük (1D) Woodie Pivot
        s15m = evaluate_eco(df_15m, df_1d, symbol, "15 Dakika (15m)", "Günlük (1D)", ss_multi)
        if s15m: signals.append(s15m)

        # 30m taraması -> TradingView Auto gereği Haftalık (1W) Woodie Pivot
        s30m = evaluate_eco(df_30m, df_1w, symbol, "30 Dakika (30m)", "Haftalık (1W)", ss_multi)
        if s30m: signals.append(s30m)

    except Exception as e:
        print(f"{symbol} analiz hatası: {e}")
    return signals

def main():
    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    if now_tsi.hour < 16 or (now_tsi.hour == 16 and now_tsi.minute < 20) or now_tsi.hour >= 23:
        print(f"ABD Seansı Kapalı (Saat: {now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return
    all_signals = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_ticker, ticker): ticker for ticker in US_TICKERS}
        for future in as_completed(futures):
            try:
                results = future.result()
                if results: all_signals.extend(results)
            except Exception as e:
                print(f"Hisse analiz hatası: {e}")
    for i, sig in enumerate(all_signals):
        send_telegram(sig)
        if i < len(all_signals) - 1:
            time.sleep(1.5)

if __name__ == "__main__":
    main()
