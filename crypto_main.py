import os
import sys
import time
import numpy as np
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("HATA: Telegram Token veya Chat ID bulunamadı!")
    sys.exit(1)

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BCHUSDT", "LTCUSDT", 
    "NEARUSDT", "APTUSDT", "DOTUSDT", "TAOUSDT", "AAVEUSDT", 
    "RENDERUSDT", "INJUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", 
    "HBARUSDT", "OPUSDT", "ARBUSDT", "UNIUSDT", "RUNEUSDT", 
    "ONDOUSDT", "POLUSDT", "SNXUSDT", "THETAUSDT", "MANAUSDT", 
    "SANDUSDT", "ZECUSDT", "HYPEUSDT", "GRAMUSDT", "SUSDT"
]

def send_telegram(message: str) -> bool:
    """Telegram'a HTML formatında güvenli bildirim gönderir."""
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

def get_binance_klines(symbol: str, interval: str) -> pd.DataFrame:
    """Binance resmi engelsiz sunucusu üzerinden kline çeker."""
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
                df['Volume'] = df['Volume'].astype(float)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    interval_map = {"15m": "15", "1h": "60", "4h": "240"}
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
                df['Volume'] = df['Volume'].astype(float)
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

def calculate_slingshot(df: pd.DataFrame, idx: int = -1):
    """Sling Shot System: Düz Kanal Rengi ve Noktasal Trend Rengi hesabı."""
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
    """Görseldeki gibi düşen eğimli direnç kırılımını ve dipten yükselen trend başlangıcını hesaplar."""
    if df is None or len(df) < 20:
        return "Standart Hareket", "Yeni Trend Yok"
    try:
        window = min(lookback, len(df))
        sub_h = df['High'].values[-window:]
        sub_l = df['Low'].values[-window:]
        sub_c = df['Close'].values[-window:]
        curr_c = sub_c[-1]
        curr_h = sub_h[-1]

        # 1. DÜŞEN TREND ÇİZGİSİ HESABI
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
                        dusen_kirilim_durumu = f"✅ Düşen Trend Çizgisi Yukarı Kırıldı! (Trend: ${line_val_now:,.4f}) (Boğa)"
                    else:
                        dusen_kirilim_durumu = f"🚀 Düşen Trend Üzerinde Seyrediyor (Trend: ${line_val_now:,.4f}) (Boğa)"
                elif curr_h >= line_val_now and curr_c <= line_val_now:
                    dusen_kirilim_durumu = f"⚠️ Düşen Trend Çizgisi Test Ediliyor (Direnç: ${line_val_now:,.4f}) (Nötr)"
                else:
                    dusen_kirilim_durumu = f"Düşen Trend Altında (Direnç: ${line_val_now:,.4f})"

        # 2. YENİ YÜKSELEN TREND BAŞLATTI MI?
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
                        trend_baslatma_durumu = f"📈 Yeni Yükselen Trend Başlattı! (Dipten Destek: ${up_line_now:,.4f}, Yükselen Dip Onaylı) (Boğa)"
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
    """Modern Gün İçi Piyasalara Uyumlu Mum Formasyonu Tanıma Motoru (Hatası Giderilmiş Skalar Sürüm)."""
    if df is None or len(df) < 5:
        return "Standart Mum"
    try:
        sub = df.iloc[-5:]
        o0, o1, o2, o3, o4 = sub['Open'].values
        h0, h1, h2, h3, h4 = sub['High'].values
        l0, l1, l2, l3, l4 = sub['Low'].values
        c0, c1, c2, c3, c4 = sub['Close'].values

        body4 = abs(c4 - o4)
        range4 = max(h4 - l4, 1e-10)
        upper_wick4 = h4 - max(o4, c4)
        lower_wick4 = min(o4, c4) - l4
        is_bull4 = c4 > o4
        is_bear4 = c4 < o4

        body3 = abs(c3 - o3)
        range3 = max(h3 - l3, 1e-10)
        body2 = abs(c2 - o2)
        range2 = max(h2 - l2, 1e-10)

        is_bull3 = c3 > o3
        is_bear3 = c3 < o3
        is_bull2 = c2 > o2
        is_bear2 = c2 < o2
        is_bull1 = c1 > o1
        is_bear1 = c1 < o1

        # 1. THREE LINE STRIKE
        if is_bear1 and is_bear2 and is_bear3 and is_bull4:
            if c3 < c2 < c1 and c4 >= max(o1, o2):
                return "⚔️ Three Line Strike (Boğa)"
        if is_bull1 and is_bull2 and is_bull3 and is_bear4:
            if c3 > c2 > c1 and c4 <= min(o1, o2):
                return "⚔️ Three Line Strike (Ayı)"

        # 2. THREE BLACK CROWS (Üç Kara Karga)
        if is_bear2 and is_bear3 and is_bear4:
            if c4 < c3 < c2 and body4 > range4 * 0.4 and body3 > range3 * 0.4:
                return "🦅 Üç Kara Karga - Three Black Crows (Ayı)"

        # 3. THREE WHITE SOLDIERS (Üç Beyaz Asker)
        if is_bull2 and is_bull3 and is_bull4:
            if c4 > c3 > c2 and body4 > range4 * 0.4 and body3 > range3 * 0.4:
                return "🛡️ Üç Beyaz Asker - Three White Soldiers (Boğa)"

        # 4. MORNING STAR (Sabah Yıldızı - Boğa)
        if is_bear2 and (body2 > range2 * 0.4) and (body3 < range3 * 0.35) and is_bull4:
            if c4 > (o2 + c2) / 2:
                return "⭐ Sabah Yıldızı - Morning Star (Boğa)"

        # 5. EVENING STAR (Akşam Yıldızı - Ayı)
        if is_bull2 and (body2 > range2 * 0.4) and (body3 < range3 * 0.35) and is_bear4:
            if c4 < (o2 + c2) / 2:
                return "🌙 Akşam Yıldızı - Evening Star (Ayı)"

        # 6. ENGULFING (Yutan Boğa / Yutan Ayı)
        if is_bear3 and is_bull4 and c4 >= o3 and body4 >= body3:
            return "🟢 Yutan Boğa - Bullish Engulfing (Boğa)"
        if is_bull3 and is_bear4 and c4 <= o3 and body4 >= body3:
            return "🔴 Yutan Ayı - Bearish Engulfing (Ayı)"

        # 7. PIERCING LINE / DARK CLOUD COVER
        if is_bear3 and is_bull4 and c4 > (o3 + c3) / 2 and c4 < o3 and body4 >= range4 * 0.4:
            return "⚡ Delen Çizgi - Piercing Line (Boğa)"
        if is_bull3 and is_bear4 and c4 < (o3 + c3) / 2 and c4 > o3 and body4 >= range4 * 0.4:
            return "☁️ Kara Bulut Örtüsü - Dark Cloud (Ayı)"

        # 8. ÇEKİÇ & PINBAR (Hammer - Boğa)
        if lower_wick4 >= 1.5 * body4 and upper_wick4 <= 0.4 * body4 and body4 > 0:
            return "🔨 Çekiç - Hammer / Pinbar (Boğa)"

        # 9. TERS ÇEKİÇ (Inverted Hammer - Boğa)
        if upper_wick4 >= 1.5 * body4 and lower_wick4 <= 0.4 * body4 and is_bull4 and body4 > 0:
            return "🪓 Ters Çekiç - Inverted Hammer (Boğa)"

        # 10. KAYAN YILDIZ (Shooting Star - Ayı)
        if upper_wick4 >= 1.5 * body4 and lower_wick4 <= 0.4 * body4 and is_bear4 and body4 > 0:
            return "🌠 Kayan Yıldız - Shooting Star (Ayı)"

        # 11. ASILI ADAM (Hanging Man - Ayı)
        if lower_wick4 >= 1.5 * body4 and upper_wick4 <= 0.4 * body4 and is_bear4 and body4 > 0:
            return "🪢 Asılı Adam - Hanging Man (Ayı)"

        # 12. DOJI (Kararsızlık / Dönüş Hazırlığı)
        if body4 / range4 < 0.12:
            if lower_wick4 >= 2 * upper_wick4:
                return "🦗 Yusufçuk Doji - Dragonfly (Boğa)"
            elif upper_wick4 >= 2 * lower_wick4:
                return "🪦 Mezar Taşı Doji - Gravestone (Ayı)"
            else:
                return "⚖️ Doji (Kararsız Mum)"

        # 13. MARUBOZU (Güçlü Momentum Gövdesi)
        if body4 / range4 >= 0.75:
            return "🚀 Güçlü Boğa Marubozu (Boğa)" if is_bull4 else "🩸 Güçlü Ayı Marubozu (Ayı)"

        return "Standart Mum"
    except Exception as e:
        return "Standart Mum"

def detect_ict_smc_models(df: pd.DataFrame) -> str:
    """Pinkman Akademi - 12 Kurulum (ICT & Price Action) Tanıma Motoru."""
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

        # 1. LIQUIDITY SWEEP / TURTLE SOUP (Model 2 & 10)
        swing_low = np.min(low[-16:-2])
        if (low[-2] < swing_low and close[-2] > swing_low) or (curr_l < swing_low and curr_c > swing_low):
            return "🧲 Liquidity Sweep / Turtle Soup (Tuzak Dip Temizlendi) (Boğa)"

        swing_high = np.max(high[-16:-2])
        if (high[-2] > swing_high and close[-2] < swing_high) or (curr_h > swing_high and curr_c < swing_high):
            return "🧲 Buy-Side Liquidity Sweep (Tepe Likiditesi Alındı) (Ayı)"

        # 2. FVG - FAIR VALUE GAP (Model 4 & 5)
        if low[-2] > high[-4]:
            if high[-4] <= curr_l <= low[-2] or high[-4] <= curr_c <= low[-2]:
                return "⚡ Bullish FVG (Dengesizlik Boşluğu Test Ediliyor) (Boğa)"
        if high[-2] < low[-4]:
            if high[-2] <= curr_h <= low[-4] or high[-2] <= curr_c <= low[-4]:
                return "⚡ Bearish FVG (Satış Dengesizliği Test Ediliyor) (Ayı)"

        # 3. ORDER BLOCK & BREAKER (Model 3 & 6)
        for i in range(2, 6):
            if close[-i] > opens[-i] and (close[-i] - opens[-i]) > np.std(np.abs(close - opens)) * 1.5:
                ob_idx = -(i + 1)
                if low[ob_idx] <= curr_l <= high[ob_idx] or low[ob_idx] <= curr_c <= high[ob_idx]:
                    return "🧱 Bullish Order Block (OB Kurumsal Alım Bölgesi) (Boğa)"
                    break

        # 4. OTE - OPTIMAL TRADE ENTRY (%62-%79 FIBONACCI) (Model 9)
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
    """Kuvvetli Majör Destek & Direnç Seviyeleri (Mikro %0.1 gürültüleri eler)."""
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

def calculate_score_and_rvol(df: pd.DataFrame, idx: int, sig_type: str, ss_multi: dict, d_sup: float, d_res: float, candle_pat: str, dusen_trend: str, yeni_trend: str):
    """Göreceli Hacim (RVol) ve 1-5 Yıldız Sinyal Güven Puanı hesabı."""
    try:
        vol = df['Volume'].squeeze()
        curr_vol = float(vol.iloc[idx])
        window = 20
        if len(vol) > window + 1:
            avg_vol = float(vol.iloc[-window-1:-1].mean())
        else:
            avg_vol = float(vol.mean())
            
        rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0
        
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
        if ("Boğa" in candle_pat and sig_type == "BUY") or ("Kırıldı" in dusen_trend) or ("Yeni Yükselen" in yeni_trend):
            puan += 1

        puan = min(puan, 5)
        yildizlar = "⭐" * puan
        skor_metni = f"{yildizlar} ({puan}/5)"
        return hacim_metni, skor_metni
    except Exception:
        return "⚪ Normal", "⭐⭐⭐ (3/5)"

def evaluate_eco_crypto(df: pd.DataFrame, symbol: str, tf_label: str, ss_multi: dict):
    """TradingView Evan Cabral Oscillators: SADECE o an açık olan canlı mumdaki kesişim."""
    if df is None or df.empty or len(df) < 20:
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

    # SADECE VE SADECE O AN AÇIK OLAN CANLI MUM KONTROL EDİLİR
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
    candle_price = float(close.iloc[target_idx])
    time_str = candle_time.strftime('%H:%M')
    coin_name = symbol.replace("USDT", "")
    tv_link = f"https://tr.tradingview.com/chart/?symbol=BINANCE:{symbol}"

    # Geometrik Trend Çizgileri ve Formasyon Teyitleri
    dusen_trend, yeni_trend = detect_diagonal_trendline_and_initiation(df)
    candle_pat = detect_candlestick_patterns(df)
    smc_model = detect_ict_smc_models(df)

    sup, res, d_sup, d_res = calculate_strong_sr(df, target_idx)
    hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, sig_type, ss_multi, d_sup, d_res, candle_pat, dusen_trend, yeni_trend)

    ss_15m_k, ss_15m_n = ss_multi.get("15m", ("Belirsiz", "Belirsiz"))
    ss_1h_k, ss_1h_n = ss_multi.get("1h", ("Belirsiz", "Belirsiz"))
    ss_4h_k, ss_4h_n = ss_multi.get("4h", ("Belirsiz", "Belirsiz"))

    sr_metni = ""
    if sup is not None and res is not None:
        sr_metni = (
            f"\n\n<b>🎯 Kuvvetli Destek & Direnç:</b>\n"
            f"▫️ <b>Ana Destek:</b> ${sup:,.4f} (<code>{d_sup:+.1f}%</code>)\n"
            f"▫️ <b>Ana Direnç:</b> ${res:,.4f} (<code>{d_res:+.1f}%</code>)"
        )

    tag = "🟢 <b>KRİPTO AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>KRİPTO SAT SİNYALİ</b>"

    # Sadeleştirilmiş Telegram Kartı
    return (
        f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
        f"🪙 <b>Koin:</b> <a href=\"{tv_link}\">#{coin_name}/USDT</a> <i>(Grafiği Aç)</i>\n"
        f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
        f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
        f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n\n"
        f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
        f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
        f"<b>🕯️ Formasyon & Trend Teyitleri:</b>\n"
        f"▫️ <b>Düşen Trend Kırılımı:</b> {dusen_trend}\n"
        f"▫️ <b>Trend Başlatma Durumu:</b> {yeni_trend}\n"
        f"▫️ <b>Mum Formasyonu:</b> {candle_pat}\n"
        f"▫️ <b>ICT / SMC Modeli:</b> {smc_model}\n\n"
        f"<b>📈 Trend Teyitleri (SlingShot Multi-TF):</b>\n"
        f"▫️ <b>15 Dakika (15m):</b> {ss_15m_k} Kanal | {ss_15m_n} Nokta\n"
        f"▫️ <b>1 Saat (1h):</b> {ss_1h_k} Kanal | {ss_1h_n} Nokta\n"
        f"▫️ <b>4 Saat (4h):</b> {ss_4h_k} Kanal | {ss_4h_n} Nokta"
        f"{sr_metni}"
    )

def scan_single_coin(symbol: str, scan_15m: bool, scan_1h: bool):
    """Tek bir koin için 15m, 1h ve 4h SlingShot teyitlerini hesaplayıp tarar."""
    found_signals = []
    try:
        df_15m = get_binance_klines(symbol, "15m")
        df_1h = get_binance_klines(symbol, "1h")
        df_4h = get_binance_klines(symbol, "4h")

        ss_multi = {
            "15m": calculate_slingshot(df_15m, -1),
            "1h": calculate_slingshot(df_1h, -1),
            "4h": calculate_slingshot(df_4h, -1)
        }

        # 1. 15 Dakikalık Canlı Mum
        if scan_15m and df_15m is not None and not df_15m.empty:
            s15 = evaluate_eco_crypto(df_15m, symbol, "15 Dakika (15m)", ss_multi)
            if s15:
                found_signals.append(s15)

        # 2. 1 Saatlik Canlı Mum
        if scan_1h and df_1h is not None and not df_1h.empty:
            s1h = evaluate_eco_crypto(df_1h, symbol, "1 Saat (1h)", ss_multi)
            if s1h:
                found_signals.append(s1h)

    except Exception as e:
        print(f"{symbol} analiz hatası: {e}")
    return found_signals

def determine_crypto_modes(now_tsi):
    """Kripto için saate göre hangi periyodun taranacağını belirler."""
    h = now_tsi.hour
    m = now_tsi.minute
    
    # Gece 00:00 - 08:00 TSİ arası tamamen sessiz (durur)
    if 0 <= h < 8:
        return False, False
        
    scan_15m = True
    scan_1h = (40 <= m <= 55)
    return scan_15m, scan_1h

def main():
    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    scan_15m, scan_1h = determine_crypto_modes(now_tsi)
    
    if not scan_15m and not scan_1h:
        print(f"Gece sessiz mod aktif (Saat: {now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    modlar = []
    if scan_15m: modlar.append("15m Canlı Mum")
    if scan_1h: modlar.append("1h Canlı Mum")

    print(f"Kripto Taraması Başlıyor (Saat: {now_tsi.strftime('%H:%M')} TSİ) -> Aktif Modlar: {', '.join(modlar)} ({len(COINS)} Koin)...")
    all_signals = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_coin, coin, scan_15m, scan_1h): coin for coin in COINS}
        for future in as_completed(futures):
            try:
                sigs = future.result()
                if sigs:
                    all_signals.extend(sigs)
            except Exception as e:
                print(f"İş parçacığı hatası: {e}")

    toplam = 0
    for i, sig in enumerate(all_signals):
        success = send_telegram(sig)
        if success:
            toplam += 1
        if i < len(all_signals) - 1:
            time.sleep(1.5)

    print(f"Kripto Taraması bitti! Bulunan toplam sinyal: {toplam}")

if __name__ == "__main__":
    main()
