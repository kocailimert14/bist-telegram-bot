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
        elif (v_ma1 < v_ma2) and (v_ma2 < v_ma3):
            nokta_renk = "🔴 Kırmızı"
        else:
            nokta_renk = "🟡 Sarı"

        return kanal_renk, nokta_renk
    except Exception:
        return "Belirsiz", "Belirsiz"

def calculate_smc(df: pd.DataFrame, idx: int = -1):
    """LuxAlgo Smart Money Concepts (SMC): Trend, Bölge, Yapı Kırılımı ve Order Block."""
    if df is None or len(df) < 30:
        return "Belirsiz", "Belirsiz", "Belirsiz", "Belirsiz"

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)
    
    p_len = 5
    trend = 0
    last_structure = "▫️ Yapı Korunuyor"
    
    swing_high = high[0]
    swing_low = low[0]
    swing_high_crossed = False
    swing_low_crossed = False
    
    bullish_obs = []
    bearish_obs = []
    
    for i in range(p_len, n):
        is_p_high = True
        is_p_low = True
        cand_high = high[i - p_len]
        cand_low = low[i - p_len]
        
        for k in range(i - 2 * p_len, i + 1):
            if k < 0 or k >= n or k == (i - p_len):
                continue
            if high[k] >= cand_high:
                is_p_high = False
            if low[k] <= cand_low:
                is_p_low = False
                
        if is_p_high:
            swing_high = cand_high
            swing_high_crossed = False
            start_k = max(0, i - 2 * p_len)
            highest_k = start_k + np.argmax(high[start_k:i])
            bearish_obs.append((low[highest_k], high[highest_k]))
            if len(bearish_obs) > 5:
                bearish_obs.pop(0)

        if is_p_low:
            swing_low = cand_low
            swing_low_crossed = False
            start_k = max(0, i - 2 * p_len)
            lowest_k = start_k + np.argmin(low[start_k:i])
            bullish_obs.append((low[lowest_k], high[lowest_k]))
            if len(bullish_obs) > 5:
                bullish_obs.pop(0)

        c_price = close[i]
        if c_price > swing_high and not swing_high_crossed:
            swing_high_crossed = True
            last_structure = "⚡ CHoCH (Boğa Dönüşü)" if trend == -1 else "⚡ BOS (Boğa Devamı)"
            trend = 1
        elif c_price < swing_low and not swing_low_crossed:
            swing_low_crossed = True
            last_structure = "⚡ CHoCH (Ayı Dönüşü)" if trend == 1 else "⚡ BOS (Ayı Devamı)"
            trend = -1

    smc_trend = "🟢 Boğa (Bullish)" if trend == 1 else "🔴 Ayı (Bearish)" if trend == -1 else "⚪ Nötr"

    curr_price = float(close[idx])
    recent_high = np.max(high[-40:])
    recent_low = np.min(low[-40:])
    rng = recent_high - recent_low
    
    if rng > 0:
        percent = (curr_price - recent_low) / rng
        if percent >= 0.525:
            smc_zone = "🔴 Premium (Pahalı / Satış Bölgesi)"
        elif percent <= 0.475:
            smc_zone = "🟢 Discount (Ucuz / Alım Bölgesi)"
        else:
            smc_zone = "⚪ Denge (Equilibrium)"
    else:
        smc_zone = "⚪ Denge"

    smc_ob = "⚪ Blok Dışı"
    for ob_low, ob_high in reversed(bullish_obs):
        if ob_low <= curr_price <= ob_high:
            smc_ob = "🟢 Alım Bloğunda (Talep)"
            break
            
    if smc_ob == "⚪ Blok Dışı":
        for ob_low, ob_high in reversed(bearish_obs):
            if ob_low <= curr_price <= ob_high:
                smc_ob = "🔴 Satış Bloğunda (Arz)"
                break

    return smc_trend, smc_zone, last_structure, smc_ob

def evaluate_eco_crypto(df: pd.DataFrame, symbol: str, tf_label: str):
    """TradingView Evan Cabral Oscillators (ECO), SlingShot ve SMC teyitleri."""
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

    c_curr = float(stoch.iloc[-1])
    c_prev = float(stoch.iloc[-2])
    c_prev2 = float(stoch.iloc[-3]) if len(stoch) >= 3 else c_prev

    coin_name = symbol.replace("USDT", "")

    # 1. AL SİNYALİ
    if (c_prev < 10 and c_curr > 10) or (c_prev2 < 10 and c_prev > 10):
        if c_prev < 10 and c_curr > 10:
            target_idx = -1
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        else:
            target_idx = -2
            durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')

        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)
        smc_trend, smc_zone, smc_struct, smc_ob = calculate_smc(df, target_idx)

        return (
            f"🟢 <b>KRİPTO AL SİNYALİ (Evan Cabral - ECO)</b>\n\n"
            f"🪙 <b>Koin:</b> #{coin_name}/USDT\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
            f"📊 <b>DMI-Stoch:</b> {stoch.iloc[target_idx]:.1f} (Önceki: {stoch.iloc[target_idx-1]:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch 10 seviyesini yukarı kesti ('B')\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}\n\n"
            f"<b>🏛 Akıllı Para Konsepti (SMC):</b>\n"
            f"▫️ <b>Kurumsal Trend:</b> {smc_trend}\n"
            f"▫️ <b>Fiyat Bölgesi:</b> {smc_zone}\n"
            f"▫️ <b>Yapı Kırılımı:</b> {smc_struct}\n"
            f"▫️ <b>Order Block:</b> {smc_ob}"
        )

    # 2. SAT SİNYALİ
    elif (c_prev > 90 and c_curr < 90) or (c_prev2 > 90 and c_prev < 90):
        if c_prev > 90 and c_curr < 90:
            target_idx = -1
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        else:
            target_idx = -2
            durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')

        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)
        smc_trend, smc_zone, smc_struct, smc_ob = calculate_smc(df, target_idx)

        return (
            f"🔴 <b>KRİPTO SAT SİNYALİ (Evan Cabral - ECO)</b>\n\n"
            f"🪙 <b>Koin:</b> #{coin_name}/USDT\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
            f"📊 <b>DMI-Stoch:</b> {stoch.iloc[target_idx]:.1f} (Önceki: {stoch.iloc[target_idx-1]:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch 90 seviyesini aşağı kesti ('S')\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}\n\n"
            f"<b>🏛 Akıllı Para Konsepti (SMC):</b>\n"
            f"▫️ <b>Kurumsal Trend:</b> {smc_trend}\n"
            f"▫️ <b>Fiyat Bölgesi:</b> {smc_zone}\n"
            f"▫️ <b>Yapı Kırılımı:</b> {smc_struct}\n"
            f"▫️ <b>Order Block:</b> {smc_ob}"
        )

    return None

def scan_single_coin(symbol: str):
    """Tek bir koin için sadece 1h periyodunu tarar."""
    found_signals = []
    try:
        for tf_key, label in TIMEFRAMES:
            df = get_binance_klines(symbol, tf_key)
            sig = evaluate_eco_crypto(df, symbol, label)
            if sig:
                found_signals.append(sig)
    except Exception as e:
        print(f"{symbol} analiz hatası: {e}")
    return found_signals

def main():
    # GECE SESSİZ MOD KORUMASI (Saat 00:00 - 08:00 TSİ arası kesinlikle çalışmaz)
    now_tsi = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=3)
    if 0 <= now_tsi.hour < 8:
        print(f"Gece sessiz mod aktif (Saat: {now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    print(f"Kripto Evan Cabral (ECO) Taraması Başlıyor ({len(COINS)} Koin; 1 Saatlik)...")
    all_signals = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_coin, coin): coin for coin in COINS}
        for future in as_completed(futures):
            try:
                sigs = future.result()
                if sigs:
                    all_signals.extend(sigs)
            except Exception as e:
                print(f"İş parçacığı hatası: {e}")

    # Sinyalleri Telegram flood limitine takılmadan 1.5 saniye arayla güvenle gönder
    toplam = 0
    for i, sig in enumerate(all_signals):
        success = send_telegram(sig)
        if success:
            toplam += 1
        if i < len(all_signals) - 1:
            time.sleep(1.5)

    print(f"Tarama bitti! Üretilen yeni sinyal sayısı: {toplam}")

if __name__ == "__main__":
    main()
