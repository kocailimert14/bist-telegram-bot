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
                df['Volume'] = df['Volume'].astype(float)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    # Yedek: Bybit
    interval_map = {"15m": "15", "1h": "60"}
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
        elif (v_ma1 < v_lower if False else v_ma2 < v_ma3):
            nokta_renk = "🔴 Kırmızı"
        else:
            nokta_renk = "🟡 Sarı"

        return kanal_renk, nokta_renk
    except Exception:
        return "Belirsiz", "Belirsiz"

def calculate_support_resistance(df: pd.DataFrame, idx: int = -1):
    """Otomatik Swing Destek & Direnç Seviyeleri ve Yüzdesel Mesafeleri."""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        curr_price = float(close.iloc[idx])
        window = min(30, len(df))
        
        sub_high = high.iloc[-window:]
        sub_low = low.iloc[-window:]
        
        higher_highs = sub_high[sub_high > curr_price]
        resistance = float(higher_highs.min()) if not higher_highs.empty else float(sub_high.max())
        
        lower_lows = sub_low[sub_low < curr_price]
        support = float(lower_lows.max()) if not lower_lows.empty else float(sub_low.min())
        
        dist_sup = ((support - curr_price) / curr_price) * 100
        dist_res = ((resistance - curr_price) / curr_price) * 100
        
        return support, resistance, dist_sup, dist_res
    except Exception:
        return None, None, 0.0, 0.0

def calculate_score_and_rvol(df: pd.DataFrame, idx: int, sig_type: str, kanal_renk: str, nokta_renk: str, d_sup: float, d_res: float):
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

        puan = 1 # ECO kesişim puanı
        if (sig_type == "BUY" and "Yeşil" in kanal_renk) or (sig_type == "SELL" and "Kırmızı" in kanal_renk):
            puan += 1
        if (sig_type == "BUY" and "Yeşil" in nokta_renk) or (sig_type == "SELL" and "Kırmızı" in nokta_renk):
            puan += 1
        if rvol >= 1.1:
            puan += 1
        if sig_type == "BUY" and abs(d_res) >= 0.50:
            puan += 1
        elif sig_type == "SELL" and abs(d_sup) >= 0.50:
            puan += 1

        yildizlar = "⭐" * puan
        skor_metni = f"{yildizlar} ({puan}/5)"
        return hacim_metni, skor_metni
    except Exception:
        return "⚪ Normal", "⭐⭐⭐ (3/5)"

def evaluate_eco_crypto(df: pd.DataFrame, symbol: str, tf_label: str, tf_key: str):
    """TradingView Evan Cabral Oscillators (ECO), SlingShot, Destek-Direnç ve Puanlama."""
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

    cross_up = (stoch.shift(1) < 10) & (stoch > 10)
    cross_down = (stoch.shift(1) > 90) & (stoch < 90)

    target_idx = None
    sig_type = None
    durum_metni = ""

    # KURAL 1: 15 Dakikalıkta SADECE o an açık olan canlı mum kontrol edilir (iloc[-1])
    if tf_key == "15m":
        if cross_up.iloc[-1]:
            target_idx = -1
            sig_type = "BUY"
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        elif cross_down.iloc[-1]:
            target_idx = -1
            sig_type = "SELL"
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"

    # KURAL 2: 1 Saatlikte HEM canlı mum (iloc[-1]) HEM bir önceki mum (iloc[-2]) kontrol edilir
    elif tf_key == "1h":
        if cross_up.iloc[-1]:
            target_idx = -1
            sig_type = "BUY"
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        elif cross_down.iloc[-1]:
            target_idx = -1
            sig_type = "SELL"
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
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
        candle_price = float(close.iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')
        coin_name = symbol.replace("USDT", "")
        
        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)
        sup, res, d_sup, d_res = calculate_support_resistance(df, target_idx)
        hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, sig_type, kanal_renk, nokta_renk, d_sup, d_res)

        sr_metni = ""
        if sup is not None and res is not None:
            sr_metni = (
                f"\n\n<b>🎯 Destek & Direnç Seviyeleri:</b>\n"
                f"▫️ <b>En Yakın Destek:</b> ${sup:,.4f} (<code>{d_sup:+.1f}%</code>)\n"
                f"▫️ <b>En Yakın Direnç:</b> ${res:,.4f} (<code>{d_res:+.1f}%</code>)"
            )

        tag = "🟢 <b>KRİPTO AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>KRİPTO SAT SİNYALİ</b>"
        trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

        return (
            f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
            f"🪙 <b>Koin:</b> #{coin_name}/USDT\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n"
            f"📊 <b>DMI-Stoch:</b> {stoch.iloc[target_idx]:.1f} (Önceki: {stoch.iloc[target_idx-1]:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch {trigger}\n\n"
            f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
            f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}"
            f"{sr_metni}"
        )

    return None

def scan_single_coin(symbol: str, scan_15m: bool, scan_1h: bool):
    """Sadece planlanan saatteki ilgili periyotları analiz eder."""
    found_signals = []
    try:
        if scan_15m:
            df_15m = get_binance_klines(symbol, "15m")
            s15 = evaluate_eco_crypto(df_15m, symbol, "15 Dakika (15m)", "15m")
            if s15:
                found_signals.append(s15)

        if scan_1h:
            df_1h = get_binance_klines(symbol, "1h")
            s1h = evaluate_eco_crypto(df_1h, symbol, "1 Saat (1h)", "1h")
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
    now_tsi = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=3)
    scan_15m, scan_1h = determine_crypto_modes(now_tsi)
    
    if not scan_15m and not scan_1h:
        print(f"Gece sessiz mod aktif (Saat: {now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    modlar = []
    if scan_15m: modlar.append("15 Dakika (Canlı Mum)")
    if scan_1h: modlar.append("1 Saat (Canlı + Önceki Mum)")

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

    print(f"Kripto Taraması bitti! Üretilen yeni sinyal sayısı: {toplam}")

if __name__ == "__main__":
    main()
