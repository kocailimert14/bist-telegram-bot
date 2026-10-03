import os
import sys
import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# TELEGRAM AYARLARI
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("UYARI: TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID ortam değişkeni tanımlı değil!")

# ==========================================
# QUANTFURY KRİPTO LİSTESİ (Binance USDT Pariteleri)
# ==========================================
QUANTFURY_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BCHUSDT", "LTCUSDT",
    "NEARUSDT", "APTUSDT", "DOTUSDT", "TAOUSDT", "AAVEUSDT",
    "RENDERUSDT", "INJUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
    "HBARUSDT", "OPUSDT", "ARBUSDT", "UNIUSDT", "RUNEUSDT",
    "ONDOUSDT", "POLUSDT", "SNXUSDT", "THETAUSDT", "MANAUSDT",
    "SANDUSDT", "ZECUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT",
    "FETUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "FLOKIUSDT",
    "BONKUSDT", "ICPUSDT", "FTMUSDT", "XLMUSDT", "ALGOUSDT",
    "VETUSDT", "GRTUSDT", "STXUSDT", "CRVUSDT", "DYDXUSDT",
    "IMXUSDT", "GALAUSDT", "AXSUSDT", "CHZUSDT", "EOSUSDT",
    "TRXUSDT", "BNBUSDT"
]

# ==========================================
# İNDİKATÖR PARAMETRELERİ (Görsellerdeki Ayarlar)
# ==========================================
OSCAR_LEN = 8                  # Oscar Candles
OSCAR_SMOOTHING = "SMA"        # Oscar Smoothing: SMA
SLOW_OSCAR_LEN = 16            # Slow Oscar Candles
SLOW_OSCAR_SMOOTHING = "WMA"   # Slow Oscar Smoothing: WMA
CROSS_SENSITIVITY = 1.5        # Cross Signal Sensitivity
BOTTOM_SIGNAL_LINE = 35.0      # Bottom Signal Line
TOP_SIGNAL_LINE = 65.0         # Top Signal Line

# Tekrarlayan sinyalleri engellemek için hafıza
SENT_SIGNALS = set()


def send_telegram(message: str) -> bool:
    """Telegram kanalına HTML formatında mesaj gönderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Mesajı (Simülasyon)]:\n", message)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=12)
        if res.status_code == 200:
            return True
        elif res.status_code == 429:
            retry_after = res.json().get("parameters", {}).get("retry_after", 30)
            print(f"Telegram Flood Uyarısı: {retry_after} sn beklenmeli!")
            return False
        else:
            print(f"Telegram hatası ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")
        return False


def get_binance_klines(symbol: str, interval: str = "3m", limit: int = 100) -> pd.DataFrame:
    """Binance Public API üzerinden mum verilerini çeker."""
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 20:
                df = pd.DataFrame(data, columns=[
                    'time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'qav', 'num_trades', 'tbv', 'tqv', 'ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                # Türkiye saati (UTC+3)
                df.index = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    # Bybit Spot Fallback
    try:
        bybit_int = "3" if interval == "3m" else "5"
        url_bybit = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bybit_int}&limit={limit}"
        res = requests.get(url_bybit, timeout=8)
        if res.status_code == 200:
            raw_list = res.json().get('result', {}).get('list', [])
            if raw_list and len(raw_list) >= 20:
                raw_list = raw_list[::-1]
                df = pd.DataFrame(raw_list, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turn'])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df['time'].astype(np.int64), unit='ms') + pd.Timedelta(hours=3)
                return df
    except Exception:
        pass

    return pd.DataFrame()


def calculate_oscar(df: pd.DataFrame):
    """
    OSCAR İndikatörünün Pine Script ile birebir hesaplaması:
    A = highest(close, len)
    B = lowest(close, len)
    OscarRough = (close - B) / (A - B) * 100
    Oscar1 = (OscarRough[1] / 3) * 2
    OscarThird = OscarRough / 3
    Oscar = Oscar1 + OscarThird
    smoothedOscarRough = sma(OscarRough, len)
    smoothedOscar = sma(Oscar, len)
    """
    if df is None or len(df) < 25:
        return None

    close = df['Close']
    a = close.rolling(window=OSCAR_LEN).max()
    b = close.rolling(window=OSCAR_LEN).min()

    denom = (a - b).replace(0, 1e-10)
    oscar_rough = (close - b) / denom * 100.0

    oscar_rough_prev = oscar_rough.shift(1)
    oscar = (oscar_rough_prev / 3.0) * 2.0 + (oscar_rough / 3.0)

    # Oscar Smoothing: SMA
    smoothed_rough = oscar_rough.rolling(window=OSCAR_LEN).mean()
    smoothed_oscar = oscar.rolling(window=OSCAR_LEN).mean()

    cross_sens = (smoothed_rough - smoothed_oscar).abs()

    # Kesişimler
    crossover = (smoothed_rough > smoothed_oscar) & (smoothed_rough.shift(1) <= smoothed_oscar.shift(1))
    crossunder = (smoothed_rough < smoothed_oscar) & (smoothed_rough.shift(1) >= smoothed_oscar.shift(1))

    # 4 Ayrı Sinyal Şartı
    # 1. Siyah Üçgen (Dip Kesişimi AL)
    sig_black_buy = crossover & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough < BOTTOM_SIGNAL_LINE)

    # 2. Yeşil Ok (Alım Teyidi / Devam)
    sig_green_buy = crossover.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > smoothed_oscar) & (smoothed_oscar < BOTTOM_SIGNAL_LINE)

    # 3. Siyah Ters Üçgen (Tepe Kırılımı SAT)
    sig_black_sell = crossunder & (cross_sens > CROSS_SENSITIVITY) & (smoothed_rough > TOP_SIGNAL_LINE)

    # 4. Kırmızı Ters Ok (Satım Teyidi / Devam)
    sig_red_sell = crossunder.shift(1) & (cross_sens > CROSS_SENSITIVITY) & (smoothed_oscar > smoothed_rough) & (smoothed_oscar > TOP_SIGNAL_LINE)

    return {
        'smoothed_rough': smoothed_rough,
        'smoothed_oscar': smoothed_oscar,
        'cross_sens': cross_sens,
        'black_buy': sig_black_buy,
        'green_buy': sig_green_buy,
        'black_sell': sig_black_sell,
        'red_sell': sig_red_sell
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    """
    En yüksek istatistiksel başarı oranına sahip Boğa ve Ayı mum formasyonları.
    """
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

        # 1. BOĞA FORMASYONLARI
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

        if (is_bear[3] and is_bull[4] and c[4] >= o[3] and o[4] <= c[3]):
            return "🟢 Bullish Engulfing - Yutan Boğa (Dönüş / Boğa) (%63 Başarı)"

        if (lower_wick[4] >= 2 * body[4] and upper_wick[4] <= 0.15 * cr[4]):
            return "🔨 Hammer - Çekiç (Yükseliş Dönüş / Boğa) (%60 Başarı)"

        # 2. AYI FORMASYONLARI
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
            return "🕰️ Two Black Gapping - Boşluklu İki Kırmızı (Düşüş Devam / Ayı) (%68 Başarı)"

        if (is_bull[1] and is_bull[2] and is_bull[3] and is_bear[4] and 
            c[3] > c[2] > c[1] and o[4] >= c[3] and c[4] <= o[1]):
            return "⚔️ Bearish Three-Line Strike (Düşüş Dönüş / Ayı) (%65 Başarı)"

        if (is_bull[3] and is_bear[4] and o[4] >= c[3] and c[4] <= o[3]):
            return "🔴 Bearish Engulfing - Yutan Ayı (Dönüş / Ayı) (%63 Başarı)"

        if (upper_wick[4] >= 2 * body[4] and lower_wick[4] <= 0.15 * cr[4]):
            return "🌠 Shooting Star - Kayan Yıldız (Düşüş Dönüş / Ayı) (%60 Başarı)"

        return "Standart Mum"
    except Exception:
        return "Standart Mum"


def scan_single_coin(symbol: str):
    """Tek bir koini 3 dakikalık periyotta analiz eder ve sinyal varsa Telegram mesajı üretir."""
    try:
        df = get_binance_klines(symbol, interval="3m", limit=80)
        if df.empty or len(df) < 25:
            return None

        res = calculate_oscar(df)
        if res is None:
            return None

        # Son tamamlanan muma bakılır
        target_idx = -1
        candle_time = df.index[target_idx]
        candle_price = float(df['Close'].iloc[target_idx])
        time_str = candle_time.strftime('%H:%M')

        b_buy = bool(res['black_buy'].iloc[target_idx])
        g_buy = bool(res['green_buy'].iloc[target_idx])
        b_sell = bool(res['black_sell'].iloc[target_idx])
        r_sell = bool(res['red_sell'].iloc[target_idx])

        detected_signal = None
        signal_title = ""
        signal_icon = ""

        if b_buy:
            detected_signal = "BLACK_BUY"
            signal_title = "⬛ Siyah Üçgen (Dip Kesişimi AL)"
            signal_icon = "🟢"
        elif g_buy:
            detected_signal = "GREEN_BUY"
            signal_title = "🟩 Yeşil Ok (Alım Teyidi / Devam)"
            signal_icon = "🟢"
        elif b_sell:
            detected_signal = "BLACK_SELL"
            signal_title = "⬛ Siyah Ters Üçgen (Tepe Kırılımı SAT)"
            signal_icon = "🔴"
        elif r_sell:
            detected_signal = "RED_SELL"
            signal_title = "🟥 Kırmızı Ok (Satım Teyidi / Devam)"
            signal_icon = "🔴"

        if not detected_signal:
            return None

        # Aynı mum için tekrar sinyal atmayı engelle
        sig_key = (symbol, str(candle_time), detected_signal)
        if sig_key in SENT_SIGNALS:
            return None
        SENT_SIGNALS.add(sig_key)

        coin_name = symbol.replace("USDT", "")
        tv_link = f"https://tr.tradingview.com/chart/?symbol=BINANCE:{coin_name}USDT"

        # Mum Formasyonu Tespiti
        candle_pat = detect_candlestick_patterns(df)

        message = (
            f"{signal_icon} <b>OSCAR İNDİKATÖR SİNYALİ</b>\n\n"
            f"🪙 <b>Koin:</b> <a href=\"{tv_link}\">#{coin_name}/USDT</a> <i>(Grafiği Aç)</i>\n"
            f"🎯 <b>Sinyal:</b> {signal_title}\n"
            f"⏱ <b>Periyot:</b> 3 Dakika (3m)\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"💵 <b>Fiyat:</b> ${candle_price:,.4f}\n\n"
            f"🕯️ <b>Mum Formasyonu:</b> {candle_pat}\n"
        )
        return message

    except Exception as e:
        return None


def run_scanner():
    """Tüm Quantfury koinlerini paralel olarak tarar."""
    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- [{now_str}] OSCAR 3m Tarama Başlatıldı ({len(QUANTFURY_COINS)} Koin) ---")

    found_signals = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in QUANTFURY_COINS}
        for future in as_completed(future_to_coin):
            msg = future.result()
            if msg:
                send_telegram(msg)
                found_signals += 1

    elapsed = time.time() - start_time
    print(f"Tarama tamamlandı! Bulunan Sinyal: {found_signals} | Süre: {elapsed:.2f} saniye")


if __name__ == "__main__":
    is_loop = "--loop" in sys.argv
    duration_minutes = None

    if "--duration" in sys.argv:
        try:
            d_idx = sys.argv.index("--duration")
            duration_minutes = float(sys.argv[d_idx + 1])
        except Exception:
            pass

    if is_loop:
        dur_msg = f"({duration_minutes} dakika boyunca)" if duration_minutes else "(süresiz)"
        print(f"OSCAR 3m Tarayıcı döngü modunda çalışıyor {dur_msg} - Her 3 dakikada bir kontrol...")
        start_ts = time.time()
        while True:
            run_scanner()
            if duration_minutes and (time.time() - start_ts) >= (duration_minutes * 60 - 10):
                print(f"Belirlenen çalışma süresi ({duration_minutes} dakika) tamamlandı. Görev başarıyla sonlandırılıyor.")
                break
            time.sleep(180)  # 3 dakika bekle
    else:
        run_scanner()
