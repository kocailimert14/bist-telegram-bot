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

# BIST 100 En Büyük ve En Çok İşlem Gören 100 Şirket
BIST_TICKERS = [
    "THYAO.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS", "GARAN.IS", "AKBNK.IS", "YKBNK.IS", "ISCTR.IS", "BIMAS.IS",
    "SISE.IS",  "SAHOL.IS", "FROTO.IS", "TOASO.IS", "ENKAI.IS", "PGSUS.IS", "KOZAL.IS", "PETKM.IS", "EKGYO.IS", "HEKTS.IS",
    "SASA.IS",  "ASTOR.IS", "ALARK.IS", "ARCLK.IS", "GUBRF.IS", "KRDMD.IS", "KRDMB.IS", "ODAS.IS",  "OYAKC.IS", "SOKM.IS",
    "TAVHL.IS", "PKART.IS", "TKFEN.IS", "TTKOM.IS", "TCELL.IS", "VESTL.IS", "MGROS.IS", "TURSG.IS", "VAKBN.IS", "HALKB.IS",
    "ISGYO.IS", "DOHOL.IS", "KOZAA.IS", "IPEKE.IS", "BERA.IS",  "CIMSA.IS", "AKSA.IS",  "AKSEN.IS", "QUAGR.IS", "CANTE.IS",
    "MIATK.IS", "REEDR.IS", "SDTTR.IS", "KONTR.IS", "EUPWR.IS", "GESAN.IS", "CWENE.IS", "ALFAS.IS", "BOSSA.IS", "BRSAN.IS",
    "BRYAT.IS", "CCOLA.IS", "DOAS.IS",  "EGEEN.IS", "ECZYT.IS", "GENIL.IS", "GOKNR.IS", "GWIND.IS", "ISMEN.IS", "IZENR.IS",
    "KCAER.IS", "KMPUR.IS", "KONKA.IS", "KORDS.IS", "MTRKS.IS", "OTKAR.IS", "PARSN.IS", "PENTA.IS", "SAYAS.IS", "SELEC.IS",
    "SMRTG.IS", "TABGD.IS", "TATEN.IS", "TMSN.IS",  "TRGYO.IS", "TSKB.IS",  "TTRAK.IS", "ULKER.IS", "VESBE.IS", "YEOTK.IS",
    "ZOREN.IS", "AGHOL.IS", "AHGAZ.IS", "AKFYE.IS", "ALGYO.IS", "ANSGR.IS", "BIENY.IS", "BIOEN.IS", "BOBET.IS", "CUSAN.IS"
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

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance verisini temizler ve Türkiye Saatine (TSİ) kilitler."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['High', 'Low', 'Close'])
    
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_convert('Europe/Istanbul')
    else:
        df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
    return df

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

def calculate_score_and_rvol(df: pd.DataFrame, idx: int, sig_type: str, ss_multi: dict, d_sup: float, d_res: float):
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
        if sig_type == "BUY" and abs(d_res) >= 0.50:
            puan += 1
        elif sig_type == "SELL" and abs(d_sup) >= 0.50:
            puan += 1

        yildizlar = "⭐" * puan
        skor_metni = f"{yildizlar} ({puan}/5)"
        return hacim_metni, skor_metni
    except Exception:
        return "⚪ Normal", "⭐⭐⭐ (3/5)"

def make_bist_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """TradingView BIST 4 saatlik mumlarını (09:00-13:00 ve 13:00-18:10) oluşturur."""
    df = clean_df(df_1h)
    if df.empty or len(df) < 15:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    df_copy['half'] = np.where(df_copy.index.hour < 13, 1, 2)
    
    df_4h = df_copy.groupby(['date', 'half']).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    new_idx = []
    for d, h in df_4h.index:
        hour_str = "09:00:00" if h == 1 else "13:00:00"
        new_idx.append(pd.Timestamp(f"{d} {hour_str}+03:00"))
    df_4h.index = pd.DatetimeIndex(new_idx)
    return df_4h

def evaluate_eco(df: pd.DataFrame, symbol: str, tf_label: str, ss_multi: dict):
    """TradingView Pine Script ECO: SADECE o an açık olan canlı mumdaki kesişim."""
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

    # SADECE O AN AÇIK OLAN CANLI MUM KONTROL EDİLİR
    c_prev = float(stoch.iloc[-2]) # Stoch[1]
    c_curr = float(stoch.iloc[-1]) # Stoch (Canlı Mum)

    sig_type = None
    if c_prev < 10 and c_curr > 10:
        sig_type = "BUY"
    elif c_prev > 90 and c_curr < 90:
        sig_type = "SELL"
    else:
        return None

    target_idx = -1
    candle_time = df.index[target_idx]

    # Zaman Tazelik Kontrolü (Bayat veri koruması)
    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    if candle_time.date() != now_tsi.date():
        return None

    age_minutes = (now_tsi - candle_time).total_seconds() / 60.0
    if tf_label == "1 Saat (1h)" and age_minutes > 120:
        return None
    if tf_label == "4 Saat (4h)" and now_tsi.hour >= 16 and candle_time.hour < 13:
        return None

    candle_price = float(close.iloc[target_idx])

    if "Saat" in tf_label:
        time_str = candle_time.strftime('%H:00')
    else:
        time_str = candle_time.strftime('%d.%m.%Y')

    hisse_adi = symbol.replace('.IS', '')
    tv_link = f"https://tr.tradingview.com/chart/?symbol=BIST:{hisse_adi}"

    sup, res, d_sup, d_res = calculate_strong_sr(df, target_idx)
    hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, sig_type, ss_multi, d_sup, d_res)

    ss_15m_k, ss_15m_n = ss_multi.get("15m", ("Belirsiz", "Belirsiz"))
    ss_1h_k, ss_1h_n = ss_multi.get("1h", ("Belirsiz", "Belirsiz"))
    ss_4h_k, ss_4h_n = ss_multi.get("4h", ("Belirsiz", "Belirsiz"))

    sr_metni = ""
    if sup is not None and res is not None:
        sr_metni = (
            f"\n\n<b>🎯 Kuvvetli Destek & Direnç:</b>\n"
            f"▫️ <b>Ana Destek:</b> {sup:.2f} TL (<code>{d_sup:+.1f}%</code>)\n"
            f"▫️ <b>Ana Direnç:</b> {res:.2f} TL (<code>{d_res:+.1f}%</code>)"
        )

    tag = "🟢 <b>BIST AL SİNYALİ</b>" if sig_type == "BUY" else "🔴 <b>BIST SAT SİNYALİ</b>"
    trigger = "10 seviyesini yukarı kesti ('B')" if sig_type == "BUY" else "90 seviyesini aşağı kesti ('S')"

    return (
        f"{tag} <b>(Evan Cabral - ECO)</b>\n\n"
        f"📌 <b>Hisse:</b> <a href=\"{tv_link}\">#{hisse_adi}</a> <i>(Grafiği Aç)</i>\n"
        f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
        f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
        f"⚡ <b>Mum Durumu:</b> ⚠️ CANLI MUM (Anlık Sinyal)\n"
        f"💵 <b>Fiyat:</b> {candle_price:.2f} TL\n"
        f"📊 <b>DMI-Stoch:</b> {c_curr:.1f} (Önceki: {c_prev:.1f})\n"
        f"🎯 <b>Tetikleyici:</b> DMI-Stoch {trigger}\n\n"
        f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
        f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
        f"<b>📈 Trend Teyitleri (SlingShot Multi-TF):</b>\n"
        f"▫️ <b>15 Dakika (15m):</b> {ss_15m_k} Kanal | {ss_15m_n} Nokta\n"
        f"▫️ <b>1 Saat (1h):</b> {ss_1h_k} Kanal | {ss_1h_n} Nokta\n"
        f"▫️ <b>4 Saat (4h):</b> {ss_4h_k} Kanal | {ss_4h_n} Nokta"
        f"{sr_metni}"
    )

def analyze_ticker(symbol: str, scan_1h: bool, scan_4h: bool, scan_1d: bool):
    """Sadece planlanan saatteki ilgili periyotları canlı mumda analiz eder."""
    signals = []
    df_1h = None
    df_4h = None
    df_15m = None
    
    # 1. SlingShot Multi-TF verilerini hazırla
    try:
        df_1h = yf.download(symbol, period="2mo", interval="1h", progress=False)
        clean_1h = clean_df(df_1h)
        df_4h = make_bist_4h(clean_1h)
        df_15m = yf.download(symbol, period="5d", interval="15m", progress=False)
        clean_15m = clean_df(df_15m)

        ss_multi = {
            "15m": calculate_slingshot(clean_15m, -1),
            "1h": calculate_slingshot(clean_1h, -1),
            "4h": calculate_slingshot(df_4h, -1)
        }
    except Exception:
        ss_multi = {"15m": ("Belirsiz", "Belirsiz"), "1h": ("Belirsiz", "Belirsiz"), "4h": ("Belirsiz", "Belirsiz")}

    # 1. 1 Saatlik Tarama
    if scan_1h and df_1h is not None and not df_1h.empty:
        try:
            s1h = evaluate_eco(df_1h, symbol, "1 Saat (1h)", ss_multi)
            if s1h:
                signals.append(s1h)
        except Exception:
            pass

    # 2. 4 Saatlik Tarama (SADECE 12:30 ve 17:30)
    if scan_4h and df_4h is not None and not df_4h.empty:
        try:
            s4h = evaluate_eco(df_4h, symbol, "4 Saat (4h)", ss_multi)
            if s4h:
                signals.append(s4h)
        except Exception:
            pass

    # 3. Günlük (1D) Tarama (SADECE 17:30)
    if scan_1d:
        try:
            df_1d = yf.download(symbol, period="1y", interval="1d", progress=False)
            s1d = evaluate_eco(df_1d, symbol, "Günlük (1D)", ss_multi)
            if s1d:
                signals.append(s1d)
        except Exception:
            pass

    return signals

def determine_scan_modes(now_tsi):
    """Hangi periyodun taranacağını saate göre KESİN OLARAK belirler."""
    h = now_tsi.hour
    m = now_tsi.minute
    
    if h >= 19 or h < 9 or (h == 9 and m < 45):
        return False, False, False

    scan_4h = (h == 12 and 15 <= m <= 35) or (h == 17 and 15 <= m <= 35)
    scan_1d = (h == 17 and 15 <= m <= 35)
    scan_1h = not (scan_4h or scan_1d)

    return scan_1h, scan_4h, scan_1d

def main():
    now_tsi = pd.Timestamp.now(tz="Europe/Istanbul")
    scan_1h, scan_4h, scan_1d = determine_scan_modes(now_tsi)
    
    if not scan_1h and not scan_4h and not scan_1d:
        print(f"Seans dışı saat ({now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    aktif_modlar = []
    if scan_1h: aktif_modlar.append("1 Saatlik")
    if scan_4h: aktif_modlar.append("4 Saatlik")
    if scan_1d: aktif_modlar.append("Günlük")
    
    print(f"BIST 100 Taraması Başlıyor (Saat: {now_tsi.strftime('%H:%M')} TSİ) -> Aktif Modlar: {', '.join(aktif_modlar)} ({len(BIST_TICKERS)} Hisse)...")
    all_signals = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_ticker, ticker, scan_1h, scan_4h, scan_1d): ticker for ticker in BIST_TICKERS}
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    all_signals.extend(results)
            except Exception as e:
                print(f"Hisse analiz hatası: {e}")

    toplam = 0
    for i, sig in enumerate(all_signals):
        success = send_telegram(sig)
        if success:
            toplam += 1
        if i < len(all_signals) - 1:
            time.sleep(1.5)

    print(f"BIST Taraması bitti! Bulunan toplam sinyal: {toplam}")

if __name__ == "__main__":
    main()
