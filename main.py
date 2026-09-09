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

# Borsa İstanbul'daki Tüm Hisseler (552 Hisse Tam Liste)
BIST_TICKERS = sorted(list(set([
    "A1CAP.IS", "ACSEL.IS", "ADEL.IS", "ADESE.IS", "ADGYO.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS", "AGHOL.IS", "AGROT.IS",
    "AGYO.IS", "AHGAZ.IS", "AHSGY.IS", "AKBNK.IS", "AKCNS.IS", "AKENR.IS", "AKFGY.IS", "AKFYE.IS", "AKGRT.IS", "AKMGY.IS",
    "AKSA.IS", "AKSEN.IS", "AKSGY.IS", "AKSUE.IS", "AKTIF.IS", "ALARK.IS", "ALBRK.IS", "ALCAR.IS", "ALCTL.IS", "ALFAS.IS",
    "ALGYO.IS", "ALKA.IS", "ALKIM.IS", "ALMAD.IS", "ALTNY.IS", "ALVES.IS", "ANELE.IS", "ANGEN.IS", "ANHYT.IS", "ANSGR.IS",
    "ARASE.IS", "ARCLK.IS", "ARDYZ.IS", "ARENA.IS", "ARSAN.IS", "ARTMS.IS", "ARZUM.IS", "ASELS.IS", "ASGYO.IS", "ASTOR.IS",
    "ASUZU.IS", "ATAGY.IS", "ATAKP.IS", "ATATP.IS", "ATEKS.IS", "ATSYH.IS", "AVGYO.IS", "AVHOL.IS", "AVOD.IS", "AVPGY.IS",
    "AVTUR.IS", "AYCES.IS", "AYDEM.IS", "AYEN.IS", "AYES.IS", "AYGAZ.IS", "AZTEK.IS", "BAGFS.IS", "BAKAB.IS", "BALAT.IS",
    "BANVT.IS", "BARMA.IS", "BASCM.IS", "BASGZ.IS", "BAYRK.IS", "BEGYO.IS", "BENGV.IS", "BERA.IS", "BEYAZ.IS", "BFREN.IS",
    "BIENY.IS", "BIGCH.IS", "BIMAS.IS", "BINHO.IS", "BIOEN.IS", "BIZIM.IS", "BJKAS.IS", "BLCYT.IS", "BMSCH.IS", "BMSTL.IS",
    "BNTAS.IS", "BOBET.IS", "BORLS.IS", "BORSK.IS", "BOSSA.IS", "BRCEG.IS", "BRCVO.IS", "BRISA.IS", "BRKO.IS", "BRKSN.IS",
    "BRMEN.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "BUCIM.IS", "BURCE.IS", "BURVA.IS", "BVSAN.IS", "BYDNR.IS",
    "CANTE.IS", "CASA.IS", "CATES.IS", "CCOLA.IS", "CELHA.IS", "CEMAS.IS", "CEMTS.IS", "CEMZY.IS", "CEOEM.IS", "CIMSA.IS",
    "CLEBI.IS", "CMBTN.IS", "CMENT.IS", "CONSE.IS", "COSMO.IS", "CRDFA.IS", "CRFSA.IS", "CUSAN.IS", "CVKMD.IS", "CWENE.IS",
    "DAGI.IS", "DAGHL.IS", "DAPGM.IS", "DARDL.IS", "DENGE.IS", "DERHL.IS", "DERIM.IS", "DESA.IS", "DESPC.IS", "DEVA.IS",
    "DGATE.IS", "DGGYO.IS", "DGNMO.IS", "DIRIT.IS", "DITAS.IS", "DMRGD.IS", "DMSAS.IS", "DNISI.IS", "DOAS.IS", "DOBUR.IS",
    "DOCO.IS", "DOFER.IS", "DOGUB.IS", "DOHOL.IS", "DOKTA.IS", "DURDO.IS", "DURKN.IS", "DYOBY.IS", "DZGYO.IS", "EBEBK.IS",
    "ECZYT.IS", "EDATA.IS", "EDIP.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EKGYO.IS", "EKIZ.IS", "EKOS.IS",
    "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "EMNIS.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "ENSRI.IS", "ENTRA.IS", "EPLAS.IS",
    "ERBOS.IS", "ERCB.IS", "EREGL.IS", "ERSU.IS", "ESCAR.IS", "ESCOM.IS", "ESEN.IS", "ETILR.IS", "ETYAT.IS", "EUHOL.IS",
    "EUKYO.IS", "EUPWR.IS", "EUREN.IS", "EUYO.IS", "EYGYO.IS", "FADE.IS", "FENER.IS", "FLAP.IS", "FMIZP.IS", "FONET.IS",
    "FORMT.IS", "FORTE.IS", "FRIGO.IS", "FROTO.IS", "FZLGY.IS", "GARAN.IS", "GARFA.IS", "GEDIK.IS", "GEDZA.IS", "GENIL.IS",
    "GENTS.IS", "GEREL.IS", "GESAN.IS", "GIPTA.IS", "GLBMD.IS", "GLCVY.IS", "GLRYH.IS", "GLYHO.IS", "GMTAS.IS", "GOKNR.IS",
    "GOLTS.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS", "GRSEL.IS", "GRTHO.IS", "GSDDE.IS", "GSDHO.IS", "GSRAY.IS", "GUBRF.IS",
    "GWIND.IS", "GZNMI.IS", "HALKB.IS", "HATEK.IS", "HATSN.IS", "HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "HOROZ.IS",
    "HRKET.IS", "HTTBT.IS", "HUBVC.IS", "HUNER.IS", "HURGZ.IS", "ICBCT.IS", "ICUGS.IS", "IDGYO.IS", "IEYHO.IS", "IHAAS.IS",
    "IHEVA.IS", "IHGZT.IS", "IHLAS.IS", "IHLGM.IS", "IHYAY.IS", "IMASM.IS", "INDES.IS", "INFO.IS", "INGRM.IS", "INVES.IS",
    "IPEKE.IS", "ISATR.IS", "ISBIR.IS", "ISBTR.IS", "ISCTR.IS", "ISDMR.IS", "ISFIN.IS", "ISGSY.IS", "ISGYO.IS", "ISKPL.IS",
    "ISKUR.IS", "ISMEN.IS", "ISYAT.IS", "IZENR.IS", "IZFAS.IS", "IZINV.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS", "KAREL.IS",
    "KARSN.IS", "KARTN.IS", "KARYA.IS", "KATMR.IS", "KAYSE.IS", "KBORU.IS", "KCAER.IS", "KCHOL.IS", "KFEIN.IS", "KGYO.IS",
    "KIMMR.IS", "KLGYO.IS", "KLKIM.IS", "KLMSN.IS", "KLNMA.IS", "KLRHO.IS", "KLSER.IS", "KLSYN.IS", "KMPUR.IS", "KNFRT.IS",
    "KOCMT.IS", "KONKA.IS", "KONTR.IS", "KONYA.IS", "KOPOL.IS", "KORDS.IS", "KOTON.IS", "KOZAA.IS", "KOZAL.IS", "KRDMA.IS",
    "KRDMB.IS", "KRDMD.IS", "KRGYO.IS", "KRONT.IS", "KRPLS.IS", "KRSTL.IS", "KRTEK.IS", "KRVGD.IS", "KSTUR.IS", "KTLEV.IS",
    "KTSKR.IS", "KUTPO.IS", "KUVVA.IS", "KUYAS.IS", "KZBGY.IS", "KZGYO.IS", "LIDER.IS", "LIDFA.IS", "LILAK.IS", "LINK.IS",
    "LKMNH.IS", "LMKDC.IS", "LOGOS.IS", "LRSHO.IS", "LUKSK.IS", "LYDHO.IS", "MAALT.IS", "MACKO.IS", "MAGEN.IS", "MAKIM.IS",
    "MAKTK.IS", "MANAS.IS", "MARBL.IS", "MARKA.IS", "MARTI.IS", "MAVI.IS", "MEDTR.IS", "MEGAP.IS", "MEGMT.IS", "MEKAG.IS",
    "MEPET.IS", "MERCN.IS", "MERIT.IS", "MERKO.IS", "METRO.IS", "METUR.IS", "MEYHO.IS", "MGENR.IS", "MGROS.IS", "MIATK.IS",
    "MIPAZ.IS", "MMCAS.IS", "MNDRS.IS", "MNDTR.IS", "MOBTL.IS", "MOGAN.IS", "MOPAS.IS", "MPARK.IS", "MRGYO.IS", "MRSHL.IS",
    "MSGYO.IS", "MTRKS.IS", "MTRYO.IS", "MZHLD.IS", "NATEN.IS", "NETAS.IS", "NIBAS.IS", "NTGAZ.IS", "NTHOL.IS", "NUGYO.IS",
    "NUHCM.IS", "OBAMS.IS", "OBASE.IS", "ODAS.IS", "OFSYM.IS", "ONCSM.IS", "ORCAY.IS", "ORGE.IS", "ORMA.IS", "OSMEN.IS",
    "OSTIM.IS", "OTKAR.IS", "OTTO.IS", "OYAKC.IS", "OYAYO.IS", "OYLUM.IS", "OYYAT.IS", "OZATD.IS", "OZGYO.IS", "OZKGY.IS",
    "OZRDN.IS", "OZSUB.IS", "PAGYO.IS", "PAMEL.IS", "PAPIL.IS", "PARSN.IS", "PASEU.IS", "PATEK.IS", "PCILT.IS", "PEKGY.IS",
    "PENGD.IS", "PENTA.IS", "PETKM.IS", "PETUN.IS", "PGSUS.IS", "PINSU.IS", "PKART.IS", "PKENT.IS", "PLTUR.IS", "PNLSN.IS",
    "PNSUT.IS", "POLHO.IS", "POLTK.IS", "PRDGS.IS", "PRKAB.IS", "PRKME.IS", "PRZMA.IS", "PSDTC.IS", "PSGYO.IS", "QNBFB.IS",
    "QNBFL.IS", "QUAGR.IS", "RALYH.IS", "RAYSG.IS", "REEDR.IS", "RNPOL.IS", "RODRG.IS", "ROYAL.IS", "RTALB.IS", "RUBNS.IS",
    "RYGYO.IS", "RYSAS.IS", "SAFKR.IS", "SAHOL.IS", "SAMAT.IS", "SANEL.IS", "SANFM.IS", "SANKO.IS", "SARKY.IS", "SARTN.IS",
    "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SEGYO.IS", "SEKFK.IS", "SEKUR.IS", "SELEC.IS", "SELVA.IS", "SEYKM.IS", "SILVR.IS",
    "SISE.IS", "SKBNK.IS", "SKTAS.IS", "SKYMD.IS", "SMART.IS", "SMRTG.IS", "SNAYS.IS", "SNICA.IS", "SNKRN.IS", "SNPAM.IS",
    "SODSN.IS", "SOKE.IS", "SOKM.IS", "SONME.IS", "SRVGY.IS", "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "SUWEN.IS", "TABGD.IS",
    "TARKM.IS", "TATEN.IS", "TATGD.IS", "TAVHL.IS", "TBORG.IS", "TCELL.IS", "TDGYO.IS", "TEKTU.IS", "TERA.IS", "TEZOL.IS",
    "TETMT.IS", "TGSAS.IS", "THYAO.IS", "TKFEN.IS", "TKNSA.IS", "TLMAN.IS", "TMPOL.IS", "TMSN.IS", "TNZTP.IS", "TOASO.IS",
    "TRCAS.IS", "TRGYO.IS", "TRILC.IS", "TSGYO.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS", "TTRAK.IS", "TUCLK.IS", "TUKAS.IS",
    "TUPRS.IS", "TUREX.IS", "TURGG.IS", "TURSG.IS", "UFUK.IS", "ULAS.IS", "ULKER.IS", "ULUFA.IS", "ULUSE.IS", "ULUUN.IS",
    "UNLU.IS", "USAK.IS", "VAKBN.IS", "VAKFN.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VERTU.IS", "VERUS.IS", "VESBE.IS",
    "VESTL.IS", "VKFYO.IS", "VKGYO.IS", "VKING.IS", "VRGYO.IS", "YAPRK.IS", "YATAS.IS", "YAYLA.IS", "YGGYO.IS", "YGYO.IS",
    "YEOTK.IS", "YESIL.IS", "YKBNK.IS", "YKSLN.IS", "YONGA.IS", "YUNSA.IS", "YYAPI.IS", "YYLGD.IS", "ZEDUR.IS", "ZELOT.IS",
    "ZOREN.IS", "ZRGYO.IS"
])))

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

        puan = 1
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

def make_bist_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """TradingView BIST 4 saatlik mumlarını (09:00-13:00 ve 13:00-18:10) oluşturur."""
    df = clean_df(df_1h)
    if df.empty or len(df) < 15:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    # TSİ saatine göre 13:00 ayrımı
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

def evaluate_eco(df: pd.DataFrame, symbol: str, tf_label: str):
    """Pine Script ECO göstergesini saf mantığıyla hesaplar."""
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

    c_curr = float(stoch.iloc[-1])
    c_prev = float(stoch.iloc[-2])
    c_prev2 = float(stoch.iloc[-3]) if len(stoch) >= 3 else c_prev

    hisse_adi = symbol.replace('.IS', '')

    # 1. AL SİNYALİ KONTROLÜ
    if (c_prev < 10 and c_curr > 10) or (c_prev2 < 10 and c_prev > 10):
        if c_prev < 10 and c_curr > 10:
            target_idx = -1
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        else:
            target_idx = -2
            durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])

        if "Saat" in tf_label:
            if getattr(candle_time, 'minute', 0) != 0:
                candle_time = candle_time + pd.Timedelta(minutes=30)
            time_str = candle_time.strftime('%H:00')
        else:
            time_str = candle_time.strftime('%d.%m.%Y')

        # TradingView Doğrudan Grafik Linki
        tv_link = f"https://tr.tradingview.com/chart/?symbol=BIST:{hisse_adi}"

        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)
        sup, res, d_sup, d_res = calculate_strong_sr(df, target_idx)
        hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, "BUY", kanal_renk, nokta_renk, d_sup, d_res)

        sr_metni = ""
        if sup is not None and res is not None:
            sr_metni = (
                f"\n\n<b>🎯 Kuvvetli Destek & Direnç:</b>\n"
                f"▫️ <b>Ana Destek:</b> {sup:.2f} TL (<code>{d_sup:+.1f}%</code>)\n"
                f"▫️ <b>Ana Direnç:</b> {res:.2f} TL (<code>{d_res:+.1f}%</code>)"
            )

        return (
            f"🟢 <b>BIST AL SİNYALİ (Evan Cabral - ECO)</b>\n\n"
            f"📌 <b>Hisse:</b> <a href=\"{tv_link}\">#{hisse_adi}</a> <i>(Grafiği Aç)</i>\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> {candle_price:.2f} TL\n"
            f"📊 <b>DMI-Stoch:</b> {stoch.iloc[target_idx]:.1f} (Önceki: {stoch.iloc[target_idx-1]:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch 10 seviyesini yukarı kesti ('B')\n\n"
            f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
            f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}"
            f"{sr_metni}"
        )

    # 2. SAT SİNYALİ KONTROLÜ
    elif (c_prev > 90 and c_curr < 90) or (c_prev2 > 90 and c_prev < 90):
        if c_prev > 90 and c_curr < 90:
            target_idx = -1
            durum_metni = "⚠️ CANLI MUM (Anlık Sinyal)"
        else:
            target_idx = -2
            durum_metni = "✅ KAPANMIŞ MUM (Kesinleşmiş)"

        candle_time = df.index[target_idx]
        candle_price = float(close.iloc[target_idx])

        if "Saat" in tf_label:
            if getattr(candle_time, 'minute', 0) != 0:
                candle_time = candle_time + pd.Timedelta(minutes=30)
            time_str = candle_time.strftime('%H:00')
        else:
            time_str = candle_time.strftime('%d.%m.%Y')

        # TradingView Doğrudan Grafik Linki
        tv_link = f"https://tr.tradingview.com/chart/?symbol=BIST:{hisse_adi}"

        kanal_renk, nokta_renk = calculate_slingshot(df, target_idx)
        sup, res, d_sup, d_res = calculate_strong_sr(df, target_idx)
        hacim_metni, skor_metni = calculate_score_and_rvol(df, target_idx, "SELL", kanal_renk, nokta_renk, d_sup, d_res)

        sr_metni = ""
        if sup is not None and res is not None:
            sr_metni = (
                f"\n\n<b>🎯 Kuvvetli Destek & Direnç:</b>\n"
                f"▫️ <b>Ana Destek:</b> {sup:.2f} TL (<code>{d_sup:+.1f}%</code>)\n"
                f"▫️ <b>Ana Direnç:</b> {res:.2f} TL (<code>{d_res:+.1f}%</code>)"
            )

        return (
            f"🔴 <b>BIST SAT SİNYALİ (Evan Cabral - ECO)</b>\n\n"
            f"📌 <b>Hisse:</b> <a href=\"{tv_link}\">#{hisse_adi}</a> <i>(Grafiği Aç)</i>\n"
            f"⏱ <b>Zaman Dilimi:</b> {tf_label}\n"
            f"🕒 <b>Mum Saati:</b> <code>{time_str}</code> (TSİ)\n"
            f"⚡ <b>Mum Durumu:</b> {durum_metni}\n"
            f"💵 <b>Fiyat:</b> {candle_price:.2f} TL\n"
            f"📊 <b>DMI-Stoch:</b> {stoch.iloc[target_idx]:.1f} (Önceki: {stoch.iloc[target_idx-1]:.1f})\n"
            f"🎯 <b>Tetikleyici:</b> DMI-Stoch 90 seviyesini aşağı kesti ('S')\n\n"
            f"<b>⭐ Sinyal Güven Puanı:</b> {skor_metni}\n"
            f"<b>📊 Hacim Gücü:</b> {hacim_metni}\n\n"
            f"<b>📈 Trend Teyitleri (SlingShot):</b>\n"
            f"▫️ <b>Düz Trend Kanalı:</b> {kanal_renk}\n"
            f"▫️ <b>Noktasal Trend:</b> {nokta_renk}"
            f"{sr_metni}"
        )

    return None

def analyze_ticker(symbol: str, scan_1h: bool, scan_4h: bool, scan_1d: bool):
    """Sadece planlanan saatteki ilgili periyotları analiz eder."""
    signals = []
    df_1h = None
    
    if scan_1h or scan_4h:
        try:
            df_1h = yf.download(symbol, period="2mo", interval="1h", progress=False)
        except Exception:
            pass

    # 1. 1 Saatlik Tarama
    if scan_1h and df_1h is not None and not df_1h.empty:
        try:
            s1h = evaluate_eco(df_1h, symbol, "1 Saat (1h)")
            if s1h:
                signals.append(s1h)
        except Exception:
            pass

    # 2. 4 Saatlik Tarama (SADECE 12:30 ve 17:30)
    if scan_4h and df_1h is not None and not df_1h.empty:
        try:
            df_4h = make_bist_4h(df_1h)
            s4h = evaluate_eco(df_4h, symbol, "4 Saat (4h)")
            if s4h:
                signals.append(s4h)
        except Exception:
            pass

    # 3. Günlük (1D) Tarama (SADECE 17:30)
    if scan_1d:
        try:
            df_1d = yf.download(symbol, period="1y", interval="1d", progress=False)
            s1d = evaluate_eco(df_1d, symbol, "Günlük (1D)")
            if s1d:
                signals.append(s1d)
        except Exception:
            pass

    return signals

def determine_scan_modes(now_tsi):
    """Hangi periyodun taranacağını saate göre KESİN OLARAK belirler."""
    h = now_tsi.hour
    m = now_tsi.minute
    
    # Seans dışı (18:15 sonrası veya 09:30 öncesi) kesinlikle çalışmaz
    if h >= 19 or h < 9 or (h == 9 and m < 45):
        return False, False, False

    # 1. 4 Saatlik Tarama: SADECE saat 12:30 ve 17:30 civarında (dakika 15 ile 35 arası)
    scan_4h = (h == 12 and 15 <= m <= 35) or (h == 17 and 15 <= m <= 35)

    # 2. Günlük Tarama: SADECE saat 17:30 civarında (dakika 15 ile 35 arası)
    scan_1d = (h == 17 and 15 <= m <= 35)

    # 3. Saatlik Tarama: 12:30 ve 17:30 DIŞINDAKİ tüm seans taramalarında SADECE 1 Saatlik çalışır!
    # Saat 16:00, 16:45 veya başka bir saatte çalışsa bile 4h ve 1d KESİNLİKLE FALSE kalır!
    scan_1h = not (scan_4h or scan_1d)

    return scan_1h, scan_4h, scan_1d

def main():
    now_tsi = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=3)
    scan_1h, scan_4h, scan_1d = determine_scan_modes(now_tsi)
    
    # Eğer seans dışıysa hiçbir şey yapmadan anında kapanır
    if not scan_1h and not scan_4h and not scan_1d:
        print(f"Seans dışı saat ({now_tsi.strftime('%H:%M')} TSİ). Tarama yapılmıyor.")
        return

    aktif_modlar = []
    if scan_1h: aktif_modlar.append("1 Saatlik")
    if scan_4h: aktif_modlar.append("4 Saatlik")
    if scan_1d: aktif_modlar.append("Günlük")
    
    print(f"BIST Taraması Başlıyor (Saat: {now_tsi.strftime('%H:%M')} TSİ) -> Aktif Modlar: {', '.join(aktif_modlar)} ({len(BIST_TICKERS)} Hisse)...")
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

    # Sinyaller Telegram flood limitine takılmadan 1.5 saniye arayla güvenle gönderilir
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
