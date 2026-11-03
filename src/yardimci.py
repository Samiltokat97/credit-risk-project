"""Streamlit uygulamasi icin ortak yardimci fonksiyonlar ve sabitler."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Proje kok klasoru (src/ klasorunun bir ustu)
KOK = Path(__file__).resolve().parent.parent
MODEL_YOLU = KOK / "models" / "xgb_model.joblib"
SIRA_YOLU = KOK / "models" / "ozellik_sirasi.joblib"
RAPOR_KLASORU = KOK / "reports"

# Renk paleti (tum sayfalarda ayni renkler)
RENK_IYI = "#2e7d32"
RENK_KOTU = "#c62828"
RENK_UYARI = "#ef8f00"
RENK_ANA = "#1f4e79"

# Karar tutumu -> (kacirilan temerrut kac kat pahali, model skoru esigi)
# Esikler notebook'ta OOF tahminleri uzerinde maliyete gore secildi.
SENARYOLAR = {
    "Rahat": {"oran": 3, "esik": 0.80,
              "aciklama": "Bir temerrüdü kaçırmak, bir iyi müşteriyi reddetmekten 3 kat pahalı."},
    "Dengeli": {"oran": 5, "esik": 0.70,
                "aciklama": "Bir temerrüdü kaçırmak, bir iyi müşteriyi reddetmekten 5 kat pahalı."},
    "Temkinli": {"oran": 10, "esik": 0.60,
                 "aciklama": "Bir temerrüdü kaçırmak, bir iyi müşteriyi reddetmekten 10 kat pahalı."},
}

TURKCE_AD = {
    "RevolvingUtilizationOfUnsecuredLines": "Kredi limiti kullanımı",
    "age": "Yaş",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 gün gecikme sayısı",
    "DebtRatio": "Borç / gelir oranı",
    "MonthlyIncome": "Aylık gelir",
    "NumberOfOpenCreditLinesAndLoans": "Açık kredi ve kart sayısı",
    "NumberOfTimes90DaysLate": "90+ gün gecikme sayısı",
    "NumberRealEstateLoansOrLines": "Konut/ipotek kredisi sayısı",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 gün gecikme sayısı",
    "NumberOfDependents": "Bakmakla yükümlü kişi sayısı",
    "gecikme_ozel_kod": "Özel gecikme kodu",
    "gelir_eksik": "Gelir bilgisi yok",
}

# Oran olan sutunlar yuzde olarak gosterilir
ORAN_SUTUNLARI = {"RevolvingUtilizationOfUnsecuredLines", "DebtRatio"}


@st.cache_resource(show_spinner="Model yükleniyor...")
def model_yukle():
    """Egitilmis modeli ve ozellik sirasini diskten bir kez yukler."""
    model = joblib.load(MODEL_YOLU)
    sutunlar = joblib.load(SIRA_YOLU)
    return model, list(sutunlar)


@st.cache_resource(show_spinner=False)
def aciklayici_yukle():
    """SHAP aciklayicisini bir kez olusturur."""
    import shap
    model, _ = model_yukle()
    return shap.TreeExplainer(model)


def satir_olustur(sutunlar, yas, bakmakla, gelir_var, gelir, borc_yuzde, limit_yuzde,
                  acik_kredi, ipotek, g30, g60, g90):
    """Form girdilerinden modelin bekledigi tek satirlik DataFrame'i kurar.

    Notebook'taki temizlik kurallari birebir uygulanir:
    - limit kullanimi en fazla 2 (yuzde 200) ile sinirlanir
    - gelir yoksa MonthlyIncome ve DebtRatio NaN, gelir_eksik = 1
    """
    limit = min(limit_yuzde / 100, 2.0)
    veri = {
        "RevolvingUtilizationOfUnsecuredLines": limit,
        "age": yas,
        "NumberOfTime30-59DaysPastDueNotWorse": g30,
        "DebtRatio": (borc_yuzde / 100) if gelir_var else float("nan"),
        "MonthlyIncome": float(gelir) if gelir_var else float("nan"),
        "NumberOfOpenCreditLinesAndLoans": acik_kredi,
        "NumberOfTimes90DaysLate": g90,
        "NumberRealEstateLoansOrLines": ipotek,
        "NumberOfTime60-89DaysPastDueNotWorse": g60,
        "NumberOfDependents": bakmakla,
        "gecikme_ozel_kod": 0,
        "gelir_eksik": 0 if gelir_var else 1,
    }
    return pd.DataFrame([veri])[sutunlar]


def deger_yaz(sutun, v):
    """Bir ozellik degerini kullaniciya anlasilir bicimde yazar."""
    if pd.isna(v):
        return "Belirtilmedi"
    if sutun in ORAN_SUTUNLARI:
        return f"%{v * 100:.0f}"
    if sutun == "MonthlyIncome":
        return f"{v:,.0f} ₺".replace(",", ".")
    return f"{v:.0f}"


def risk_seviyesi(skor):
    """Model skorunu 3 kademeli, anlasilir bir risk etiketine cevirir."""
    if skor < 0.5:
        return "Düşük", RENK_IYI
    if skor < 0.7:
        return "Orta", RENK_UYARI
    return "Yüksek", RENK_KOTU


def css_ekle():
    """Uygulamaya ortak kart/baslik stilini ekler."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1200px;}
        .karar-kart {border-radius: 14px; padding: 22px 26px; margin-bottom: 8px;}
        .karar-kart h2 {margin: 0 0 6px 0; font-size: 1.7rem;}
        .karar-kart p {margin: 0; font-size: 1.0rem; line-height: 1.5;}
        .kart-iyi {background:#e8f5e9; border-left: 8px solid #2e7d32; color:#1b5e20;}
        .kart-kotu {background:#ffebee; border-left: 8px solid #c62828; color:#8e1b1b;}
        .metrik {background:#f3f6fa; border-radius:12px; padding:14px 18px;}
        .metrik .e {font-size:0.85rem; color:#5b6b7b;}
        .metrik .d {font-size:1.7rem; font-weight:700; color:#1f4e79;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metrik_kutu(etiket, deger):
    """Kucuk, tutarli gorunumlu bir metrik kutusu cizer."""
    st.markdown(
        f'<div class="metrik"><div class="e">{etiket}</div><div class="d">{deger}</div></div>',
        unsafe_allow_html=True,
    )
