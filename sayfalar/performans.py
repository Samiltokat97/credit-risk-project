"""Sayfa 2: Modelin test verisindeki performansi ve esik secimi."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.yardimci import RAPOR_KLASORU, RENK_ANA, RENK_KOTU, SENARYOLAR, css_ekle, metrik_kutu

css_ekle()
st.title("📊 Model Performansı")
st.caption("Model, hiç görmediği test verisinde (%20) değerlendirildi. Sayılar notebook çıktılarından alınmıştır.")

# ------------------------------------------------------------ Ozet kutular
k1, k2, k3, k4 = st.columns(4)
with k1: metrik_kutu("Veri sayısı", "150.000")
with k2: metrik_kutu("Temerrüt oranı", "%6,7")
with k3: metrik_kutu("ROC-AUC (ayarlı)", "0,870")
with k4: metrik_kutu("PR-AUC (ayarlı)", "0,402")

st.divider()
sol, sag = st.columns(2, gap="large")

# ------------------------------------------------------------ Model karsilastirmasi
with sol:
    st.subheader("Üç modelin karşılaştırması")
    modeller = ["Lojistik regresyon\n(baseline)", "LightGBM", "XGBoost\n(ayarlı)"]
    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=modeller, y=[0.861, 0.866, 0.870], marker_color=RENK_ANA,
                text=["0,861", "0,866", "0,870"], textposition="outside")
    fig.add_bar(name="PR-AUC", x=modeller, y=[0.381, 0.395, 0.402], marker_color="#9fb3c8",
                text=["0,381", "0,395", "0,402"], textposition="outside")
    fig.update_layout(barmode="group", height=360, yaxis_range=[0, 1], plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Modeller birbirine çok yakın: sınırı model değil, verideki bilgi belirliyor.")

# ------------------------------------------------------------ Esik senaryolari
with sag:
    st.subheader("Karar tutumu ve red eşiği")
    adlar = list(SENARYOLAR)
    esikler = [SENARYOLAR[a]["esik"] * 100 for a in adlar]
    oranlar = [SENARYOLAR[a]["oran"] for a in adlar]
    fig2 = go.Figure(go.Bar(
        x=[f"{a}\n(1:{o})" for a, o in zip(adlar, oranlar)], y=esikler,
        marker_color=[RENK_ANA, "#3b7bbf", RENK_KOTU],
        text=[f"{e:.0f}" for e in esikler], textposition="outside"))
    fig2.update_layout(height=360, yaxis_range=[0, 100], yaxis_title="Red eşiği (risk skoru)",
                       plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Kaçırılan temerrüt ne kadar pahalıysa eşik o kadar düşer: banka daha fazla başvuruyu riskli sayar.")

st.info("**Neden tek bir eşik yok?** Bir temerrüdü kaçırmak (kayıp kredi), iyi bir müşteriyi reddetmekten (kaçan faiz geliri) "
        "çok daha pahalıdır. Eşik, bu maliyet oranına göre eğitim verisinin çapraz doğrulama tahminlerinde seçildi; "
        "test verisinde yalnızca bir kez uygulandı.")

# ------------------------------------------------------------ Metrik aciklamalari
st.subheader("Metrikler ne anlama geliyor?")
st.dataframe(pd.DataFrame({
    "Metrik": ["ROC-AUC", "PR-AUC", "Recall (duyarlılık)", "Precision (kesinlik)"],
    "Anlamı": [
        "Rastgele seçilen bir temerrütçünün, iyi bir müşteriden daha yüksek skor alma olasılığı.",
        "Nadir sınıfta (temerrüt) kesinlik-duyarlılık dengesini özetler; dengesiz veride en dürüst metrik.",
        "Gerçek temerrütçülerin yüzde kaçı yakalandı?",
        "'Riskli' dediklerimizin yüzde kaçı gerçekten temerrüte düştü?",
    ],
}), hide_index=True, use_container_width=True)
st.caption("Neden doğruluk (accuracy) değil? Herkese 'ödeyecek' desek bile %93,3 doğruluk elde edilir; bu yüzden anlamsızdır.")

# ------------------------------------------------------------ Rapor gorselleri
st.divider()
st.subheader("Model karşılaştırması ve açıklanabilirlik")
gorseller = [
    ("model_karsilastirma.png", "Modellerin karşılaştırması"),
    ("shap_onem.png", "Genel olarak en önemli bilgiler (SHAP)"),
    ("shap_beeswarm.png", "Her bilginin etkisinin yönü (SHAP beeswarm)"),
]
mevcut = [(RAPOR_KLASORU / d, b) for d, b in gorseller if (RAPOR_KLASORU / d).exists()]
if mevcut:
    sekmeler = st.tabs([b for _, b in mevcut])
    for sekme, (yol, b) in zip(sekmeler, mevcut):
        with sekme:
            st.image(str(yol), use_container_width=True)
else:
    st.warning("reports/ klasöründe grafik bulunamadı.")
