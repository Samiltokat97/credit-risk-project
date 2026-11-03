"""Sayfa 1: Müşteri bilgilerini al, kredi kararını ve nedenlerini göster."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.yardimci import (
    RENK_ANA, RENK_IYI, RENK_KOTU, SENARYOLAR, TURKCE_AD,
    aciklayici_yukle, css_ekle, deger_yaz, metrik_kutu, model_yukle,
    risk_seviyesi, satir_olustur,
)

css_ekle()
model, sutunlar = model_yukle()

st.title("🏦 Kredi Değerlendirme")
st.caption("Bilgilerinizi girin; model başvurunuzun temerrüt (ödeyememe) riskini tahmin etsin.")

sol, sag = st.columns([1, 1.35], gap="large")

# ---------------------------------------------------------------- FORM
with sol:
    with st.form("basvuru"):
        st.subheader("Başvuru bilgileri")

        st.markdown("**Kişisel**")
        c1, c2 = st.columns(2)
        yas = c1.number_input("Yaş", 18, 100, 40)
        bakmakla = c2.number_input("Bakmakla yükümlü kişi", 0, 10, 1,
                                   help="Geçiminizi sağladığınız kişi sayısı (çocuk, eş, ebeveyn vb.)")

        st.markdown("**Gelir ve borç**")
        gelir_var = st.checkbox("Aylık gelirimi beyan ediyorum", value=True)
        gelir = st.number_input("Aylık net gelir (₺)", 0, 1_000_000, 30_000, step=1_000,
                                disabled=not gelir_var)
        borc = st.slider("Aylık borç ödemeleri gelirimin yüzde kaçı?", 0, 200, 35,
                         disabled=not gelir_var, format="%d%%",
                         help="Kredi taksitleri, kart asgari ödemeleri gibi tüm aylık borç ödemeleriniz / aylık geliriniz")
        limit = st.slider("Kart ve kredi limitlerimin yüzde kaçını kullanıyorum?", 0, 200, 30,
                          format="%d%%",
                          help="Kullanılan toplam bakiye / toplam limit. %100 üzeri limit aşımı demektir.")

        st.markdown("**Kredi geçmişi**")
        c3, c4 = st.columns(2)
        acik = c3.number_input("Açık kredi ve kart sayısı", 0, 40, 6)
        ipotek = c4.number_input("Konut/ipotek kredisi sayısı", 0, 10, 1)

        st.markdown("**Geçmiş gecikmeler** (son 2 yıl)")
        c5, c6, c7 = st.columns(3)
        g30 = c5.number_input("30-59 gün", 0, 20, 0, help="30-59 gün gecikmeli ödeme sayısı")
        g60 = c6.number_input("60-89 gün", 0, 20, 0, help="60-89 gün gecikmeli ödeme sayısı")
        g90 = c7.number_input("90+ gün", 0, 20, 0, help="90 gün ve üzeri gecikmeli ödeme sayısı")

        st.markdown("**Banka politikası**")
        tutum = st.radio(
            "Karar tutumu", list(SENARYOLAR), index=1, horizontal=True,
            help="Bankanın riskli bir müşteriyi kaçırmaya ne kadar tahammülü olduğunu belirler.",
        )
        st.caption(SENARYOLAR[tutum]["aciklama"])

        gonder = st.form_submit_button("Değerlendir", type="primary", use_container_width=True)

# Ilk acilista da bir sonuc gostermek icin form hic gonderilmese de hesapla
satir = satir_olustur(sutunlar, yas, bakmakla, gelir_var, gelir, borc, limit, acik, ipotek, g30, g60, g90)
skor = float(model.predict_proba(satir)[0, 1])
esik = SENARYOLAR[tutum]["esik"]
riskli = skor >= esik
seviye, seviye_renk = risk_seviyesi(skor)

# ---------------------------------------------------------------- SONUC
with sag:
    st.subheader("Sonuç")

    if riskli:
        st.markdown(
            '<div class="karar-kart kart-kotu"><h2>❌ Kredi önerilmiyor</h2>'
            "<p>Model, bu başvuruyu seçtiğiniz karar tutumuna göre <b>yüksek riskli</b> buluyor. "
            "Başvuru ek incelemeye alınmalı veya reddedilmelidir.</p></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="karar-kart kart-iyi"><h2>✅ Kredi verilebilir</h2>'
            "<p>Model, bu başvuruyu seçtiğiniz karar tutumuna göre <b>kabul edilebilir riskte</b> buluyor.</p></div>",
            unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1: metrik_kutu("Risk skoru (0-100)", f"{skor * 100:.0f}")
    with m2: metrik_kutu("Red eşiği", f"{esik * 100:.0f}")
    with m3: metrik_kutu("Risk seviyesi", seviye)

    # Risk gostergesi (gauge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=skor * 100,
        number={"font": {"size": 40}, "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": seviye_renk, "thickness": 0.3},
            "steps": [{"range": [0, esik * 100], "color": "#e8f5e9"},
                      {"range": [esik * 100, 100], "color": "#ffebee"}],
            "threshold": {"line": {"color": "#222", "width": 4}, "thickness": 0.85, "value": esik * 100},
        }))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Siyah çizgi red eşiğidir. İbre çizginin sağındaysa başvuru önerilmez. "
               "Skor bir olasılık değil, müşterileri riske göre sıralayan bir puandır.")

# ---------------------------------------------------------------- NEDENLER
st.divider()
st.subheader("Bu karar neden verildi?")

aciklayici = aciklayici_yukle()
shap_vals = np.asarray(aciklayici(satir).values)[0]
tablo = pd.DataFrame({
    "sutun": sutunlar,
    "etiket": [TURKCE_AD.get(s, s) for s in sutunlar],
    "deger": [deger_yaz(s, satir[s].iloc[0]) for s in sutunlar],
    "etki": shap_vals,
})
# Gelir yoksa teknik 'gelir_eksik' gibi sutunlar da kalir; sadece etkisi olanlar
tablo = tablo[tablo["etki"].abs() > 1e-6].copy()
tablo["mutlak"] = tablo["etki"].abs()
tablo = tablo.sort_values("mutlak", ascending=False).head(8).iloc[::-1]

g1, g2 = st.columns([1.4, 1], gap="large")
with g1:
    renkler = [RENK_KOTU if e > 0 else RENK_IYI for e in tablo["etki"]]
    cubuk = go.Figure(go.Bar(
        x=tablo["etki"], y=[f"{e}  ({d})" for e, d in zip(tablo["etiket"], tablo["deger"])],
        orientation="h", marker_color=renkler,
        hovertemplate="%{y}<br>Etki: %{x:.2f}<extra></extra>"))
    cubuk.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="← riski azaltır        riski artırır →",
        plot_bgcolor="white")
    cubuk.add_vline(x=0, line_color="#444", line_width=1)
    st.plotly_chart(cubuk, use_container_width=True)
with g2:
    st.markdown("**Nasıl okunur?**")
    st.markdown(
        "- 🔴 **Kırmızı** çubuklar riski **artıran** bilgilerdir.\n"
        "- 🟢 **Yeşil** çubuklar riski **azaltan** bilgilerdir.\n"
        "- Çubuk ne kadar uzunsa o bilginin etkisi o kadar büyüktür.\n"
        "- Parantez içi sizin girdiğiniz değerdir.")
    en_kotu = tablo[tablo["etki"] > 0].sort_values("etki", ascending=False).head(2)
    en_iyi = tablo[tablo["etki"] < 0].sort_values("etki").head(2)
    if len(en_kotu):
        st.markdown("**Riski en çok artıranlar:** " + ", ".join(f"{e} ({d})" for e, d in zip(en_kotu["etiket"], en_kotu["deger"])))
    if len(en_iyi):
        st.markdown("**Riski en çok azaltanlar:** " + ", ".join(f"{e} ({d})" for e, d in zip(en_iyi["etiket"], en_iyi["deger"])))

# ---------------------------------------------------------------- WHAT-IF
st.divider()
st.subheader("Ya şöyle olsaydı? (senaryo analizi)")
st.caption("Diğer tüm bilgiler sabit kalırken tek bir bilgiyi değiştirirseniz risk skoru nasıl değişir?")

sekme1, sekme2, sekme3 = st.tabs(["Limit kullanımı", "Borç / gelir oranı", "Yaş"])


def egri_ciz(sekme, ad, degerler, kurucu, mevcut, x_format):
    """Tek bir girdiyi degistirip skor egrisini cizer."""
    satirlar = pd.concat([kurucu(v) for v in degerler], ignore_index=True)
    skorlar = model.predict_proba(satirlar)[:, 1] * 100
    f = go.Figure()
    f.add_trace(go.Scatter(x=degerler, y=skorlar, mode="lines", line=dict(color=RENK_ANA, width=3),
                           name="Risk skoru"))
    f.add_hline(y=esik * 100, line_dash="dash", line_color=RENK_KOTU,
                annotation_text="Red eşiği", annotation_position="top left")
    f.add_trace(go.Scatter(x=[mevcut], y=[skor * 100], mode="markers",
                           marker=dict(size=14, color="#222"), name="Sizin değeriniz"))
    f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                    xaxis_title=ad, yaxis_title="Risk skoru (0-100)", yaxis_range=[0, 100],
                    legend=dict(orientation="h", y=1.12))
    f.update_xaxes(ticksuffix=x_format)
    sekme.plotly_chart(f, use_container_width=True)


with sekme1:
    egri_ciz(sekme1, "Limit kullanımı", list(range(0, 201, 5)),
             lambda v: satir_olustur(sutunlar, yas, bakmakla, gelir_var, gelir, borc, v, acik, ipotek, g30, g60, g90),
             limit, "%")
with sekme2:
    if gelir_var:
        egri_ciz(sekme2, "Borç ödemeleri / gelir", list(range(0, 201, 5)),
                 lambda v: satir_olustur(sutunlar, yas, bakmakla, gelir_var, gelir, v, limit, acik, ipotek, g30, g60, g90),
                 borc, "%")
    else:
        st.info("Gelir beyan edilmediği için bu analiz yapılamıyor.")
with sekme3:
    egri_ciz(sekme3, "Yaş", list(range(18, 91, 2)),
             lambda v: satir_olustur(sutunlar, v, bakmakla, gelir_var, gelir, borc, limit, acik, ipotek, g30, g60, g90),
             yas, "")

with st.expander("⚠️ Önemli not"):
    st.write("Bu araç, 2008 tarihli Kaggle veri seti ile eğitilmiş bir **eğitim/portföy projesidir**. "
             "Gerçek kredi kararı için kullanılamaz. Model sadece 10 temel bilgiyi görür; "
             "tek başına otomatik red kararı için yeterli değildir.")
