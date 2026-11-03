"""Sayfa 3: Proje hakkinda - veri, yontem, sinirlamalar."""
import plotly.graph_objects as go
import streamlit as st

from src.yardimci import RENK_ANA, RENK_KOTU, css_ekle

css_ekle()
st.title("📘 Proje Hakkında")
st.write("Kaggle **Give Me Some Credit** verisiyle, bir müşterinin önümüzdeki 2 yıl içinde "
         "**ciddi gecikmeye (90+ gün) düşme riskini** tahmin eden uçtan uca bir makine öğrenmesi projesi.")

sol, sag = st.columns([1, 1], gap="large")
with sol:
    st.subheader("Veri")
    fig = go.Figure(go.Pie(labels=["Ödedi", "Temerrüt"], values=[93.3, 6.7], hole=0.6,
                           marker_colors=[RENK_ANA, RENK_KOTU], textinfo="label+percent"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("150.000 müşteri, 10 bilgi. Sınıflar çok dengesiz: temerrüt yalnızca %6,7.")
with sag:
    st.subheader("Yöntem")
    st.markdown(
        "1. **Keşifsel analiz:** dağılımlar, eksik değerler, aykırı değerler\n"
        "2. **Temizlik:** limit kullanımı en fazla %200; gelir yoksa eksik değer + 'gelir yok' bayrağı; hatalı gecikme kodları ayrı işaretlendi\n"
        "3. **Doğru değerlendirme:** katmanlı (stratified) bölme, pipeline ile sızıntısız ön işleme\n"
        "4. **Modeller:** lojistik regresyon (baseline) → LightGBM → XGBoost\n"
        "5. **Ayar:** RandomizedSearchCV, PR-AUC ile\n"
        "6. **Eşik:** iş maliyetine göre (kaçırılan temerrüt 3/5/10 kat pahalı)\n"
        "7. **Açıklanabilirlik:** SHAP")

st.divider()
a, b = st.columns(2, gap="large")
with a:
    st.subheader("Dürüst bulgular")
    st.markdown(
        "- Ayarlanmış XGBoost, basit lojistik regresyonla yakın sonuç verdi: **sınır model değil, veri.**\n"
        "- LightGBM ve ayarlı XGBoost küçük ama tutarlı bir üstünlük sağladı (PR-AUC 0,381 → 0,395 → 0,402).\n"
        "- Skorlar kalibre olasılık değil, sıralama puanıdır (sınıf ağırlığı skoru şişirir).")
with b:
    st.subheader("Sınırlılıklar")
    st.markdown(
        "- Veri eski ve tek bir kaynaktan; güncel müşteri davranışını yansıtmayabilir.\n"
        "- Sadece 10 temel değişken var; gelir kaynağı, meslek gibi bilgiler yok.\n"
        "- Adil kredi / ayrımcılık analizi yapılmadı (yaş bir girdi).\n"
        "- Gerçek karar için hukuki ve düzenleyici gereklilikler gerekir.")

st.divider()
st.subheader("Kullanılan teknolojiler")
st.markdown("`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Plotly` · `Streamlit` · `Git/GitHub`")
st.caption("Geliştirici: Samil Tokat · github.com/Samiltokat97/credit-risk-project")
