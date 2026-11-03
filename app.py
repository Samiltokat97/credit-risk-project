"""Kredi Risk Değerlendirme - çok sayfalı Streamlit uygulaması.

Çalıştırmak için:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Kredi Risk Değerlendirme",
    page_icon="🏦",
    layout="wide",
)

# Sayfa tanimlari (sol menude bu sirayla gorunur)
sayfalar = [
    st.Page("sayfalar/degerlendirme.py", title="Kredi Değerlendirme", icon="🏦", default=True),
    st.Page("sayfalar/performans.py", title="Model Performansı", icon="📊"),
    st.Page("sayfalar/hakkinda.py", title="Proje Hakkında", icon="📘"),
]

st.navigation(sayfalar).run()

st.sidebar.divider()
st.sidebar.caption(
    "Bu uygulama bir portföy projesidir; gerçek kredi kararı vermez. "
    "Veri: Kaggle · Give Me Some Credit."
)
