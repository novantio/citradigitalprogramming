import streamlit as st

st.set_page_config(page_title="Lung Segmentation App", layout="wide")

st.title("👋 Welcome to Lung Segmentation App")

st.markdown("""
Selamat datang!

Aplikasi ini memiliki fitur:
- 🫁 Segmentasi paru-paru dari citra X-ray  
- 🔍 Preview seluruh step pemrosesan dalam bentuk grid  
- 💾 Download hasil akhir  

Gunakan menu di kiri untuk pindah ke halaman **Detect Lung Image**.
""")



st.info("Silahkan pilih halaman di sidebar untuk memulai.")