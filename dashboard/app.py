import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ============================================================================
# PATH CONFIGURATION & SETUP
# ============================================================================

# Get project directory (parent of dashboard folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Define all required file paths
PATHS = {
    'model': os.path.join(PROJECT_DIR, 'savemodel', 'model_random_forest.pkl'),
    'encoder': os.path.join(PROJECT_DIR, 'savemodel', 'label_encoder.pkl'),
    'data': os.path.join(PROJECT_DIR, 'data', 'dataset.csv'),
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL & DATA LOADING
# ============================================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load(PATHS['model'])
        encoder = joblib.load(PATHS['encoder'])
        return model, encoder
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(PATHS['data'])
        return df
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.stop()

# Load resources
model, encoder = load_model()
df_data = load_dataset()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def make_prediction(ipk, absen, kegiatan):
    input_data = np.array([[ipk, absen, kegiatan]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    hasil = encoder.inverse_transform([prediction])[0]
    confidence = probability[prediction] * 100
    return hasil, confidence, probability

def get_recommendations(ipk, absen, kegiatan):
    recommendations = []
    if ipk < 3.0:
        recommendations.append(" IPK perlu ditingkatkan (saat ini < 3.0)")
    elif ipk < 3.5:
        recommendations.append(" IPK masih bisa ditingkatkan (target ≥ 3.5)")
    else:
        recommendations.append(" IPK sudah baik (≥ 3.5)")
    
    if absen > 10:
        recommendations.append(" Absensi terlalu tinggi, kurangi ketidakhadiran")
    elif absen > 5:
        recommendations.append(" Absensi perlu dikurangi (target ≤ 5 hari)")
    else:
        recommendations.append(" Absensi terjaga dengan baik (≤ 5 hari)")
    
    if kegiatan < 3:
        recommendations.append(" Tingkatkan partisipasi dalam kegiatan/organisasi")
    else:
        recommendations.append(" Partisipasi kegiatan sudah baik")
    
    return recommendations

# ============================================================================
# SIDEBAR
# ============================================================================

st.title(" Sistem Prediksi Kelulusan Mahasiswa Tepat Waktu")
st.markdown("---")

with st.sidebar:
    st.header("ℹ Tentang Sistem")
    st.info(
        """
        **Sistem Prediksi Kelulusan Mahasiswa Tepat Waktu**
        
        Prediksi berbasis Machine Learning untuk memprediksi kemungkinan 
        seorang mahasiswa akan lulus tepat waktu berdasarkan:
        
        - **IPK** (Indeks Prestasi Kumulatif)
        - **Absen** (Jumlah ketidakhadiran)
        - **Kegiatan** (Partisipasi dalam kegiatan/organisasi)
        
        **Model:** Random Forest Classifier
        **Akurasi:** ~90%
        """
    )
    st.markdown("---")
    st.subheader(" Project Info")
    st.caption("Student Graduation Prediction System")
    st.caption("© 2024 - Data Science Project")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs([" Prediksi Individu", " Prediksi Batch", " Analisis Data"])

# TAB 1: INDIVIDUAL PREDICTION
with tab1:
    st.header("Prediksi Seorang Mahasiswa")
    st.write("Masukkan data mahasiswa untuk memprediksi status kelulusannya")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ipk = st.number_input(
            "IPK (0.0 - 4.0)",
            min_value=0.0,
            max_value=4.0,
            value=3.5,
            step=0.05,
            help="Indeks Prestasi Kumulatif"
        )
    
    with col2:
        absen = st.number_input(
            "Jumlah Absen (hari)",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
            help="Jumlah hari ketidakhadiran"
        )
    
    with col3:
        kegiatan = st.number_input(
            "Partisipasi Kegiatan (0-10)",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Skor partisipasi dalam kegiatan (0=min, 10=max)"
        )
    
    if st.button(" PREDIKSI", use_container_width=True, type="primary"):
        hasil, confidence, probability = make_prediction(ipk, absen, kegiatan)
        
        st.markdown("---")
        st.subheader(" Hasil Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasil == "Ya":
                st.metric("Status", " LULUS", delta="Positif", delta_color="inverse")
            else:
                st.metric("Status", " TIDAK LULUS", delta="Perlu Perhatian", delta_color="off")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            st.metric("Model", "Random Forest")
        
        st.markdown("---")
        st.subheader(" Probabilitas")
        
        prob_lulus = probability[1] * 100
        prob_tidak_lulus = probability[0] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f" **Lulus Tepat Waktu:** {prob_lulus:.2f}%")
            st.progress(prob_lulus / 100)
        with col2:
            st.write(f" **Tidak Lulus Tepat Waktu:** {prob_tidak_lulus:.2f}%")
            st.progress(prob_tidak_lulus / 100)
        
        st.markdown("---")
        st.subheader(" Rekomendasi")
        
        recommendations = get_recommendations(ipk, absen, kegiatan)
        for rec in recommendations:
            st.write(rec)

# TAB 2: BATCH PREDICTION
with tab2:
    st.header("Prediksi Batch (Multiple Mahasiswa)")
    st.write("Upload file CSV untuk melakukan prediksi pada banyak mahasiswa sekaligus")
    
    uploaded_file = st.file_uploader(
        " Pilih file CSV",
        type=['csv'],
        help="File harus memiliki kolom: ipk, absen, kegiatan"
    )
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            required_cols = ['ipk', 'absen', 'kegiatan']
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            
            if missing_cols:
                st.error(f" Kolom yang hilang: {', '.join(missing_cols)}")
            else:
                st.write("** Data Input:**")
                st.dataframe(df_input, use_container_width=True)
                
                if st.button(" PREDIKSI BATCH", use_container_width=True, type="primary"):
                    try:
                        X = df_input[['ipk', 'absen', 'kegiatan']]
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)
                        
                        df_results = df_input.copy()
                        df_results['Prediksi'] = encoder.inverse_transform(predictions)
                        df_results['Confidence'] = (probabilities.max(axis=1) * 100).round(2)
                        df_results['Probability_Lulus'] = (probabilities[:, 1] * 100).round(2)
                        
                        st.write("** Hasil Prediksi:**")
                        st.dataframe(df_results, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader(" Ringkasan Hasil")
                        
                        lulus_count = (predictions == 1).sum()
                        tidak_lulus_count = (predictions == 0).sum()
                        rata_confidence = df_results['Confidence'].mean()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Prediksi", len(df_results))
                        with col2:
                            st.metric("Lulus Tepat Waktu", lulus_count)
                        with col3:
                            st.metric("Tidak Lulus Tepat Waktu", tidak_lulus_count)
                        with col4:
                            st.metric("Avg Confidence", f"{rata_confidence:.1f}%")
                        
                        st.markdown("---")
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label=" Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name=f"prediksi_lulus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f" Error dalam prediksi: {str(e)}")
        except Exception as e:
            st.error(f" Error membaca file: {str(e)}")

# TAB 3: DATA ANALYSIS
with tab3:
    st.header(" Analisis Data & Model")
    
    st.subheader(" Statistik Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Mahasiswa", len(df_data))
    
    with col2:
        lulus = (df_data['lulus_tepat_waktu'] == 'Ya').sum()
        lulus_pct = (lulus / len(df_data) * 100)
        st.metric("Lulus Tepat Waktu", f"{lulus} ({lulus_pct:.1f}%)")
    
    with col3:
        tidak_lulus = (df_data['lulus_tepat_waktu'] == 'Tidak').sum()
        tidak_lulus_pct = (tidak_lulus / len(df_data) * 100)
        st.metric("Tidak Lulus Tepat Waktu", f"{tidak_lulus} ({tidak_lulus_pct:.1f}%)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Distribusi Status Kelulusan")
        status_counts = df_data['lulus_tepat_waktu'].value_counts()
        st.bar_chart(status_counts)
    
    with col2:
        st.subheader(" Feature Importance")
        features = ['IPK', 'Absen', 'Kegiatan']
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
    
    st.markdown("---")
    st.subheader(" Statistik Deskriptif Features")
    stats_df = df_data[['ipk', 'absen', 'kegiatan']].describe().round(3)
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader(" Distribusi Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**IPK Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df_data['ipk'], bins=15, edgecolor='black', alpha=0.7)
        ax.set_xlabel('IPK')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.write("**Absen Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df_data['absen'], bins=15, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Absen')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col3:
        st.write("**Kegiatan Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df_data['kegiatan'], bins=10, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Kegiatan')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
        © 2024 Student Graduation Prediction System | 
        Powered by Machine Learning & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
