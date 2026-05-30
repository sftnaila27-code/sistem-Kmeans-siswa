import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime

# ======================== KONFIGURASI AWAL ========================
st.set_page_config(
    page_title="Sistem Clustering Siswa Berprestasi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    /* Warna untuk tombol menu di sidebar */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 15px !important;
        margin: 5px 0 !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a0 100%) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important;
    }
    
    /* Warna untuk kotak upload file */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: linear-gradient(135deg, #667eea25 0%, #764ba225 100%);
        border-color: #764ba2;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Warna sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Warna header menu */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa !important;
        font-size: 0.85rem !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        border-bottom: 2px solid #a78bfa !important;
        display: inline-block !important;
        padding-bottom: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================== CUSTOM CSS UNTUK UPLOAD BOX ========================
st.markdown("""
<style>
    /* Warna untuk kotak upload file */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: linear-gradient(135deg, #667eea25 0%, #764ba225 100%);
        border-color: #764ba2;
    }
    
    /* Warna teks di dalam uploader */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px);
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ======================== FUNGSI PREPROCESSING ========================
def data_cleaning(df):
    df_clean = df.copy()
    sebelum_duplikat = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    
    kolom_nilai = ['Matematika', 'B Indonesia', 'B Inggris', 'IPA']
    for col in kolom_nilai:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            df_clean[col] = df_clean[col].clip(0, 100)
    
    return df_clean, {'duplikat_dihapus': sebelum_duplikat - len(df_clean)}

def seleksi_atribut(df):
    atribut_target = ['Matematika', 'B Indonesia', 'B Inggris', 'IPA']
    atribut_ada = [col for col in atribut_target if col in df.columns]
    
    identitas = []
    if 'Nama Siswa' in df.columns:
        identitas.append('Nama Siswa')
    if 'Kelas' in df.columns:
        identitas.append('Kelas')
    
    if len(identitas) == 0:
        df = df.reset_index()
        df = df.rename(columns={'index': 'ID Siswa'})
        identitas = ['ID Siswa']
    
    return df[identitas + atribut_ada].copy(), atribut_ada

def normalisasi_data(df, atribut):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[atribut] = scaler.fit_transform(df[atribut])
    return df_normalized, scaler

def hitung_rata_rata(df, atribut):
    df['Rata Rata'] = df[atribut].mean(axis=1).round(2)
    return df

def clustering_kmeans(df, atribut, n_clusters=3):
    X = df[atribut].values
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                    max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

def pemetaan_cluster(df, clusters):
    df_temp = df.copy()
    df_temp['Cluster_Awal'] = clusters
    rata_per_cluster = df_temp.groupby('Cluster_Awal')['Rata Rata'].mean().sort_values()
    
    if len(rata_per_cluster) >= 3:
        pemetaan = {
            rata_per_cluster.index[2]: 1,
            rata_per_cluster.index[1]: 2,
            rata_per_cluster.index[0]: 3
        }
    else:
        pemetaan = {i: i+1 for i in range(len(rata_per_cluster))}
    
    df['Cluster'] = [pemetaan.get(c, c) for c in clusters]
    kategori_map = {1: 'Prestasi Tinggi', 2: 'Prestasi Sedang', 3: 'Prestasi Rendah'}
    df['Kategori'] = df['Cluster'].map(lambda x: kategori_map.get(x, f'Cluster {x}'))
    return df

def perangkingan_siswa(df):
    df = df.sort_values(by='Rata Rata', ascending=False).reset_index(drop=True)
    df['Peringkat'] = df.index + 1
    return df

def evaluasi_model(X, labels, n_clusters):
    if n_clusters > 1 and len(set(labels)) > 1:
        return round(silhouette_score(X, labels), 4)
    return 0

def hitung_wcss(X, max_k=10):
    wcss = []
    k_range = range(1, min(max_k + 1, len(X)) + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return list(k_range), wcss

# ======================== PROSES UTAMA ========================
def proses_lengkap(uploaded_file):
    hasil_proses = {
        'status': 'success',
        'messages': [],
        'data': None,
        'silhouette_score': 0,
        'wcss_data': None,
        'k_range': None,
        'statistik': {},
        'atribut': []
    }
    
    try:
        df = pd.read_excel(uploaded_file)
        df_clean, _ = data_cleaning(df)
        df_selected, atribut = seleksi_atribut(df_clean)
        df_selected = hitung_rata_rata(df_selected, atribut)
        df_normalized, _ = normalisasi_data(df_selected, atribut)
        clusters, _ = clustering_kmeans(df_normalized, atribut)
        df_clustered = pemetaan_cluster(df_selected, clusters)
        df_final = perangkingan_siswa(df_clustered)
        
        X = df_normalized[atribut].values
        sil_score = evaluasi_model(X, df_final['Cluster'].values, 3)
        k_range, wcss = hitung_wcss(X)
        
        distribusi = {}
        for kat in df_final['Kategori'].unique():
            distribusi[kat] = int(df_final[df_final['Kategori'] == kat].shape[0])
        
        hasil_proses['statistik'] = {
            'total_siswa': len(df_final),
            'jumlah_cluster': 3,
            'silhouette': sil_score,
            'distribusi': distribusi,
            'rata_tertinggi': df_final['Rata Rata'].max(),
            'rata_terendah': df_final['Rata Rata'].min()
        }
        
        hasil_proses['data'] = df_final
        hasil_proses['atribut'] = atribut
        hasil_proses['silhouette_score'] = sil_score
        hasil_proses['wcss_data'] = wcss
        hasil_proses['k_range'] = k_range
        
    except Exception as e:
        hasil_proses['status'] = 'error'
        hasil_proses['messages'].append(str(e))
    
    return hasil_proses

# ======================== INISIALISASI SESSION STATE ========================
if 'data_terproses' not in st.session_state:
    st.session_state.data_terproses = None
if 'hasil_proses' not in st.session_state:
    st.session_state.hasil_proses = None
if 'menu_aktif' not in st.session_state:
    st.session_state.menu_aktif = "Dashboard"
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 3rem;'>🎓</div>
        <div style='font-size: 1.2rem; font-weight: 700; margin-top: 10px;'>EduCluster Pro</div>
        <div style='font-size: 0.7rem; opacity: 0.7;'>by Sufatun Aila</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 Upload Data Siswa",
        type=['xlsx', 'xls'],
        help="Format: Matematika, B Indonesia, B Inggris, IPA"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Proses Data", use_container_width=True):
            with st.spinner("Memproses data..."):
                hasil = proses_lengkap(uploaded_file)
                if hasil['status'] == 'success':
                    st.session_state.hasil_proses = hasil
                    st.session_state.data_terproses = hasil['data']
                    st.session_state.file_uploaded = True
                    st.success("✅ Proses selesai!")
                else:
                    st.error("Error: " + str(hasil['messages']))
    
    st.markdown("---")
    
    if st.session_state.file_uploaded:
        st.markdown("### 📋 Menu Navigasi")
        menu_items = {
            "🏠 Dashboard": "Dashboard",
            "📋 Dataset": "Dataset",
            "⚙️ Preprocessing": "Preprocessing",
            "📊 Hasil Clustering": "Hasil Clustering",
            "🏆 Peringkat Siswa": "Hasil Peringkat",
            "📈 Visualisasi": "Visualisasi",
            "📐 Evaluasi": "Evaluasi Model",
            "ℹ️ Tentang": "Tentang"
        }
        
        for label, key in menu_items.items():
            if st.button(label, use_container_width=True):
                st.session_state.menu_aktif = key

# ======================== MAIN CONTENT ========================
if not st.session_state.file_uploaded:
    # Halaman Landing Page
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 4rem; margin-bottom: 20px;'>🎓</div>
        <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; margin-bottom: 1rem;'>
            EduCluster Pro
        </h1>
        <p style='font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem;'>
            Sistem Clustering Cerdas untuk Identifikasi Siswa Berprestasi
        </p>
        <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
            <div style='background: #f3f4f6; padding: 1rem; border-radius: 12px; width: 200px;'>
                <div style='font-size: 2rem;'>📊</div>
                <div style='font-weight: 600;'>K-Means</div>
                <div style='font-size: 0.8rem; color: #6b7280;'>Algoritma Clustering</div>
            </div>
            <div style='background: #f3f4f6; padding: 1rem; border-radius: 12px; width: 200px;'>
                <div style='font-size: 2rem;'>🎯</div>
                <div style='font-weight: 600;'>3 Cluster</div>
                <div style='font-size: 0.8rem; color: #6b7280;'>Tinggi, Sedang, Rendah</div>
            </div>
            <div style='background: #f3f4f6; padding: 1rem; border-radius: 12px; width: 200px;'>
                <div style='font-size: 2rem;'>📈</div>
                <div style='font-weight: 600;'>Evaluasi</div>
                <div style='font-size: 0.8rem; color: #6b7280;'>Elbow + Silhouette</div>
            </div>
        </div>
        <div style='margin-top: 3rem; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 20px;'>
            <p style='color: #4b5563;'>📌 Upload file Excel di sidebar untuk memulai analisis</p>
            <p style='font-size: 0.8rem; color: #9ca3af; margin-top: 0.5rem;'>
                Format: Kolom Matematika, B Indonesia, B Inggris, IPA
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    df_hasil = st.session_state.data_terproses
    hasil = st.session_state.hasil_proses
    atribut = hasil.get('atribut', ['Matematika', 'B Indonesia', 'B Inggris', 'IPA'])
    
    warna_map = {'Prestasi Tinggi': '#3B82F6', 'Prestasi Sedang': '#22C55E', 'Prestasi Rendah': '#EF4444'}
    
    menu = st.session_state.menu_aktif
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>✨ EduCluster Pro ✨</h1>
        <p>Analisis Pengelompokan Nilai Siswa untuk Identifikasi Siswa Berprestasi dengan Metode K-Means Clustering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ======================== DASHBOARD ========================
    if menu == "Dashboard":
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{hasil['statistik']['total_siswa']}</div>
                <div class='metric-label'>Total Siswa</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{hasil['statistik']['jumlah_cluster']}</div>
                <div class='metric-label'>Jumlah Cluster</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{hasil['statistik']['silhouette']}</div>
                <div class='metric-label'>Silhouette Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>K-Means++</div>
                <div class='metric-label'>Algoritma</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("<div class='card'><div class='card-title'>📊 Distribusi Siswa per Cluster</div>", unsafe_allow_html=True)
            jumlah_per_kat = df_hasil['Kategori'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            colors = [warna_map.get(k, '#888888') for k in jumlah_per_kat.index]
            ax1.pie(jumlah_per_kat, labels=jumlah_per_kat.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown("<div class='card'><div class='card-title'>📈 Rata-rata Nilai per Cluster</div>", unsafe_allow_html=True)
            rata_kat = df_hasil.groupby('Kategori')['Rata Rata'].mean()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            colors = [warna_map.get(k, '#888888') for k in rata_kat.index]
            bars = ax2.bar(rata_kat.index, rata_kat.values, color=colors, edgecolor='black')
            ax2.set_ylim(0, 100)
            for bar, val in zip(bars, rata_kat.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', fontweight='bold')
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Preview
        st.markdown("---")
        st.markdown("<div class='card'><div class='card-title'>📋 Preview Hasil Clustering</div>", unsafe_allow_html=True)
        
        kolom_tampil = [col for col in ['Nama Siswa', 'Kelas', 'ID Siswa', 'Rata Rata', 'Kategori'] if col in df_hasil.columns]
        st.dataframe(df_hasil[kolom_tampil].head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================== HASIL CLUSTERING ========================
    elif menu == "Hasil Clustering":
        st.markdown("<div class='card'><div class='card-title'>📊 Hasil Pengelompokan Siswa</div>", unsafe_allow_html=True)
        
        pilih_kat = st.selectbox("Filter Kategori", ["Semua"] + list(df_hasil['Kategori'].unique()))
        
        if pilih_kat != "Semua":
            df_tampil = df_hasil[df_hasil['Kategori'] == pilih_kat]
        else:
            df_tampil = df_hasil
        
        kolom_tampil = [col for col in ['Nama Siswa', 'Kelas', 'ID Siswa'] + atribut + ['Rata Rata', 'Kategori'] if col in df_tampil.columns]
        st.dataframe(df_tampil[kolom_tampil], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Statistik
        st.markdown("<div class='card'><div class='card-title'>📊 Statistik Tiap Kelompok</div>", unsafe_allow_html=True)
        statistik = df_hasil.groupby('Kategori')['Rata Rata'].agg(['count', 'mean', 'min', 'max']).round(2)
        statistik.columns = ['Jumlah', 'Rata-rata', 'Min', 'Max']
        st.dataframe(statistik, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================== HASIL PERINGKAT ========================
    elif menu == "Hasil Peringkat":
        st.markdown("<div class='card'><div class='card-title'>🏆 Peringkat Kelas</div>", unsafe_allow_html=True)
        
        df_peringkat = df_hasil.sort_values(by='Rata Rata', ascending=False).reset_index(drop=True)
        df_peringkat['No'] = df_peringkat.index + 1
        
        kolom_tampil = [col for col in ['No', 'Nama Siswa', 'Kelas', 'ID Siswa', 'Rata Rata', 'Kategori'] if col in df_peringkat.columns]
        st.dataframe(df_peringkat[kolom_tampil], use_container_width=True, height=500)
        st.download_button("📥 Download CSV", df_peringkat.to_csv(index=False), "peringkat_siswa.csv")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================== VISUALISASI ========================
    elif menu == "Visualisasi":
        st.markdown("<div class='card'><div class='card-title'>📈 Visualisasi Data</div>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Perbandingan Mata Pelajaran", "🥧 Distribusi Cluster"])
        
        with tab1:
            rata_mapel = df_hasil.groupby('Kategori')[atribut].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            rata_mapel.T.plot(kind='bar', ax=ax, color=['#3B82F6', '#22C55E', '#EF4444'])
            ax.set_ylabel("Rata-rata Nilai")
            ax.set_xlabel("Mata Pelajaran")
            ax.set_title("Perbandingan Nilai per Kategori Prestasi")
            ax.legend(title="Kategori", bbox_to_anchor=(1.05, 1))
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            jumlah_kat = df_hasil['Kategori'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            colors = [warna_map.get(k, '#888888') for k in jumlah_kat.index]
            ax2.pie(jumlah_kat, labels=jumlah_kat.index, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================== EVALUASI ========================
    elif menu == "Evaluasi Model":
        st.markdown("<div class='card'><div class='card-title'>📐 Evaluasi Model Clustering</div>", unsafe_allow_html=True)
        
        # Elbow Method
        st.subheader("1. Elbow Method")
        if hasil['wcss_data'] and hasil['k_range']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hasil['k_range'], hasil['wcss_data'], 'o-', color='#667eea', linewidth=2, markersize=8)
            ax.set_xlabel('Jumlah Cluster (K)')
            ax.set_ylabel('WCSS')
            ax.set_title('Elbow Method - Penentuan K Optimal')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=3, color='red', linestyle='--', label='K=3 (Terpilih)')
            ax.legend()
            st.pyplot(fig)
        
        # Silhouette Score
        st.subheader("2. Silhouette Score")
        sil_score = hasil['statistik']['silhouette']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            warna = '#22C55E' if sil_score >= 0.5 else '#F59E0B' if sil_score >= 0.25 else '#EF4444'
            ax2.barh(['Score'], [sil_score], color=warna, height=0.4)
            ax2.set_xlim(-1, 1)
            ax2.set_xlabel('Silhouette Score')
            ax2.axvline(x=0.5, color='green', linestyle='--', label='Batas Baik (0.5)')
            ax2.legend()
            st.pyplot(fig2)
        
        with col2:
            st.metric("Nilai Silhouette", sil_score)
            if sil_score >= 0.5:
                st.success("✅ Kualitas: BAIK")
            elif sil_score >= 0.25:
                st.warning("⚠️ Kualitas: CUKUP")
            else:
                st.error("❌ Kualitas: BURUK")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
        # ======================== TENTANG ========================
    elif menu == "Tentang":
        st.header("📖 Tentang Aplikasi")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>Sistem Clustering Siswa Berprestasi</h3>
            <p style='color: rgba(255,255,255,0.9); margin-top: 10px;'>
                Aplikasi ini menerapkan <strong>K-Means Clustering</strong> untuk mengelompokkan nilai siswa 
                guna mengidentifikasi siswa berprestasi secara objektif.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: #f0f9ff; padding: 15px; border-radius: 12px; border-left: 4px solid #3B82F6; margin-bottom: 15px;'>
                <h4 style='color: #1E40AF; margin-top: 0;'>📌 Metode yang Digunakan</h4>
                <ul style='margin-bottom: 0;'>
                    <li><strong>K-Means Clustering</strong><br><small>Pengelompokan berbasis jarak Euclidean</small></li>
                    <li><strong>Min-Max Normalization</strong><br><small>Normalisasi data ke rentang 0-1</small></li>
                    <li><strong>Elbow Method</strong><br><small>Penentuan jumlah cluster optimal</small></li>
                    <li><strong>Silhouette Coefficient</strong><br><small>Evaluasi kualitas cluster</small></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #f0fdf4; padding: 15px; border-radius: 12px; border-left: 4px solid #22C55E; margin-bottom: 15px;'>
                <h4 style='color: #166534; margin-top: 0;'>📊 Output yang Dihasilkan</h4>
                <ul style='margin-bottom: 0;'>
                    <li><strong>3 Cluster Prestasi</strong><br><small>Tinggi, Sedang, Rendah</small></li>
                    <li><strong>Peringkat Siswa</strong><br><small>Berdasarkan rata-rata nilai</small></li>
                    <li><strong>Visualisasi Grafik</strong><br><small>Distribusi dan perbandingan</small></li>
                    <li><strong>Evaluasi Model</strong><br><small>Elbow Method & Silhouette Score</small></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #fef3c7; padding: 15px; border-radius: 12px; border-left: 4px solid #F59E0B; margin-bottom: 15px;'>
            <h4 style='color: #92400E; margin-top: 0;'>📋 Data yang Digunakan</h4>
            <p style='margin-bottom: 0;'>
                Nilai mata pelajaran: <strong>Matematika, Bahasa Indonesia, Bahasa Inggris, IPA</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ======================== PREPROCESSING ========================
    elif menu == "Preprocessing":
        st.markdown("<div class='card'><div class='card-title'>⚙️ Preprocessing Data</div>", unsafe_allow_html=True)
        
        st.subheader("1. Seleksi Atribut")
        st.markdown(f"""
        **Atribut yang dipilih untuk clustering:**
        - Matematika
        - B Indonesia
        - B Inggris
        - IPA
        """)
        
        st.subheader("2. Normalisasi Data (Min-Max)")
        st.latex(r"X' = \frac{X - X_{min}}{X_{max} - X_{min}}")
        
        st.subheader("3. Contoh Hasil Normalisasi")
        
        if len(atribut) > 0:
            sample_data = df_hasil[atribut].head(5)
            
            if 'Nama Siswa' in df_hasil.columns:
                labels = df_hasil['Nama Siswa'].head(5).tolist()
            else:
                labels = [f"Siswa {i+1}" for i in range(5)]
            
            # Tabel Nilai Asli
            st.markdown("**Tabel 1: Nilai Asli**")
            df_tabel1 = sample_data.copy()
            df_tabel1.insert(0, 'Siswa', labels)
            st.dataframe(df_tabel1, use_container_width=True)
            
            # Tabel Normalisasi
            st.markdown("**Tabel 2: Nilai Setelah Normalisasi (0-1)**")
            df_norm = pd.DataFrame()
            for col in atribut:
                min_val = df_hasil[col].min()
                max_val = df_hasil[col].max()
                if max_val > min_val:
                    df_norm[col] = (sample_data[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0.5
            df_norm.insert(0, 'Siswa', labels)
            st.dataframe(df_norm.round(4), use_container_width=True)
            
            st.success("✅ Normalisasi selesai! Data siap untuk clustering.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================== DATASET ========================
    elif menu == "Dataset":
        st.markdown("<div class='card'><div class='card-title'>📋 Dataset Siswa</div>", unsafe_allow_html=True)
        st.dataframe(df_hasil, use_container_width=True)
        st.info(f"Total {df_hasil.shape[0]} baris, {df_hasil.shape[1]} kolom")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================== FOOTER ========================
if st.session_state.file_uploaded:
    st.markdown("""
    <div class='footer'>
        © 2026 EduCluster Pro | Sistem Informasi | Universitas Ibrahimy
    </div>
    """, unsafe_allow_html=True)