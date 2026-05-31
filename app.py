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
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: #1e293b;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 0.8rem;
    }
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

# ======================== FUNGSI SESUAI TAHAPAN KDD (Bab 1 & Bab 3) ========================

# Tahap 1: Data Selection & Pre-processing / Cleaning
def data_cleaning(df):
    """
    Tahapan Data Cleaning (Bab 1 - KDD)
    Menghapus duplikasi, menangani nilai kosong, dan memastikan rentang nilai 0-100
    """
    df_clean = df.copy()
    sebelum_duplikat = len(df_clean)
    df_clean = df_clean.drop_duplicates() # Hapus data ganda
    
    # Definisi atribut sesuai proposal: Matematika, B.Indonesia, B.Inggris, IPA
    kolom_nilai = ['Matematika', 'B Indonesia', 'B Inggris', 'IPA']
    for col in kolom_nilai:
        if col in df_clean.columns:
            # Ubah ke numerik, ubah teks/karakter menjadi NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Isi data kosong dengan nilai tengah (median)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            # Pastikan nilai berada di rentang 0 - 100
            df_clean[col] = df_clean[col].clip(0, 100)
    
    return df_clean, {'duplikat_dihapus': sebelum_duplikat - len(df_clean)}

def seleksi_atribut(df):
    """
    Tahapan Seleksi Atribut / Transformation (Bab 2 & Bab 3)
    Memilih hanya kolom yang relevan untuk diolah
    """
    # Atribut yang dipilih sesuai Batasan Masalah
    atribut_target = ['Matematika', 'B Indonesia', 'B Inggris', 'IPA']
    atribut_ada = [col for col in atribut_target if col in df.columns]
    
    # Identitas siswa
    identitas = []
    if 'Nama Siswa' in df.columns: identitas.append('Nama Siswa')
    if 'Kelas' in df.columns: identitas.append('Kelas')
    if len(identitas) == 0: 
        df = df.reset_index()
        df = df.rename(columns={'index': 'ID Siswa'})
        identitas = ['ID Siswa']
    
    return df[identitas + atribut_ada].copy(), atribut_ada

def normalisasi_data(df, atribut):
    """
    Tahapan Normalisasi Data menggunakan Rumus Min-Max (Bab 2)
    Rumus: X' = (X - Xmin) / (Xmax - Xmin)
    """
    scaler = MinMaxScaler(feature_range=(0, 1)) # Sesuai rumus rentang 0-1
    df_normalized = df.copy()
    df_normalized[atribut] = scaler.fit_transform(df[atribut])
    return df_normalized, scaler

# Fungsi Bantuan: Perhitungan Nilai
def hitung_rata_rata(df, atribut):
    """
    Menghitung rata-rata nilai akademik untuk keperluan perangkingan
    Rumus: Rata-Rata = (Mtk + Bin + Bing + Ipa) / 4 (Bab 2)
    """
    df['Rata Rata'] = df[atribut].mean(axis=1).round(2)
    return df

# Tahap 2: Data Mining - Algoritma K-Means (Bab 2)
def clustering_kmeans(df, atribut, n_clusters=3):
    """
    Penerapan Algoritma K-Means dengan jarak Euclidean Distance
    Langkah: Tentukan K -> Inisialisasi Centroid -> Iterasi -> Stabil
    """
    X = df[atribut].values
    # Inisialisasi centroid menggunakan 'k-means++' agar hasil lebih baik
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                    max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Mengembalikan hasil cluster dan nilai pusat centroid
    return clusters, kmeans.cluster_centers_

def pemetaan_cluster(df, clusters):
    """
    Pemetaan Hasil Cluster ke Kategori:
    Cluster 1 = Prestasi Tinggi
    Cluster 2 = Prestasi Sedang
    Cluster 3 = Prestasi Rendah
    """
    df_temp = df.copy()
    df_temp['Cluster_Label'] = clusters
    
    # Analisis rata-rata nilai untuk menentukan urutan cluster
    rata_per_cluster = df_temp.groupby('Cluster_Label')['Rata Rata'].mean().sort_values(ascending=False)
    
    # Pemetaan logis: Nilai rata-rata tertinggi = Tinggi
    pemetaan = {}
    kategori = ['Prestasi Tinggi', 'Prestasi Sedang', 'Prestasi Rendah']
    
    # Membuat peta dari label acak ke kategori yang terstruktur
    for idx, cluster_id in enumerate(rata_per_cluster.index):
        pemetaan[cluster_id] = idx + 1 # 1=Tinggi, 2=Sedang, 3=Rendah
    
    df['Cluster'] = [pemetaan.get(c, 0) for c in clusters]
    df['Kategori'] = df['Cluster'].map({1: 'Prestasi Tinggi', 2: 'Prestasi Sedang', 3: 'Prestasi Rendah'})
    
    return df

def perangkingan_siswa(df):
    """
    Pengurutan siswa terbaik berdasarkan rata-rata nilai (Bab 2)
    """
    df = df.sort_values(by='Rata Rata', ascending=False).reset_index(drop=True)
    df['Peringkat'] = df.index + 1
    return df

# Tahap 3: Evaluation / Evaluasi Model (Bab 2)
 evaluasi_model(X, labels, n_clusters):
    """
    Evaluasi menggunakan Silhouette Coefficient
    """
    if n_clusters > 1 and len(set(labels)) > 1:
        return round(silhouette_score(X, labels), 4)
    return 0

def hitung_wcss(X, max_k=10):
    """
    Evaluasi menggunakan Elbow Method
    """
    wcss = []
    k_range = range(1, min(max_k + 1, len(X)) + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return list(k_range), wcss

# ======================== ALUR PROSES LENGKAP SESUAI FLOWCHART (Bab 3) ========================
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
        <div style='font-size: 0.7rem; opacity: 0.7;'>Sufatun Aila | 2022502078</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 Upload Data Nilai Siswa",
        type=['xlsx', 'xls'],
        help="Format: Nama Siswa, Kelas, Matematika, B Indonesia, B Inggris, IPA"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Proses Data Otomatis", use_container_width=True):
            with st.spinner("Memproses data sesuai alur K-Means..."):
                hasil = proses_lengkap(uploaded_file)
                if hasil['status'] == 'success':
                    st.session_state.hasil_proses = hasil
                    st.session_state.data_terproses = hasil['data']
                    st.session_state.file_uploaded = True
                    st.success("✅ Proses Selesai! Lihat hasil di menu navigasi.")
                else:
                    st.error(f"Error: {hasil['messages']}")
    
    st.markdown("---")
    
    if st.session_state.file_uploaded:
        st.markdown("### 📋 Menu Navigasi")
        menu_items = {
            "🏠 Dashboard": "Dashboard",
            "📋 Dataset Awal": "Dataset",
            "⚙️ Tahap Preprocessing": "Preprocessing",
            "📊 Hasil Clustering": "Hasil Clustering",
            "🏆 Peringkat Siswa": "Hasil Peringkat",
            "📈 Visualisasi": "Visualisasi",
            "📐 Evaluasi Model": "Evaluasi Model",
            "ℹ️ Tentang Sistem": "Tentang"
        }
        
        for label, key in menu_items.items():
            if st.button(label, use_container_width=True):
                st.session_state.menu_aktif = key

# ======================== MAIN CONTENT ========================
if not st.session_state.file_uploaded:
    # Halaman Awal
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 4rem; margin-bottom: 20px;'>🎓</div>
        <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; margin-bottom: 1rem;'>
            Analisis Pengelompokan Nilai Siswa
        </h1>
        <p style='font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem;'>
            Identifikasi Siswa Berprestasi Menggunakan Metode K-Means Clustering
        </p>
        <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
            <div style='background: #f3f4f6; padding: 1rem; border-radius: 12px; width: 200px;'>
                <div style='font-size: 2rem;'>⚙️</div>
                <div style='font-weight: 600;'>Otomatis</div>
                <div style='font-size: 0.8rem; color: #6b7280;'>Hanya Upload File</div>
            </div>
            <div style='background: #f3f4f6; padding: 1rem; border-radius: 12px; width: 200px;'>
                <div style='font-size: 2rem;'>🎯</div>
                <div style='font-weight: 600;'>3 Kelompok</div>
                <div style='font-size: 0.8rem; color: #6b7280;'>Tinggi, Sedang, Rendah</div>
            </div>
        </div>
        <div style='margin-top: 3rem; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 20px;'>
            <p style='color: #4b5563;'>📌 Silakan upload file Excel berisi data nilai siswa di sidebar</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    df_hasil = st.session_state.data_terproses
    hasil = st.session_state.hasil_proses
    atribut = hasil.get('atribut', ['Matematika', 'B Indonesia', 'B Inggris', 'IPA'])
    
    warna_map = {'Prestasi Tinggi': '#3B82F6', 'Prestasi Sedang': '#22C55E', 'Prestasi Rendah': '#EF4444'}
    menu = st.session_state.menu_aktif

    # ======================== DASHBOARD ========================
    if menu == "Dashboard":
        st.markdown("<div class='main-header'><h1>✨ Dashboard Sistem ✨</h1><p>Ringkasan hasil analisis pengelompokan siswa</p></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{hasil['statistik']['total_siswa']}</div><div class='metric-label'>Total Siswa</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{hasil['statistik']['jumlah_cluster']}</div><div class='metric-label'>Jumlah Cluster</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{hasil['statistik']['silhouette']}</div><div class='metric-label'>Nilai Silhouette</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>Otomatis</div><div class='metric-label'>Pemrosesan</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("<div class='card'><div class='card-title'>📊 Distribusi Jumlah Siswa</div>", unsafe_allow_html=True)
            jumlah_per_kat = df_hasil['Kategori'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            colors = [warna_map.get(k, '#888888') for k in jumlah_per_kat.index]
            ax1.pie(jumlah_per_kat, labels=jumlah_per_kat.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown("<div class='card'><div class='card-title'>📈 Rata-rata Nilai per Kelompok</div>", unsafe_allow_html=True)
            rata_kat = df_hasil.groupby('Kategori')['Rata Rata'].mean()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            colors = [warna_map.get(k, '#888888') for k in rata_kat.index]
            bars = ax2.bar(rata_kat.index, rata_kat.values, color=colors, edgecolor='black')
            ax2.set_ylim(0, 100)
            for bar, val in zip(bars, rata_kat.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', fontweight='bold')
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

    # ======================== PREPROCESSING ========================
    elif menu == "Preprocessing":
        st.markdown("<div class='main-header'><h1>⚙️ Tahap Preprocessing</h1><p>Penjelasan proses persiapan data sesuai tahapan KDD</p></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'><div class='card-title'>1. Seleksi Atribut</div>", unsafe_allow_html=True)
        st.markdown("""
        Sesuai **Batasan Masalah** pada Bab 1, atribut yang diproses hanya nilai mata pelajaran inti:
        - Matematika
        - Bahasa Indonesia
        - Bahasa Inggris
        - IPA
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>2. Normalisasi Data (Min-Max)</div>", unsafe_allow_html=True)
        st.latex(r"X' = \frac{X - X_{min}}{X_{max} - X_{min}}")
        st.markdown("""
        Proses ini mengubah rentang nilai asli (0-100) menjadi rentang **0 sampai 1**. 
        Tujuannya agar tidak ada mata pelajaran yang mendominasi perhitungan jarak Euclidean pada algoritma K-Means.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tampilkan Nilai Pusat Kelompok (Centroid)
        st.markdown("<div class='card'><div class='card-title'>📍 Nilai Pusat Kelompok Akhir (Centroid)</div>", unsafe_allow_html=True)
        if hasil['centroid'] is not None:
        df_centroid = pd.DataFrame(
        hasil['centroid'], 
        columns=atribut, 
        index=['Kelompok Tinggi', 'Kelompok Sedang', 'Kelompok Rendah']
    )
    st.dataframe(df_centroid.round(4), use_container_width=True)
    st.info("Nilai di atas merupakan nilai pusat kelompok hasil iterasi terakhir sebelum proses berhenti. Semakin jauh jarak antar nilai pusat, semakin baik pemisahan datanya.")
st.markdown("</div>", unsafe_allow_html=True)

        # Tampilkan contoh data centroid
        st.markdown("<div class='card'><div class='card-title'>3. Nilai Centroid (Pusat Cluster)</div>", unsafe_allow_html=True)
        if hasil['centroid'] is not None:
            df_centroid = pd.DataFrame(hasil['centroid'], columns=atribut, index=['Cluster 1', 'Cluster 2', 'Cluster 3'])
            st.dataframe(df_centroid.round(4), use_container_width=True)
            st.info("Nilai di atas merupakan pusat kelompok hasil iterasi akhir algoritma K-Means.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ======================== HASIL CLUSTERING ========================
    elif menu == "Hasil Clustering":
        st.markdown("<div class='main-header'><h1>📊 Hasil Pengelompokan</h1><p>Data siswa yang telah dikelompokkan</p></div>", unsafe_allow_html=True)
        
        pilih_kat = st.selectbox("Filter Berdasarkan Kategori", ["Semua", "Prestasi Tinggi", "Prestasi Sedang", "Prestasi Rendah"])
        
        if pilih_kat != "Semua":
            df_tampil = df_hasil[df_hasil['Kategori'] == pilih_kat]
        else:
            df_tampil = df_hasil
        
        kolom_tampil = [col for col in ['Nama Siswa', 'Kelas'] + atribut + ['Rata Rata', 'Cluster', 'Kategori'] if col in df_tampil.columns]
        st.dataframe(df_tampil[kolom_tampil], use_container_width=True)
        
        # Statistik
        st.markdown("<div class='card'><div class='card-title'>📌 Karakteristik Tiap Kelompok</div>", unsafe_allow_html=True)
        statistik = df_hasil.groupby('Kategori')[atribut + ['Rata Rata']].agg(['mean']).round(2)
        st.dataframe(statistik, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ======================== EVALUASI ========================
    elif menu == "Evaluasi Model":
        st.markdown("<div class='main-header'><h1>📐 Evaluasi Model</h1><p>Pengujian kualitas hasil clustering</p></div>", unsafe_allow_html=True)
        
        st.subheader("1. Elbow Method")
        st.markdown("Metode ini menentukan jumlah cluster optimal berdasarkan nilai WCSS. Sesuai penelitian, dipilih K=3 karena terjadi penurunan drastis pada titik tersebut.")
        if hasil['wcss_data'] and hasil['k_range']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hasil['k_range'], hasil['wcss_data'], 'o-', color='#667eea', linewidth=2, markersize=8)
            ax.set_xlabel('Jumlah Cluster (K)')
            ax.set_ylabel('WCSS')
            ax.axvline(x=3, color='red', linestyle='--', label='K=3 (Terpilih)')
            ax.legend()
            st.pyplot(fig)

        st.subheader("2. Silhouette Coefficient")
        st.markdown("Nilai ini mengukur seberapa mirip objek dengan kelompoknya dibandingkan kelompok lain. Rentang nilai [-1, 1]. Semakin mendekati 1 semakin baik.")
        sil_score = hasil['statistik']['silhouette']
        
        col1, col2 = st.columns([3,1])
        with col1:
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            warna = '#22C55E' if sil_score >= 0.5 else '#F59E0B' if sil_score >= 0.25 else '#EF4444'
            ax2.barh(['Score'], [sil_score], color=warna, height=0.4)
            ax2.set_xlim(-1, 1)
            ax2.axvline(x=0.5, color='green', linestyle='--', label='Batas Baik (0.5)')
            ax2.legend()
            st.pyplot(fig2)
        with col2:
            st.metric("Nilai", sil_score)
            if sil_score >= 0.5: st.success("✅ Baik")
            elif sil_score >= 0.25: st.warning("⚠️ Cukup")
            else: st.error("❌ Kurang")

    # ======================== TENTANG ========================
    elif menu == "Tentang":
        st.markdown("<div class='main-header'><h1>ℹ️ Tentang Penelitian</h1></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Analisis Pengelompokan Nilai Siswa Menggunakan K-Means Clustering</div>
            <p>
            Sistem ini dibangun berdasarkan proposal skripsi dengan tujuan mengidentifikasi siswa berprestasi secara objektif.
            </p>
            <p><strong>Alur kerja sistem mengikuti tahapan Knowledge Discovery in Database (KDD):</strong></p>
            <ol>
                <li><strong>Selection:</strong> Memilih data nilai siswa.</li>
                <li><strong>Preprocessing:</strong> Pembersihan data dan normalisasi.</li>
                <li><strong>Transformation:</strong> Menyiapkan data untuk algoritma.</li>
                <li><strong>Data Mining:</strong> Penerapan Algoritma K-Means.</li>
                <li><strong>Interpretation:</strong> Menampilkan hasil kelompok dan peringkat.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ======================== TAMBAHAN MENU LAINNYA ========================
    elif menu == "Dataset":
        st.markdown("<div class='card'><div class='card-title'>📋 Dataset Awal</div>", unsafe_allow_html=True)
        st.dataframe(df_hasil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "Hasil Peringkat":
        st.markdown("<div class='card'><div class='card-title'>🏆 Peringkat Siswa Berdasarkan Rata-Rata</div>", unsafe_allow_html=True)
        df_peringkat = df_hasil.sort_values(by='Rata Rata', ascending=False).reset_index(drop=True)
        df_peringkat['No'] = df_peringkat.index + 1
        kolom_tampil = [col for col in ['No', 'Nama Siswa', 'Kelas', 'Rata Rata', 'Kategori'] if col in df_peringkat.columns]
        st.dataframe(df_peringkat[kolom_tampil], use_container_width=True, height=500)
        st.download_button("📥 Download CSV", df_peringkat.to_csv(index=False), "peringkat_siswa.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "Visualisasi":
        st.markdown("<div class='card'><div class='card-title'>📈 Visualisasi Perbandingan Nilai</div>", unsafe_allow_html=True)
        rata_mapel = df_hasil.groupby('Kategori')[atribut].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        rata_mapel.T.plot(kind='bar', ax=ax, color=['#3B82F6', '#22C55E', '#EF4444'])
        ax.set_ylabel("Rata-rata Nilai")
        ax.set_title("Perbandingan Nilai Mata Pelajaran Tiap Kategori")
        ax.legend(title="Kategori")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_1=True)

# ======================== FOOTER ========================
if st.session_state.file_uploaded:
    st.markdown("""
    <div class='footer'>
        © 2026 - Sistem Informasi - Universitas Ibrahimy
    </div>
    """, unsafe_allow_html=True)