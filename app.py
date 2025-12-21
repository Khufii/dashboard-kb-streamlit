import os
import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import spearmanr, mannwhitneyu, kruskal

# Optional: ADF test (butuh statsmodels)
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# =========================================================
# 1) KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard KB Terintegrasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2) TEMA (CSS) - khusus tampilan
# =========================================================
CSS_THEME = """
<style>
:root{
  --bg: #f6f8fc;
  --card: #ffffff;
  --text: #0f172a;
  --muted: rgba(15,23,42,0.62);
  --border: rgba(15,23,42,0.10);
  --shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
  --shadow2: 0 10px 18px rgba(15, 23, 42, 0.08);
}

/* Background */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 500px at 15% 10%, rgba(59,130,246,0.20), transparent 60%),
    radial-gradient(900px 500px at 80% 18%, rgba(34,197,94,0.18), transparent 55%),
    radial-gradient(900px 500px at 70% 85%, rgba(168,85,247,0.14), transparent 60%),
    linear-gradient(180deg, #f6f8fc 0%, #eef4ff 100%);
}

/* Top padding (fix Streamlit Cloud header overlap) */
.block-container{
  padding-top: 68px !important;
  padding-bottom: 40px !important;
  max-width: 1280px;
}

/* Force text colors (light theme) */
section.main, section.main *{
  color: var(--text) !important;
}

/* Sidebar wider with text */
[data-testid="stSidebar"]{
  background: #0b1220;
  border-right: 1px solid rgba(255,255,255,0.10);
  min-width: 260px !important;
  max-width: 260px !important;
  width: 260px !important;
}

/* Sidebar title */
.sidebar-title{
  color: #ffffff !important;
  font-weight: 800;
  font-size: 20px;
  letter-spacing: 0.2px;
  padding: 14px 16px 6px 16px;
}
.sidebar-sub{
  color: rgba(255,255,255,0.65) !important;
  font-size: 12px;
  padding: 0px 16px 10px 16px;
}

/* Hide default radio label */
[data-testid="stSidebar"] .stRadio > label{ display:none; }

/* Sidebar menu items */
[data-testid="stSidebar"] div[role="radiogroup"]{
  padding: 8px 10px 12px 10px;
  gap: 8px;
}
[data-testid="stSidebar"] div[role="radiogroup"] label{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 10px 12px !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background: rgba(59,130,246,0.18);
  border: 1px solid rgba(59,130,246,0.28);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  color: rgba(255,255,255,0.92) !important;
  font-size: 14px !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label span{
  color: rgba(255,255,255,0.92) !important;
}

/* Sidebar footer */
.sidebar-foot{
  margin-top: 10px;
  padding: 10px 16px;
  color: rgba(255,255,255,0.60) !important;
  font-size: 12px;
  border-top: 1px solid rgba(255,255,255,0.10);
}

/* Topbar */
.topbar{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.topbar b{ color: var(--text) !important; }
.topbar .crumb{ color: var(--muted) !important; margin-left: 10px; }

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}

/* KPI colorful cards (custom HTML) */
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 10px;
  margin-bottom: 12px;
}
.kpi{
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow2);
  border: 1px solid rgba(255,255,255,0.35);
  position: relative;
  overflow: hidden;
}
.kpi .label{
  font-size: 13px;
  font-weight: 700;
  color: rgba(255,255,255,0.92) !important;
}
.kpi .value{
  margin-top: 6px;
  font-size: 34px;
  font-weight: 900;
  letter-spacing: -0.4px;
  color: rgba(255,255,255,1) !important;
}
.kpi .delta{
  margin-top: 4px;
  font-size: 12px;
  color: rgba(255,255,255,0.86) !important;
}
.kpi:after{
  content:"";
  position:absolute;
  right:-40px;
  top:-40px;
  width:140px;
  height:140px;
  border-radius:999px;
  background: rgba(255,255,255,0.18);
}
.kpi.blue{ background: linear-gradient(135deg, #2563eb, #60a5fa); }
.kpi.green{ background: linear-gradient(135deg, #16a34a, #86efac); }
.kpi.purple{ background: linear-gradient(135deg, #7c3aed, #c4b5fd); }
.kpi.orange{ background: linear-gradient(135deg, #f97316, #fdba74); }

/* Inputs in main */
section.main [data-baseweb="select"] > div{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
section.main [data-baseweb="select"] span{ color: var(--text) !important; }

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
[data-testid="stDataFrame"] *{ color: var(--text) !important; }
[data-testid="stDataFrame"] [role="grid"]{ background: var(--card) !important; }

/* Charts container spacing */
.chart-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 10px 12px;
  box-shadow: var(--shadow);
}

@media (max-width: 1100px){
  .kpi-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)


# =========================================================
# 3) KONSTANTA & FUNGSI BANTU
# =========================================================
MONTH_ORDER = [
    "JANUARI","FEBRUARI","MARET","APRIL","MEI","JUNI",
    "JULI","AGUSTUS","SEPTEMBER","OKTOBER","NOVEMBER","DESEMBER"
]

# Kolom detail stok pada DB1 (dipakai untuk hitung agregat SUNTIK/PIL/IMPLAN)
DB1_NUMERIC_COLUMNS = [
    "SUNTIKAN 1 BULANAN",
    "SUNTIKAN 3 BULANAN KOMBINASI",
    "SUNTIKAN 3 BULANAN PROGESTIN",
    "PIL KOMBINASI",
    "PIL PROGESTIN",
    "KONDOM",
    "IMPLAN 1 BATANG",
    "IMPLAN 2 BATANG",
    "IUD"
]

# Kolom stok agregat yang dipakai di dashboard
STOCK_METHODS = ["SUNTIK", "PIL", "IMPLAN", "KONDOM", "IUD"]
STOCK_METHODS_WITH_TOTAL = ["TOTAL_STOK"] + STOCK_METHODS


def normalize_text(x) -> str:
    """Rapikan teks: hapus spasi depan/belakang + ubah jadi HURUF BESAR."""
    return str(x).strip().upper()


def spearman_strength_label(rho: float) -> str:
    """Label kekuatan korelasi Spearman (berdasar nilai absolut rho)."""
    a = abs(rho)
    if a < 0.2: return "sangat lemah"
    if a < 0.4: return "lemah"
    if a < 0.6: return "sedang"
    if a < 0.8: return "kuat"
    return "sangat kuat"


def adf_test_result(series: pd.Series):
    """
    Uji stasioneritas ADF untuk deret waktu.
    Mengembalikan (p_value, kesimpulan).
    """
    if not HAS_STATSMODELS:
        return np.nan, "statsmodels belum terpasang"

    s = series.dropna().astype(float)
    if len(s) < 6:
        return np.nan, "data terlalu sedikit"

    p_value = adfuller(s)[1]
    conclusion = "stasioner" if p_value < 0.05 else "tidak stasioner (perlu differencing)"
    return p_value, conclusion


# =========================================================
# 4) MEMBACA & MENYIAPKAN DB1 (STOK)
# =========================================================
@st.cache_data(ttl=300)
def load_db1_stock_timeseries(excel_path_or_file) -> pd.DataFrame:
    """
    DB1 dibaca dari Excel multi-sheet.
    Nama sheet dianggap sebagai BULAN.
    """
    xls = pd.ExcelFile(excel_path_or_file)

    monthly_frames = []
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(excel_path_or_file, sheet_name=sheet_name)
        df_sheet["BULAN"] = normalize_text(sheet_name)
        monthly_frames.append(df_sheet)

    stock_raw = pd.concat(monthly_frames, ignore_index=True)

    # Validasi kolom numerik wajib ada
    missing_cols = [c for c in DB1_NUMERIC_COLUMNS if c not in stock_raw.columns]
    if missing_cols:
        raise ValueError(f"DB1: kolom numerik tidak ditemukan: {missing_cols}")

    # Pastikan kolom stok berupa angka
    stock_raw[DB1_NUMERIC_COLUMNS] = (
        stock_raw[DB1_NUMERIC_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # Buat agregat metode
    stock_raw["SUNTIK"] = (
        stock_raw["SUNTIKAN 1 BULANAN"]
        + stock_raw["SUNTIKAN 3 BULANAN KOMBINASI"]
        + stock_raw["SUNTIKAN 3 BULANAN PROGESTIN"]
    )
    stock_raw["PIL"] = stock_raw["PIL KOMBINASI"] + stock_raw["PIL PROGESTIN"]
    stock_raw["IMPLAN"] = stock_raw["IMPLAN 1 BATANG"] + stock_raw["IMPLAN 2 BATANG"]

    # Urutan bulan (supaya grafik deret waktu rapi)
    stock_raw["BULAN"] = pd.Categorical(
        stock_raw["BULAN"].apply(normalize_text),
        categories=MONTH_ORDER,
        ordered=True
    )

    # Pastikan ada kolom kabupaten untuk proses join
    if "KABUPATEN" not in stock_raw.columns:
        raise ValueError("DB1: kolom 'KABUPATEN' tidak ditemukan (dibutuhkan untuk keterkaitan).")

    stock_raw["KABUPATEN"] = stock_raw["KABUPATEN"].astype(str).str.strip().str.upper()
    return stock_raw


def get_stock_timeseries_for_kabupaten(stock_all: pd.DataFrame, kabupaten: str) -> pd.DataFrame:
    """Ambil stok per bulan untuk 1 kabupaten."""
    df = stock_all[stock_all["KABUPATEN"] == kabupaten][["BULAN"] + STOCK_METHODS].copy()
    return df.groupby("BULAN", as_index=False).sum().sort_values("BULAN")


def aggregate_stock_by_kabupaten(stock_all: pd.DataFrame) -> pd.DataFrame:
    """Agregasi stok setahun per kabupaten (menjumlahkan semua bulan)."""
    out = stock_all[["KABUPATEN"] + STOCK_METHODS].groupby("KABUPATEN", as_index=False).sum()
    out["TOTAL_STOK"] = out[STOCK_METHODS].sum(axis=1)
    return out


# =========================================================
# 5) MEMBACA & MENYIAPKAN DB2 (SDM + ADMIN)
# =========================================================
@st.cache_data(ttl=300)
def load_db2_people(excel_path_or_file) -> pd.DataFrame:
    """
    DB2 berisi jumlah tempat KB dan SDM per kabupaten.
    Menghasilkan kolom tambahan:
    - tenaga_kesehatan_total
    - sdm_per_tempat
    - admin_per_tempat
    """
    df = pd.read_excel(excel_path_or_file)

    # Samakan nama kolom agar gampang dipakai
    df.columns = [
        "kode", "kabupaten", "tempat_kb",
        "dok_kandungan", "dok_urologi", "dok_umum",
        "bidan", "perawat", "administrasi"
    ]

    # Buang baris yang kabupatennya kosong / header / bukan teks kabupaten
    df = df[df["kabupaten"].notna()]
    df["kabupaten"] = df["kabupaten"].astype(str).str.strip()
    df = df[df["kabupaten"].str.upper() != "KABUPATEN"]
    df = df[df["kabupaten"].str.isalpha()]
    df = df.reset_index(drop=True)

    # Ubah semua kolom angka jadi numeric
    numeric_cols = df.columns[2:]
    df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    # Total tenaga kesehatan = dokter + bidan + perawat
    df["tenaga_kesehatan_total"] = (
        df["dok_kandungan"].fillna(0)
        + df["dok_urologi"].fillna(0)
        + df["dok_umum"].fillna(0)
        + df["bidan"].fillna(0)
        + df["perawat"].fillna(0)
    )

    # Hindari pembagian nol (tempat_kb = 0)
    df["tempat_kb_safe"] = df["tempat_kb"].replace({0: np.nan})
    df["sdm_per_tempat"] = (df["tenaga_kesehatan_total"] / df["tempat_kb_safe"]).round(3)
    df["admin_per_tempat"] = (df["administrasi"] / df["tempat_kb_safe"]).round(3)

    # Format kolom integer (rapi)
    int_cols = [
        "tempat_kb", "dok_kandungan", "dok_urologi", "dok_umum",
        "bidan", "perawat", "administrasi", "tenaga_kesehatan_total"
    ]
    df[int_cols] = df[int_cols].round(0).astype("Int64")

    # Siapkan kunci join
    df["KABUPATEN"] = df["kabupaten"].str.upper()
    return df


# =========================================================
# 6) LOAD FILE DARI FOLDER REPO (data/)
# =========================================================
DB1_PATH = os.path.join("data", "DATA KETERSEDIAAN ALAT DAN OBAT KONTRASEPSI.xlsx")
DB2_PATH = os.path.join("data", "Jumlah tempat pelayanan kb yang memiliki tenaga kesehatan dan administrasi.xlsx")

if not os.path.exists(DB1_PATH):
    st.error(f"File DB1 tidak ditemukan: {DB1_PATH}")
    st.stop()

if not os.path.exists(DB2_PATH):
    st.error(f"File DB2 tidak ditemukan: {DB2_PATH}")
    st.stop()

# Baca data
stock_all_months_df = load_db1_stock_timeseries(DB1_PATH)
people_df = load_db2_people(DB2_PATH)

# Agregasi stok tahunan per kabupaten
stock_yearly_by_kab_df = aggregate_stock_by_kabupaten(stock_all_months_df)

# Gabungkan (join) DB1 + DB2 berdasarkan kabupaten yang sama
integrated_df = people_df.merge(stock_yearly_by_kab_df, on="KABUPATEN", how="inner")

kabupaten_list = sorted(integrated_df["KABUPATEN"].unique().tolist())
if not kabupaten_list:
    st.error("Tidak ada kabupaten yang terhubung. Pastikan penulisan kabupaten DB1 & DB2 sama.")
    st.stop()


# =========================================================
# 7) SIDEBAR MENU
# =========================================================
MENU_ITEMS = [
    ("SUMMARY",  "📊  Dashboard"),
    ("TS",       "📈  Deret Waktu"),
    ("PEOPLE",   "👥  People Analytics"),
    ("LINK",     "🔗  Keterkaitan"),
    ("KRUSKAL",  "🧪  Kruskal–Wallis"),
    ("DATASET",  "🗂️  Dataset"),
]
MENU_LABEL = {key: label for key, label in MENU_ITEMS}

with st.sidebar:
    st.markdown("<div class='sidebar-title'>KB Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Stok × SDM × Administrasi</div>", unsafe_allow_html=True)

    active_menu = st.radio(
        "Menu",
        [k for k, _ in MENU_ITEMS],
        format_func=lambda k: MENU_LABEL[k],
        label_visibility="collapsed"
    )

    st.markdown(
        "<div class='sidebar-foot'>Data dibaca dari folder <b>data/</b> di repo.</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 8) TOPBAR + FILTER KABUPATEN
# =========================================================
top_left, top_right = st.columns([2.4, 1.2], vertical_alignment="center")

with top_left:
    crumb = MENU_LABEL[active_menu]
    # hilangkan emoji di crumb agar lebih clean
    for emoji in ["📊", "📈", "👥", "🔗", "🧪", "🗂️"]:
        crumb = crumb.replace(emoji, "").strip()

    st.markdown(
        f"<div class='topbar'><b>Dashboard KB Terintegrasi</b>"
        f"<span class='crumb'>{crumb}</span></div>",
        unsafe_allow_html=True
    )

with top_right:
    selected_kabupaten = st.selectbox("Kabupaten", kabupaten_list, index=0)


# =========================================================
# 9) KPI UTAMA (RINGKASAN ANGKA)
# =========================================================
jumlah_kabupaten_terhubung = integrated_df["KABUPATEN"].nunique()
total_tempat_kb = int(integrated_df["tempat_kb"].fillna(0).sum())
total_tenaga_kesehatan = int(integrated_df["tenaga_kesehatan_total"].fillna(0).sum())
total_stok_setahun = float(integrated_df["TOTAL_STOK"].fillna(0).sum())

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi blue">
    <div class="label">Kabupaten Terhubung</div>
    <div class="value">{jumlah_kabupaten_terhubung:,}</div>
    <div class="delta">hasil join DB1 & DB2</div>
  </div>
  <div class="kpi green">
    <div class="label">Total Tempat KB</div>
    <div class="value">{total_tempat_kb:,}</div>
    <div class="delta">akumulasi seluruh kabupaten</div>
  </div>
  <div class="kpi purple">
    <div class="label">Total Tenaga Kesehatan</div>
    <div class="value">{total_tenaga_kesehatan:,}</div>
    <div class="delta">dokter + bidan + perawat</div>
  </div>
  <div class="kpi orange">
    <div class="label">Total Stok (Setahun)</div>
    <div class="value">{total_stok_setahun:,.0f}</div>
    <div class="delta">Suntik+Pil+Implan+Kondom+IUD</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# 10) ISI HALAMAN (BERDASARKAN MENU)
# =========================================================
if active_menu == "SUMMARY":
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='card'><b>Top 10 — Tenaga Kesehatan Total</b></div>", unsafe_allow_html=True)
        top10_sdm = integrated_df.sort_values("tenaga_kesehatan_total", ascending=False).head(10)
        st.dataframe(
            top10_sdm[["KABUPATEN", "tempat_kb", "tenaga_kesehatan_total", "administrasi"]],
            use_container_width=True,
            height=360
        )

        st.markdown("<div class='chart-card'><b>Grafik Top 10 — Tenaga Kesehatan</b></div>", unsafe_allow_html=True)
        st.bar_chart(top10_sdm.set_index("KABUPATEN")[["tenaga_kesehatan_total"]], use_container_width=True)

    with right:
        st.markdown("<div class='card'><b>Top 10 — Total Stok Setahun</b></div>", unsafe_allow_html=True)
        top10_stok = integrated_df.sort_values("TOTAL_STOK", ascending=False).head(10)
        st.dataframe(
            top10_stok[["KABUPATEN", "TOTAL_STOK", "SUNTIK", "PIL", "IMPLAN", "KONDOM", "IUD"]],
            use_container_width=True,
            height=360
        )

        st.markdown("<div class='chart-card'><b>Grafik Top 10 — Total Stok</b></div>", unsafe_allow_html=True)
        st.bar_chart(top10_stok.set_index("KABUPATEN")[["TOTAL_STOK"]], use_container_width=True)


elif active_menu == "TS":
    st.markdown(
        f"<div class='card'><b>Deret Waktu Persediaan</b><br/>"
        f"<span style='color:rgba(15,23,42,0.62)'>{selected_kabupaten}</span></div>",
        unsafe_allow_html=True
    )

    ts_df = get_stock_timeseries_for_kabupaten(stock_all_months_df, selected_kabupaten)

    selected_stock_vars = st.multiselect("Variabel stok", STOCK_METHODS, default=STOCK_METHODS)

    if selected_stock_vars:
        st.markdown("<div class='chart-card'><b>Grafik Deret Waktu</b></div>", unsafe_allow_html=True)
        st.line_chart(ts_df.set_index("BULAN")[selected_stock_vars], use_container_width=True)
    else:
        st.info("Pilih minimal 1 variabel.")

    # Moving Average (3 bulan)
    st.markdown("<div class='chart-card'><b>Moving Average (3 bulan)</b></div>", unsafe_allow_html=True)
    ma_df = ts_df.copy()
    for v in STOCK_METHODS:
        ma_df[f"MA3_{v}"] = ma_df[v].rolling(3).mean()

    show_ma_cols = [f"MA3_{v}" for v in selected_stock_vars] if selected_stock_vars else [f"MA3_{v}" for v in STOCK_METHODS]
    st.line_chart(ma_df.set_index("BULAN")[show_ma_cols], use_container_width=True)

    # ADF
    st.markdown("<div class='card'><b>Uji stasioneritas (ADF)</b></div>", unsafe_allow_html=True)
    rows = []
    for v in STOCK_METHODS:
        p_value, conclusion = adf_test_result(ts_df[v])
        rows.append({"Variabel": v, "p-value": p_value, "Kesimpulan": conclusion})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


elif active_menu == "PEOPLE":
    st.markdown(
        f"<div class='card'><b>People Analytics</b><br/>"
        f"<span style='color:rgba(15,23,42,0.62)'>{selected_kabupaten}</span></div>",
        unsafe_allow_html=True
    )

    selected_row = integrated_df[integrated_df["KABUPATEN"] == selected_kabupaten].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tempat KB", f"{int(selected_row['tempat_kb']) if pd.notna(selected_row['tempat_kb']) else 0:,}")
    c2.metric("Tenaga Kesehatan Total", f"{int(selected_row['tenaga_kesehatan_total']) if pd.notna(selected_row['tenaga_kesehatan_total']) else 0:,}")
    c3.metric("Administrasi", f"{int(selected_row['administrasi']) if pd.notna(selected_row['administrasi']) else 0:,}")
    c4.metric("Total Stok Setahun", f"{float(selected_row['TOTAL_STOK']):,.0f}")

    composition_df = pd.DataFrame({
        "Metode": STOCK_METHODS,
        "Total Setahun": [float(selected_row[v]) for v in STOCK_METHODS]
    })

    st.markdown("<div class='card'><b>Komposisi stok setahun</b></div>", unsafe_allow_html=True)
    st.dataframe(composition_df, use_container_width=True)

    st.markdown("<div class='chart-card'><b>Grafik komposisi stok</b></div>", unsafe_allow_html=True)
    st.bar_chart(composition_df.set_index("Metode"), use_container_width=True)

    st.markdown("<div class='card'><b>Deskriptif variabel kunci</b></div>", unsafe_allow_html=True)
    st.dataframe(
        integrated_df[["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat", "TOTAL_STOK"]].describe(),
        use_container_width=True
    )


elif active_menu == "LINK":
    st.markdown(
        "<div class='card'><b>Analisis Keterkaitan</b><br/>"
        "<span style='color:rgba(15,23,42,0.62)'>Spearman + Scatter</span></div>",
        unsafe_allow_html=True
    )

    people_x_options = ["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat"]
    stock_y_options = STOCK_METHODS_WITH_TOTAL

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variabel People (X)", people_x_options, index=1)
    with col2:
        y_var = st.selectbox("Variabel Stok (Y)", stock_y_options, index=0)

    valid_df = integrated_df[[x_var, y_var, "KABUPATEN", "administrasi"]].dropna()

    if len(valid_df) < 5:
        st.warning("Data valid terlalu sedikit untuk analisis korelasi.")
    else:
        rho, p_value = spearmanr(valid_df[x_var], valid_df[y_var], nan_policy="omit")
        a, b, c = st.columns(3)
        a.metric("Spearman rho", f"{rho:.3f}")
        b.metric("p-value", f"{p_value:.4f}")
        c.metric("Kekuatan", spearman_strength_label(rho))

        st.markdown("<div class='chart-card'><b>Scatter</b></div>", unsafe_allow_html=True)
        st.scatter_chart(valid_df.set_index("KABUPATEN")[[x_var, y_var]], use_container_width=True)

    # Mann–Whitney: bandingkan stok ketika admin ada vs tidak
    st.markdown("<div class='card'><b>Uji beda (Mann–Whitney): Admin ada vs tidak</b></div>", unsafe_allow_html=True)

    mw_df = integrated_df[[y_var, "administrasi"]].dropna()
    group_admin_exists = mw_df[mw_df["administrasi"].astype(float) > 0][y_var].astype(float)
    group_admin_none = mw_df[mw_df["administrasi"].astype(float) == 0][y_var].astype(float)

    if len(group_admin_exists) == 0 or len(group_admin_none) == 0:
        st.warning("Tidak cukup data untuk membagi grup admin > 0 vs admin = 0.")
    else:
        u_stat, p_mw = mannwhitneyu(group_admin_exists, group_admin_none, alternative="two-sided")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("U", f"{u_stat:.3f}")
        m2.metric("p-value", f"{p_mw:.4f}")
        m3.metric("Median (Admin ada)", f"{float(np.median(group_admin_exists)):.3f}")
        m4.metric("Median (Admin tidak)", f"{float(np.median(group_admin_none)):.3f}")


elif active_menu == "KRUSKAL":
    st.markdown(
        "<div class='card'><b>Uji Kruskal–Wallis</b><br/>"
        "<span style='color:rgba(15,23,42,0.62)'>DB2 (Rasio) & DB1 (Stok) berdasar kategori (Rendah/Sedang/Tinggi)</span></div>",
        unsafe_allow_html=True
    )
    st.write("")

    group_base = st.selectbox(
        "Kelompokkan berdasarkan (DB2)",
        ["tempat_kb", "tenaga_kesehatan_total", "administrasi"],
        index=0
    )

    # Buat kategori 3 level (Rendah/Sedang/Tinggi)
    try:
        integrated_df["_kategori_base"] = pd.qcut(
            integrated_df[group_base].astype(float),
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )
    except Exception:
        integrated_df["_kategori_base"] = pd.cut(
            integrated_df[group_base].astype(float),
            bins=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )

    tab1, tab2 = st.tabs(["Kruskal DB2 (People)", "Kruskal DB1 (Stok)"])

    with tab1:
        st.markdown("<div class='card'><b>DB2 — Rasio per Fasilitas</b></div>", unsafe_allow_html=True)
        y_people = st.selectbox("Variabel people", ["sdm_per_tempat", "admin_per_tempat"], index=0)

        d = integrated_df[[y_people, "_kategori_base"]].dropna()
        labels = ["Rendah", "Sedang", "Tinggi"]
        groups = [d[d["_kategori_base"] == lab][y_people].astype(float).dropna().values for lab in labels]

        if sum(len(g) > 0 for g in groups) < 2:
            st.warning("Data tidak cukup untuk Kruskal (minimal 2 grup).")
        else:
            g_use = [g for g in groups if len(g) > 0]
            h_stat, p_kw = kruskal(*g_use)

            a, b = st.columns(2)
            a.metric("H statistic", f"{h_stat:.3f}")
            b.metric("p-value", f"{p_kw:.4f}")

            if p_kw < 0.05:
                st.success("Ada perbedaan signifikan antar kategori (α=5%).")
            else:
                st.info("Tidak ada perbedaan signifikan antar kategori (α=5%).")

            med = d.groupby("_kategori_base")[y_people].median().reindex(labels)
            st.dataframe(
                med.reset_index().rename(columns={"_kategori_base": "Kategori", y_people: "Median"}),
                use_container_width=True
            )
            st.markdown("<div class='chart-card'><b>Grafik median</b></div>", unsafe_allow_html=True)
            st.bar_chart(med, use_container_width=True)

    with tab2:
        st.markdown("<div class='card'><b>DB1 — Stok Kontrasepsi</b></div>", unsafe_allow_html=True)
        y_stok = st.selectbox("Variabel stok", ["TOTAL_STOK", "SUNTIK", "PIL", "IMPLAN", "KONDOM", "IUD"], index=0)

        d = integrated_df[[y_stok, "_kategori_base"]].dropna()
        labels = ["Rendah", "Sedang", "Tinggi"]
        groups = [d[d["_kategori_base"] == lab][y_stok].astype(float).dropna().values for lab in labels]

        if sum(len(g) > 0 for g in groups) < 2:
            st.warning("Data tidak cukup untuk Kruskal (minimal 2 grup).")
        else:
            g_use = [g for g in groups if len(g) > 0]
            h_stat, p_kw = kruskal(*g_use)

            a, b = st.columns(2)
            a.metric("H statistic", f"{h_stat:.3f}")
            b.metric("p-value", f"{p_kw:.4f}")

            if p_kw < 0.05:
                st.success("Ada perbedaan signifikan stok antar kategori (α=5%).")
            else:
                st.info("Tidak ada perbedaan signifikan stok antar kategori (α=5%).")

            med = d.groupby("_kategori_base")[y_stok].median().reindex(labels)
            st.dataframe(
                med.reset_index().rename(columns={"_kategori_base": "Kategori", y_stok: "Median"}),
                use_container_width=True
            )
            st.markdown("<div class='chart-card'><b>Grafik median stok</b></div>", unsafe_allow_html=True)
            st.bar_chart(med, use_container_width=True)

    # Bersihkan kolom sementara
    if "_kategori_base" in integrated_df.columns:
        integrated_df.drop(columns=["_kategori_base"], inplace=True)


elif active_menu == "DATASET":
    st.markdown("<div class='card'><b>Dataset Terintegrasi (hasil join)</b></div>", unsafe_allow_html=True)

    st.dataframe(integrated_df, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download dataset terintegrasi (CSV)",
        data=integrated_df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_terintegrasi_kb.csv",
        mime="text/csv"
    )
