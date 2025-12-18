import os
import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import spearmanr, mannwhitneyu, kruskal

# optional (ADF)
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard KB Terintegrasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# THEME CSS (COLORFUL)
# =========================
st.markdown(
    """
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
""",
    unsafe_allow_html=True
)

# =========================
# CONSTANTS
# =========================
URUTAN_BULAN = [
    "JANUARI","FEBRUARI","MARET","APRIL","MEI","JUNI",
    "JULI","AGUSTUS","SEPTEMBER","OKTOBER","NOVEMBER","DESEMBER"
]
KOLUM_NUMERIK_DB1 = [
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
VAR_STOK = ["SUNTIK", "PIL", "IMPLAN", "KONDOM", "IUD"]
VAR_ALL_STOK = ["TOTAL_STOK"] + VAR_STOK

def norm_str(x):
    return str(x).strip().upper()

# =========================
# LOAD DB1
# =========================
@st.cache_data(ttl=300)
def load_db1_time_series(path_or_file) -> pd.DataFrame:
    xls = pd.ExcelFile(path_or_file)
    all_data = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path_or_file, sheet_name=sheet)
        df["BULAN"] = norm_str(sheet)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    miss = [c for c in KOLUM_NUMERIK_DB1 if c not in df_all.columns]
    if miss:
        raise ValueError(f"DB1: kolom numerik tidak ditemukan: {miss}")

    df_all[KOLUM_NUMERIK_DB1] = df_all[KOLUM_NUMERIK_DB1].apply(pd.to_numeric, errors="coerce").fillna(0)

    df_all["SUNTIK"] = (
        df_all["SUNTIKAN 1 BULANAN"]
        + df_all["SUNTIKAN 3 BULANAN KOMBINASI"]
        + df_all["SUNTIKAN 3 BULANAN PROGESTIN"]
    )
    df_all["PIL"] = df_all["PIL KOMBINASI"] + df_all["PIL PROGESTIN"]
    df_all["IMPLAN"] = df_all["IMPLAN 1 BATANG"] + df_all["IMPLAN 2 BATANG"]

    df_all["BULAN"] = pd.Categorical(
        df_all["BULAN"].apply(norm_str),
        categories=URUTAN_BULAN,
        ordered=True
    )

    if "KABUPATEN" not in df_all.columns:
        raise ValueError("DB1: kolom 'KABUPATEN' tidak ditemukan (dibutuhkan untuk keterkaitan).")

    df_all["KABUPATEN"] = df_all["KABUPATEN"].astype(str).str.strip().str.upper()
    return df_all

def db1_ts_per_kab(df_all: pd.DataFrame, kabupaten: str) -> pd.DataFrame:
    df_k = df_all[df_all["KABUPATEN"] == kabupaten][["BULAN"] + VAR_STOK].copy()
    return df_k.groupby("BULAN", as_index=False).sum().sort_values("BULAN")

def db1_agg_per_kab(df_all: pd.DataFrame) -> pd.DataFrame:
    out = df_all[["KABUPATEN"] + VAR_STOK].groupby("KABUPATEN", as_index=False).sum()
    out["TOTAL_STOK"] = out[VAR_STOK].sum(axis=1)
    return out

def adf_status(series: pd.Series):
    if not HAS_STATSMODELS:
        return np.nan, "statsmodels belum terpasang"
    s = series.dropna().astype(float)
    if len(s) < 6:
        return np.nan, "data terlalu sedikit"
    p = adfuller(s)[1]
    return p, ("stasioner" if p < 0.05 else "tidak stasioner (perlu differencing)")

# =========================
# LOAD DB2
# =========================
@st.cache_data(ttl=300)
def load_db2_people(path_or_file) -> pd.DataFrame:
    df = pd.read_excel(path_or_file)
    df.columns = [
        "kode","kabupaten","tempat_kb","dok_kandungan","dok_urologi",
        "dok_umum","bidan","perawat","administrasi"
    ]

    df = df[df["kabupaten"].notna()]
    df["kabupaten"] = df["kabupaten"].astype(str).str.strip()
    df = df[df["kabupaten"].str.upper() != "KABUPATEN"]
    df = df[df["kabupaten"].str.isalpha()]
    df = df.reset_index(drop=True)

    num_cols = df.columns[2:]
    df[num_cols] = df[num_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    df["tenaga_kesehatan_total"] = (
        df["dok_kandungan"].fillna(0)
        + df["dok_urologi"].fillna(0)
        + df["dok_umum"].fillna(0)
        + df["bidan"].fillna(0)
        + df["perawat"].fillna(0)
    )

    df["tempat_kb_safe"] = df["tempat_kb"].replace({0: np.nan})
    df["sdm_per_tempat"] = (df["tenaga_kesehatan_total"] / df["tempat_kb_safe"]).round(3)
    df["admin_per_tempat"] = (df["administrasi"] / df["tempat_kb_safe"]).round(3)

    int_cols = ["tempat_kb","dok_kandungan","dok_urologi","dok_umum","bidan","perawat","administrasi","tenaga_kesehatan_total"]
    df[int_cols] = df[int_cols].round(0).astype("Int64")

    df["KABUPATEN"] = df["kabupaten"].str.upper()
    return df

def spearman_strength(rho: float) -> str:
    a = abs(rho)
    if a < 0.2: return "sangat lemah"
    if a < 0.4: return "lemah"
    if a < 0.6: return "sedang"
    if a < 0.8: return "kuat"
    return "sangat kuat"

# =========================
# LOAD FROM REPO
# =========================
DB1_PATH = os.path.join("data", "DATA KETERSEDIAAN ALAT DAN OBAT KONTRASEPSI.xlsx")
DB2_PATH = os.path.join("data", "Jumlah tempat pelayanan kb yang memiliki tenaga kesehatan dan administrasi.xlsx")

if not os.path.exists(DB1_PATH):
    st.error(f"File DB1 tidak ditemukan: {DB1_PATH}")
    st.stop()
if not os.path.exists(DB2_PATH):
    st.error(f"File DB2 tidak ditemukan: {DB2_PATH}")
    st.stop()

df1_all = load_db1_time_series(DB1_PATH)
df2 = load_db2_people(DB2_PATH)
df1_kab = db1_agg_per_kab(df1_all)
df_int = df2.merge(df1_kab, on="KABUPATEN", how="inner")

kab_list = sorted(df_int["KABUPATEN"].unique().tolist())
if not kab_list:
    st.error("Tidak ada kabupaten yang terhubung. Pastikan penulisan kabupaten DB1 & DB2 sama.")
    st.stop()

# =========================
# SIDEBAR MENU (ICON + TEXT)
# =========================
MENU = [
    ("SUMMARY", "📊  Dashboard"),
    ("TS", "📈  Deret Waktu"),
    ("PEOPLE", "👥  People Analytics"),
    ("LINK", "🔗  Keterkaitan"),
    ("KRUSKAL", "🧪  Kruskal–Wallis"),
    ("DATASET", "🗂️  Dataset")
]
MENU_KEY_TO_NAME = {k: v for k, v in MENU}

with st.sidebar:
    st.markdown("<div class='sidebar-title'>KB Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Stok × SDM × Administrasi</div>", unsafe_allow_html=True)

    menu_key = st.radio(
        "Menu",
        [k for k, _ in MENU],
        format_func=lambda k: MENU_KEY_TO_NAME[k],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-foot'>Data dibaca dari folder <b>data/</b> di repo.</div>", unsafe_allow_html=True)

# =========================
# TOPBAR + FILTER KABUPATEN (KANAN ATAS)
# =========================
top_l, top_r = st.columns([2.4, 1.2], vertical_alignment="center")
with top_l:
    st.markdown(
        f"<div class='topbar'><b>Dashboard KB Terintegrasi</b>"
        f"<span class='crumb'>{MENU_KEY_TO_NAME[menu_key].replace('📊  ','').replace('📈  ','').replace('👥  ','').replace('🔗  ','').replace('🧪  ','').replace('🗂️  ','')}</span></div>",
        unsafe_allow_html=True
    )
with top_r:
    kab = st.selectbox("Kabupaten", kab_list, index=0)

# =========================
# COLORFUL KPI CARDS
# =========================
kab_count = df_int["KABUPATEN"].nunique()
tot_tempat = int(df_int["tempat_kb"].fillna(0).sum())
tot_sdm = int(df_int["tenaga_kesehatan_total"].fillna(0).sum())
tot_stok = float(df_int["TOTAL_STOK"].fillna(0).sum())

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi blue">
    <div class="label">Kabupaten Terhubung</div>
    <div class="value">{kab_count:,}</div>
    <div class="delta">hasil join DB1 & DB2</div>
  </div>
  <div class="kpi green">
    <div class="label">Total Tempat KB</div>
    <div class="value">{tot_tempat:,}</div>
    <div class="delta">akumulasi seluruh kabupaten</div>
  </div>
  <div class="kpi purple">
    <div class="label">Total Tenaga Kesehatan</div>
    <div class="value">{tot_sdm:,}</div>
    <div class="delta">dokter + bidan + perawat</div>
  </div>
  <div class="kpi orange">
    <div class="label">Total Stok (Setahun)</div>
    <div class="value">{tot_stok:,.0f}</div>
    <div class="delta">Suntik+Pil+Implan+Kondom+IUD</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# =========================
# CONTENT
# =========================
if menu_key == "SUMMARY":
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='card'><b>Top 10 — Tenaga Kesehatan Total</b></div>", unsafe_allow_html=True)
        top10_sdm = df_int.sort_values("tenaga_kesehatan_total", ascending=False).head(10)
        st.dataframe(top10_sdm[["KABUPATEN","tempat_kb","tenaga_kesehatan_total","administrasi"]], use_container_width=True, height=360)

        st.markdown("<div class='chart-card'><b>Grafik Top 10 — Tenaga Kesehatan</b></div>", unsafe_allow_html=True)
        st.bar_chart(top10_sdm.set_index("KABUPATEN")[["tenaga_kesehatan_total"]], use_container_width=True)

    with right:
        st.markdown("<div class='card'><b>Top 10 — Total Stok Setahun</b></div>", unsafe_allow_html=True)
        top10_stok = df_int.sort_values("TOTAL_STOK", ascending=False).head(10)
        st.dataframe(top10_stok[["KABUPATEN","TOTAL_STOK","SUNTIK","PIL","IMPLAN","KONDOM","IUD"]], use_container_width=True, height=360)

        st.markdown("<div class='chart-card'><b>Grafik Top 10 — Total Stok</b></div>", unsafe_allow_html=True)
        st.bar_chart(top10_stok.set_index("KABUPATEN")[["TOTAL_STOK"]], use_container_width=True)

elif menu_key == "TS":
    st.markdown(
        f"<div class='card'><b>Deret Waktu Persediaan</b><br/>"
        f"<span style='color:rgba(15,23,42,0.62)'>{kab}</span></div>",
        unsafe_allow_html=True
    )

    df_ts = db1_ts_per_kab(df1_all, kab)

    vsel = st.multiselect("Variabel stok", VAR_STOK, default=VAR_STOK)
    if vsel:
        st.markdown("<div class='chart-card'><b>Grafik Deret Waktu</b></div>", unsafe_allow_html=True)
        st.line_chart(df_ts.set_index("BULAN")[vsel], use_container_width=True)
    else:
        st.info("Pilih minimal 1 variabel.")

    st.markdown("<div class='chart-card'><b>Moving Average (3 bulan)</b></div>", unsafe_allow_html=True)
    ma_df = df_ts.copy()
    for v in VAR_STOK:
        ma_df[f"MA3_{v}"] = ma_df[v].rolling(3).mean()
    show_ma = [f"MA3_{v}" for v in vsel] if vsel else [f"MA3_{v}" for v in VAR_STOK]
    st.line_chart(ma_df.set_index("BULAN")[show_ma], use_container_width=True)

    st.markdown("<div class='card'><b>Uji stasioneritas (ADF)</b></div>", unsafe_allow_html=True)
    rows = []
    for v in VAR_STOK:
        p, status = adf_status(df_ts[v])
        rows.append({"Variabel": v, "p-value": p, "Kesimpulan": status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif menu_key == "PEOPLE":
    st.markdown(
        f"<div class='card'><b>People Analytics</b><br/>"
        f"<span style='color:rgba(15,23,42,0.62)'>{kab}</span></div>",
        unsafe_allow_html=True
    )

    row = df_int[df_int["KABUPATEN"] == kab].iloc[0]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Tempat KB", f"{int(row['tempat_kb']) if pd.notna(row['tempat_kb']) else 0:,}")
    p2.metric("Tenaga Kesehatan Total", f"{int(row['tenaga_kesehatan_total']) if pd.notna(row['tenaga_kesehatan_total']) else 0:,}")
    p3.metric("Administrasi", f"{int(row['administrasi']) if pd.notna(row['administrasi']) else 0:,}")
    p4.metric("Total Stok Setahun", f"{float(row['TOTAL_STOK']):,.0f}")

    comp = pd.DataFrame({"Metode": VAR_STOK, "Total Setahun": [float(row[v]) for v in VAR_STOK]})
    st.markdown("<div class='card'><b>Komposisi stok setahun</b></div>", unsafe_allow_html=True)
    st.dataframe(comp, use_container_width=True)

    st.markdown("<div class='chart-card'><b>Grafik komposisi stok</b></div>", unsafe_allow_html=True)
    st.bar_chart(comp.set_index("Metode"), use_container_width=True)

    st.markdown("<div class='card'><b>Deskriptif variabel kunci</b></div>", unsafe_allow_html=True)
    st.dataframe(
        df_int[["tempat_kb","tenaga_kesehatan_total","administrasi","sdm_per_tempat","admin_per_tempat","TOTAL_STOK"]].describe(),
        use_container_width=True
    )

elif menu_key == "LINK":
    st.markdown("<div class='card'><b>Analisis Keterkaitan</b><br/><span style='color:rgba(15,23,42,0.62)'>Spearman + Scatter</span></div>", unsafe_allow_html=True)

    x_opts = ["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat"]
    y_opts = VAR_ALL_STOK

    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("Variabel People (X)", x_opts, index=1)
    with c2:
        y_var = st.selectbox("Variabel Stok (Y)", y_opts, index=0)

    d = df_int[[x_var, y_var, "KABUPATEN", "administrasi"]].dropna()
    if len(d) < 5:
        st.warning("Data valid terlalu sedikit untuk analisis korelasi.")
    else:
        rho, p = spearmanr(d[x_var], d[y_var], nan_policy="omit")
        a, b, c = st.columns(3)
        a.metric("Spearman rho", f"{rho:.3f}")
        b.metric("p-value", f"{p:.4f}")
        c.metric("Kekuatan", spearman_strength(rho))
        st.markdown("<div class='chart-card'><b>Scatter</b></div>", unsafe_allow_html=True)
        st.scatter_chart(d.set_index("KABUPATEN")[[x_var, y_var]], use_container_width=True)

    st.markdown("<div class='card'><b>Uji beda (Mann–Whitney): Admin ada vs tidak</b></div>", unsafe_allow_html=True)
    d2 = df_int[[y_var, "administrasi"]].dropna()
    g_admin = d2[d2["administrasi"].astype(float) > 0][y_var].astype(float)
    g_no = d2[d2["administrasi"].astype(float) == 0][y_var].astype(float)

    if len(g_admin) == 0 or len(g_no) == 0:
        st.warning("Tidak cukup data untuk membagi grup admin > 0 vs admin = 0.")
    else:
        u, p_mw = mannwhitneyu(g_admin, g_no, alternative="two-sided")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("U", f"{u:.3f}")
        m2.metric("p-value", f"{p_mw:.4f}")
        m3.metric("Median (Admin ada)", f"{float(np.median(g_admin)):.3f}")
        m4.metric("Median (Admin tidak)", f"{float(np.median(g_no)):.3f}")

elif menu_key == "KRUSKAL":
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

    # kategori 3 level
    try:
        df_int["_kategori_base"] = pd.qcut(
            df_int[group_base].astype(float),
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )
    except Exception:
        df_int["_kategori_base"] = pd.cut(
            df_int[group_base].astype(float),
            bins=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )

    tab1, tab2 = st.tabs(["Kruskal DB2 (People)", "Kruskal DB1 (Stok)"])

    with tab1:
        st.markdown("<div class='card'><b>DB2 — Rasio per Fasilitas</b></div>", unsafe_allow_html=True)
        y_people = st.selectbox("Variabel people", ["sdm_per_tempat", "admin_per_tempat"], index=0)

        d = df_int[[y_people, "_kategori_base"]].dropna()
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
            st.dataframe(med.reset_index().rename(columns={"_kategori_base":"Kategori", y_people:"Median"}), use_container_width=True)
            st.markdown("<div class='chart-card'><b>Grafik median</b></div>", unsafe_allow_html=True)
            st.bar_chart(med, use_container_width=True)

    with tab2:
        st.markdown("<div class='card'><b>DB1 — Stok Kontrasepsi</b></div>", unsafe_allow_html=True)
        y_stok = st.selectbox("Variabel stok", ["TOTAL_STOK","SUNTIK","PIL","IMPLAN","KONDOM","IUD"], index=0)

        d = df_int[[y_stok, "_kategori_base"]].dropna()
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
            st.dataframe(med.reset_index().rename(columns={"_kategori_base":"Kategori", y_stok:"Median"}), use_container_width=True)
            st.markdown("<div class='chart-card'><b>Grafik median stok</b></div>", unsafe_allow_html=True)
            st.bar_chart(med, use_container_width=True)

    if "_kategori_base" in df_int.columns:
        df_int.drop(columns=["_kategori_base"], inplace=True)

elif menu_key == "DATASET":
    st.markdown("<div class='card'><b>Dataset Terintegrasi (hasil join)</b></div>", unsafe_allow_html=True)
    st.dataframe(df_int, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download dataset terintegrasi (CSV)",
        data=df_int.to_csv(index=False).encode("utf-8"),
        file_name="dataset_terintegrasi_kb.csv",
        mime="text/csv"
    )
