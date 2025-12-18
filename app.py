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
    initial_sidebar_state="expanded"  # <-- pastikan sidebar muncul
)

# =========================
# CSS: LIGHT + BLUE THEME
# (JANGAN hide header total supaya tombol sidebar tetap ada)
# =========================
st.markdown(
    """
<style>
:root{
  --bg1: #f3f7ff;
  --bg2: #eef2ff;
  --card: #ffffff;
  --text: #0f172a;
  --muted: rgba(15,23,42,0.62);
  --border: rgba(15,23,42,0.10);
  --shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
  --blue: #2563eb;
}

/* App background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(37,99,235,0.20), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(96,165,250,0.18), transparent 55%),
              linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

/* Force dark text in MAIN */
section.main, section.main *{
  color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: var(--card);
  border-right: 1px solid var(--border);
}

/* Make sidebar narrow (icon bar) */
[data-testid="stSidebar"]{
  min-width: 90px !important;
  max-width: 90px !important;
  width: 90px !important;
}

/* Radio group inside sidebar: icon buttons */
[data-testid="stSidebar"] div[role="radiogroup"]{ gap: 10px; }
[data-testid="stSidebar"] div[role="radiogroup"] label{
  justify-content: center !important;
  padding: 10px 0px !important;
  border-radius: 14px !important;
  margin: 0px 10px !important;
  background: transparent;
  border: 1px solid transparent;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background: rgba(37,99,235,0.10);
  border: 1px solid rgba(37,99,235,0.20);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  font-size: 20px !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio > label{ display: none; }

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

/* Metrics */
[data-testid="stMetric"]{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: var(--shadow);
}
section.main [data-testid="stMetricLabel"]{ color: var(--muted) !important; }
section.main [data-testid="stMetricValue"]{ color: var(--text) !important; }

/* Inputs */
section.main [data-baseweb="select"] > div{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
section.main [data-baseweb="select"] span{ color: var(--text) !important; }

/* Tags */
section.main span[data-baseweb="tag"]{
  background: rgba(37,99,235,0.10) !important;
  border: 1px solid rgba(37,99,235,0.25) !important;
  border-radius: 999px !important;
}
section.main span[data-baseweb="tag"] *{ color: var(--blue) !important; }

/* Dataframe (force light) */
[data-testid="stDataFrame"]{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
[data-testid="stDataFrame"] *{ color: var(--text) !important; }
[data-testid="stDataFrame"] [role="grid"]{ background: var(--card) !important; }

/* Layout width */
.block-container{
  padding-top: 12px;
  max-width: 1220px;
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
# LOAD & PREP DB1
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
    out = df_k.groupby("BULAN", as_index=False).sum().sort_values("BULAN")
    return out

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
    status = "stasioner" if p < 0.05 else "tidak stasioner (perlu differencing)"
    return p, status

# =========================
# LOAD & PREP DB2
# =========================
@st.cache_data(ttl=300)
def load_db2_people(path_or_file) -> pd.DataFrame:
    df = pd.read_excel(path_or_file)

    df.columns = [
        "kode",
        "kabupaten",
        "tempat_kb",
        "dok_kandungan",
        "dok_urologi",
        "dok_umum",
        "bidan",
        "perawat",
        "administrasi"
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

    int_cols = [
        "tempat_kb","dok_kandungan","dok_urologi",
        "dok_umum","bidan","perawat","administrasi",
        "tenaga_kesehatan_total"
    ]
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
# LOAD DATA FROM REPO
# =========================
DB1_PATH = os.path.join("data", "DATA KETERSEDIAAN ALAT DAN OBAT KONTRASEPSI.xlsx")
DB2_PATH = os.path.join("data", "Jumlah tempat pelayanan kb yang memiliki tenaga kesehatan dan administrasi.xlsx")

if not os.path.exists(DB1_PATH):
    st.error(f"File DB1 tidak ditemukan: {DB1_PATH}")
    st.stop()
if not os.path.exists(DB2_PATH):
    st.error(f"File DB2 tidak ditemukan: {DB2_PATH}")
    st.stop()

try:
    df1_all = load_db1_time_series(DB1_PATH)
    df2 = load_db2_people(DB2_PATH)
except Exception as e:
    st.error(f"Gagal memproses data: {e}")
    st.stop()

df1_kab = db1_agg_per_kab(df1_all)
df_int = df2.merge(df1_kab, on="KABUPATEN", how="inner")

kab_list = sorted(df_int["KABUPATEN"].unique().tolist())
if len(kab_list) == 0:
    st.error("Tidak ada kabupaten yang berhasil terhubung (cek penulisan kolom KABUPATEN di kedua file).")
    st.stop()

# =========================
# ICON NAV (LEFT)
# =========================
MENU_KEYS = ["SUMMARY", "TS", "PEOPLE", "LINK", "KRUSKAL", "DATASET"]
MENU_ICON = {
    "SUMMARY": "🏠",
    "TS": "📈",
    "PEOPLE": "👥",
    "LINK": "🔗",
    "KRUSKAL": "🧪",
    "DATASET": "🗂️",
}
MENU_NAME = {
    "SUMMARY": "Summary",
    "TS": "Deret Waktu",
    "PEOPLE": "People Analytics",
    "LINK": "Keterkaitan",
    "KRUSKAL": "Kruskal–Wallis",
    "DATASET": "Dataset",
}

with st.sidebar:
    menu_key = st.radio(
        "Menu",
        MENU_KEYS,
        index=0,
        format_func=lambda k: MENU_ICON[k],
        label_visibility="collapsed"
    )

# =========================
# TOPBAR + FILTER (KANAN ATAS)
# =========================
top_l, top_r = st.columns([2.7, 1.3], vertical_alignment="center")
with top_l:
    st.markdown(
        f"""
        <div class="topbar">
          <b>Dashboard KB Terintegrasi</b>
          <span class="crumb">{MENU_NAME[menu_key]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with top_r:
    kab = st.selectbox("Kabupaten", kab_list, index=0, label_visibility="collapsed")

st.write("")

# =========================
# KPI ROW
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Kabupaten terhubung", f"{df_int['KABUPATEN'].nunique():,}")
k2.metric("Total Tempat KB", f"{int(df_int['tempat_kb'].fillna(0).sum()):,}")
k3.metric("Total Tenaga Kesehatan", f"{int(df_int['tenaga_kesehatan_total'].fillna(0).sum()):,}")
k4.metric("Total Stok (setahun)", f"{float(df_int['TOTAL_STOK'].fillna(0).sum()):,.0f}")
st.write("")

# =========================
# CONTENT
# =========================
if menu_key == "SUMMARY":
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='card'><b>Top 10 — Tenaga Kesehatan Total</b></div>", unsafe_allow_html=True)
        top10_sdm = df_int.sort_values("tenaga_kesehatan_total", ascending=False).head(10)
        st.dataframe(top10_sdm[["KABUPATEN","tempat_kb","tenaga_kesehatan_total","administrasi"]], use_container_width=True)

        st.markdown("<div class='card'><b>Grafik Top 10 — Tenaga Kesehatan</b></div>", unsafe_allow_html=True)
        chart_df = top10_sdm.set_index("KABUPATEN")[["tenaga_kesehatan_total"]]
        st.bar_chart(chart_df, use_container_width=True)

    with right:
        st.markdown("<div class='card'><b>Top 10 — Total Stok Setahun</b></div>", unsafe_allow_html=True)
        top10_stok = df_int.sort_values("TOTAL_STOK", ascending=False).head(10)
        st.dataframe(top10_stok[["KABUPATEN","TOTAL_STOK","SUNTIK","PIL","IMPLAN","KONDOM","IUD"]], use_container_width=True)

        st.markdown("<div class='card'><b>Grafik Top 10 — Total Stok</b></div>", unsafe_allow_html=True)
        chart_df2 = top10_stok.set_index("KABUPATEN")[["TOTAL_STOK"]]
        st.bar_chart(chart_df2, use_container_width=True)

elif menu_key == "TS":
    st.markdown(f"<div class='card'><b>Deret Waktu Persediaan</b><br/><span style='color:rgba(15,23,42,0.62)'>{kab}</span></div>", unsafe_allow_html=True)

    df_ts = db1_ts_per_kab(df1_all, kab)

    vsel = st.multiselect("Variabel stok", VAR_STOK, default=VAR_STOK)
    if vsel:
        st.line_chart(df_ts.set_index("BULAN")[vsel], use_container_width=True)
    else:
        st.info("Pilih minimal 1 variabel.")

    # Moving average tambahan
    st.markdown("<div class='card'><b>Moving Average (3 bulan)</b></div>", unsafe_allow_html=True)
    ma_df = df_ts.copy()
    for v in VAR_STOK:
        ma_df[f"MA3_{v}"] = ma_df[v].rolling(3).mean()
    show_ma = [f"MA3_{v}" for v in vsel] if vsel else [f"MA3_{v}" for v in VAR_STOK]
    st.line_chart(ma_df.set_index("BULAN")[show_ma], use_container_width=True)

    # ADF
    st.markdown("<div class='card'><b>Uji stasioneritas (ADF)</b></div>", unsafe_allow_html=True)
    rows = []
    for v in VAR_STOK:
        p, status = adf_status(df_ts[v])
        rows.append({"Variabel": v, "p-value": p, "Kesimpulan": status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif menu_key == "PEOPLE":
    st.markdown(f"<div class='card'><b>People Analytics</b><br/><span style='color:rgba(15,23,42,0.62)'>{kab}</span></div>", unsafe_allow_html=True)

    row = df_int[df_int["KABUPATEN"] == kab].iloc[0]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Tempat KB", f"{int(row['tempat_kb']) if pd.notna(row['tempat_kb']) else 0:,}")
    a2.metric("Tenaga Kesehatan Total", f"{int(row['tenaga_kesehatan_total']) if pd.notna(row['tenaga_kesehatan_total']) else 0:,}")
    a3.metric("Administrasi", f"{int(row['administrasi']) if pd.notna(row['administrasi']) else 0:,}")
    a4.metric("Total Stok Setahun", f"{float(row['TOTAL_STOK']):,.0f}")

    st.markdown("<div class='card'><b>Komposisi stok setahun</b></div>", unsafe_allow_html=True)
    comp = pd.DataFrame({"Metode": VAR_STOK, "Total Setahun": [float(row[v]) for v in VAR_STOK]})
    st.dataframe(comp, use_container_width=True)

    st.markdown("<div class='card'><b>Grafik komposisi stok (Bar)</b></div>", unsafe_allow_html=True)
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

    cx1, cx2 = st.columns(2)
    with cx1:
        x_var = st.selectbox("Variabel People (X)", x_opts, index=1)
    with cx2:
        y_var = st.selectbox("Variabel Stok (Y)", y_opts, index=0)

    d = df_int[[x_var, y_var, "KABUPATEN"]].dropna()
    if len(d) < 5:
        st.warning("Data valid terlalu sedikit untuk analisis.")
    else:
        rho, p = spearmanr(d[x_var], d[y_var], nan_policy="omit")
        strength = spearman_strength(rho)

        m1, m2, m3 = st.columns(3)
        m1.metric("Spearman rho", f"{rho:.3f}")
        m2.metric("p-value", f"{p:.4f}")
        m3.metric("Kekuatan", strength)

        # Scatter
        st.markdown("<div class='card'><b>Scatter</b></div>", unsafe_allow_html=True)
        st.scatter_chart(d.set_index("KABUPATEN")[[x_var, y_var]], use_container_width=True)

    # Heatmap-like table (Spearman matrix) sederhana
    st.markdown("<div class='card'><b>Matriks korelasi Spearman (People vs Stok)</b></div>", unsafe_allow_html=True)
    people_vars = ["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat"]
    stok_vars = ["TOTAL_STOK"] + VAR_STOK
    corr_rows = []
    for px in people_vars:
        for sy in stok_vars:
            dd = df_int[[px, sy]].dropna()
            if len(dd) >= 5:
                r, pv = spearmanr(dd[px], dd[sy], nan_policy="omit")
            else:
                r, pv = np.nan, np.nan
            corr_rows.append({"People": px, "Stok": sy, "rho": r, "p_value": pv})
    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(corr_df, use_container_width=True)

elif menu_key == "KRUSKAL":
    st.markdown("<div class='card'><b>Uji Kruskal–Wallis</b><br/><span style='color:rgba(15,23,42,0.62)'>Perbedaan stok antar grup Administrasi</span></div>", unsafe_allow_html=True)

    y_var = st.selectbox("Variabel stok untuk diuji", VAR_ALL_STOK, index=0)

    # Buat grup berdasarkan jumlah administrasi
    # 0 | 1-2 | 3-5 | >5  (boleh kamu ubah)
    d = df_int[[y_var, "administrasi"]].dropna()
    d["administrasi"] = d["administrasi"].astype(float)
    d[y_var] = d[y_var].astype(float)

    def admin_group(a):
        if a == 0: return "0"
        if a <= 2: return "1-2"
        if a <= 5: return "3-5"
        return ">5"

    d["admin_group"] = d["administrasi"].apply(admin_group)

    groups = []
    labels = []
    for g in ["0", "1-2", "3-5", ">5"]:
        vals = d[d["admin_group"] == g][y_var].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(g)

    if len(groups) < 2:
        st.warning("Grup tidak cukup untuk Kruskal (minimal 2 grup dengan data).")
    else:
        stat, p_kw = kruskal(*groups)
        c1, c2 = st.columns(2)
        c1.metric("Kruskal H", f"{stat:.3f}")
        c2.metric("p-value", f"{p_kw:.4f}")

        if p_kw < 0.05:
            st.success("Ada perbedaan signifikan antar grup (α=5%).")
        else:
            st.info("Tidak ada perbedaan signifikan antar grup (α=5%).")

        st.markdown("<div class='card'><b>Ringkasan per grup</b></div>", unsafe_allow_html=True)
        summary = (
            d.groupby("admin_group")[y_var]
            .agg(["count", "median", "mean"])
            .reindex(labels)
            .reset_index()
            .rename(columns={"admin_group": "Grup Admin"})
        )
        st.dataframe(summary, use_container_width=True)

        # Box plot sederhana via st.bar_chart median per grup
        st.markdown("<div class='card'><b>Grafik Median per Grup (aproksimasi boxplot)</b></div>", unsafe_allow_html=True)
        med = d.groupby("admin_group")[y_var].median().reindex(labels)
        st.bar_chart(med, use_container_width=True)

elif menu_key == "DATASET":
    st.markdown("<div class='card'><b>Dataset Terintegrasi (hasil join)</b></div>", unsafe_allow_html=True)
    st.dataframe(df_int, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download dataset terintegrasi (CSV)",
        data=df_int.to_csv(index=False).encode("utf-8"),
        file_name="dataset_terintegrasi_kb.csv",
        mime="text/csv"
    )
