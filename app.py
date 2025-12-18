import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import spearmanr, mannwhitneyu

# optional (ADF)
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

st.set_page_config(page_title="Dashboard KB Terintegrasi", layout="wide")

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

def norm_str(x):
    return str(x).strip().upper()

# =========================
# LOAD & PREP DB1 (Time series)
# =========================
@st.cache_data(ttl=300)
def load_db1_time_series(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    all_data = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        df["BULAN"] = norm_str(sheet)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # ensure columns exist
    miss = [c for c in KOLUM_NUMERIK_DB1 if c not in df_all.columns]
    if miss:
        raise ValueError(f"DB1: kolom numerik tidak ditemukan: {miss}")

    # numeric conversion
    df_all[KOLUM_NUMERIK_DB1] = df_all[KOLUM_NUMERIK_DB1].apply(pd.to_numeric, errors="coerce").fillna(0)

    # derived
    df_all["SUNTIK"] = (
        df_all["SUNTIKAN 1 BULANAN"]
        + df_all["SUNTIKAN 3 BULANAN KOMBINASI"]
        + df_all["SUNTIKAN 3 BULANAN PROGESTIN"]
    )
    df_all["PIL"] = df_all["PIL KOMBINASI"] + df_all["PIL PROGESTIN"]
    df_all["IMPLAN"] = df_all["IMPLAN 1 BATANG"] + df_all["IMPLAN 2 BATANG"]

    # month ordering
    df_all["BULAN"] = pd.Categorical(df_all["BULAN"].apply(norm_str), categories=URUTAN_BULAN, ordered=True)

    # kabupaten normalization (required for linking)
    if "KABUPATEN" in df_all.columns:
        df_all["KABUPATEN"] = df_all["KABUPATEN"].astype(str).str.strip().str.upper()
    else:
        raise ValueError("DB1: kolom 'KABUPATEN' tidak ditemukan (dibutuhkan untuk keterkaitan).")

    return df_all

def db1_ts_per_kab(df_all: pd.DataFrame, kabupaten: str) -> pd.DataFrame:
    df_k = df_all[df_all["KABUPATEN"] == kabupaten][["BULAN"] + VAR_STOK].copy()
    out = df_k.groupby("BULAN", as_index=False).sum().sort_values("BULAN")
    return out

def db1_agg_per_kab(df_all: pd.DataFrame) -> pd.DataFrame:
    # total setahun per kabupaten (untuk join dengan DB2)
    out = (
        df_all[["KABUPATEN"] + VAR_STOK]
        .groupby("KABUPATEN", as_index=False)
        .sum()
    )
    # tambahan: total semua metode
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
# LOAD & PREP DB2 (People analytics)
# =========================
@st.cache_data(ttl=300)
def load_db2_people(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    # rename persis seperti kode kamu
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

    # rounding ints
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
# UI
# =========================
st.title("📊 Dashboard KB Terintegrasi (Stok × SDM × Administrasi)")

with st.sidebar:
    st.header("Upload Data")
    f_db1 = st.file_uploader("DB1 — Persediaan Kontrasepsi (Excel multi-sheet Jan–Des)", type=["xlsx", "xls"], key="db1")
    f_db2 = st.file_uploader("DB2 — Tempat KB & Tenaga Kesehatan (Excel)", type=["xlsx", "xls"], key="db2")

if not f_db1 or not f_db2:
    st.info("Upload **kedua file** (DB1 dan DB2) agar dashboard bisa saling berketerkaitan.")
    st.stop()

# Load
try:
    df1_all = load_db1_time_series(f_db1)
    df2 = load_db2_people(f_db2)
except Exception as e:
    st.error(f"Gagal memproses data: {e}")
    st.stop()

# Build integrated dataset
df1_kab = db1_agg_per_kab(df1_all)  # stok tahunan per kab
df_int = df2.merge(df1_kab, on="KABUPATEN", how="inner")

missing_link = set(df2["KABUPATEN"]) - set(df1_kab["KABUPATEN"])
if len(missing_link) > 0:
    st.warning(f"Ada {len(missing_link)} kabupaten di DB2 yang tidak ketemu di DB1 (penulisan beda/typo).")

# Sidebar filters
with st.sidebar:
    st.header("Filter Analisis")
    kab_list = sorted(df_int["KABUPATEN"].unique().tolist())
    kab = st.selectbox("Pilih Kabupaten (untuk detail)", kab_list, index=0)

# =========================
# KPIs
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Kabupaten terhubung", f"{df_int['KABUPATEN'].nunique():,}")
col2.metric("Total Tempat KB", f"{int(df_int['tempat_kb'].fillna(0).sum()):,}")
col3.metric("Total Tenaga Kesehatan", f"{int(df_int['tenaga_kesehatan_total'].fillna(0).sum()):,}")
col4.metric("Total Stok (setahun)", f"{float(df_int['TOTAL_STOK'].fillna(0).sum()):,.0f}")

st.divider()

tabA, tabB, tabC, tabD = st.tabs([
    "Detail Kabupaten (Terhubung)",
    "Deret Waktu Stok",
    "People Analytics",
    "Analisis Keterkaitan (Cross)"
])

# =========================
# Detail Kabupaten
# =========================
with tabA:
    st.subheader(f"Profil Terpadu — {kab}")

    row = df_int[df_int["KABUPATEN"] == kab].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tempat KB", f"{int(row['tempat_kb']) if pd.notna(row['tempat_kb']) else 0:,}")
    c2.metric("Tenaga Kesehatan Total", f"{int(row['tenaga_kesehatan_total']) if pd.notna(row['tenaga_kesehatan_total']) else 0:,}")
    c3.metric("Administrasi", f"{int(row['administrasi']) if pd.notna(row['administrasi']) else 0:,}")
    c4.metric("Total Stok Setahun", f"{float(row['TOTAL_STOK']):,.0f}")

    st.markdown("**Rasio**")
    r1, r2 = st.columns(2)
    r1.metric("SDM per Tempat", f"{float(row['sdm_per_tempat']) if pd.notna(row['sdm_per_tempat']) else 0:.3f}")
    r2.metric("Admin per Tempat", f"{float(row['admin_per_tempat']) if pd.notna(row['admin_per_tempat']) else 0:.3f}")

    st.markdown("**Komposisi stok setahun**")
    st.dataframe(
        pd.DataFrame(
            {"Metode": VAR_STOK, "Total Setahun": [float(row[v]) for v in VAR_STOK]}
        ),
        use_container_width=True
    )

# =========================
# Deret Waktu Stok (kabupaten)
# =========================
with tabB:
    st.subheader("Deret Waktu Persediaan (per bulan)")

    df_ts = db1_ts_per_kab(df1_all, kab)

    vsel = st.multiselect("Variabel stok", VAR_STOK, default=VAR_STOK)
    if vsel:
        st.line_chart(df_ts.set_index("BULAN")[vsel], use_container_width=True)

    st.markdown("**Uji stasioneritas (ADF) per variabel**")
    rows = []
    for v in VAR_STOK:
        p, status = adf_status(df_ts[v])
        rows.append({"Variabel": v, "p-value": p, "Kesimpulan": status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================
# People Analytics
# =========================
with tabC:
    st.subheader("People Analytics (ringkas + ranking)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Top 10 Kabupaten — Tenaga Kesehatan Total**")
        top10_sdm = df_int.sort_values("tenaga_kesehatan_total", ascending=False).head(10)
        st.dataframe(top10_sdm[["KABUPATEN","tempat_kb","tenaga_kesehatan_total","administrasi"]], use_container_width=True)

    with right:
        st.markdown("**Top 10 Kabupaten — Total Stok Setahun**")
        top10_stok = df_int.sort_values("TOTAL_STOK", ascending=False).head(10)
        st.dataframe(top10_stok[["KABUPATEN","TOTAL_STOK","SUNTIK","PIL","IMPLAN","KONDOM","IUD"]], use_container_width=True)

    st.markdown("**Deskriptif variabel kunci**")
    st.dataframe(
        df_int[["tempat_kb","tenaga_kesehatan_total","administrasi","sdm_per_tempat","admin_per_tempat","TOTAL_STOK"]].describe(),
        use_container_width=True
    )

# =========================
# Cross analysis (keterkaitan)
# =========================
with tabD:
    st.subheader("Analisis Keterkaitan: SDM/Admin ↔ Stok Kontrasepsi (lintas kabupaten)")

    x_opts = ["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat"]
    y_opts = ["TOTAL_STOK"] + VAR_STOK

    cx1, cx2 = st.columns(2)
    with cx1:
        x_var = st.selectbox("Variabel People (X)", x_opts, index=1)
    with cx2:
        y_var = st.selectbox("Variabel Stok (Y)", y_opts, index=0)

    d = df_int[[x_var, y_var, "KABUPATEN", "administrasi"]].dropna()

    if len(d) < 5:
        st.warning("Data valid terlalu sedikit untuk analisis korelasi.")
    else:
        rho, p = spearmanr(d[x_var], d[y_var], nan_policy="omit")
        strength = spearman_strength(rho)

        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman rho", f"{rho:.3f}")
        c2.metric("p-value", f"{p:.4f}")
        c3.metric("Kekuatan", strength)

        st.caption("Interpretasi umum: p < 0.05 → hubungan signifikan (α=5%).")
        st.scatter_chart(d.set_index("KABUPATEN")[[x_var, y_var]])

    st.divider()
    st.subheader("Uji beda (Mann–Whitney): Stok Y pada kabupaten Admin ada vs tidak ada")

    d2 = df_int[[y_var, "administrasi"]].dropna()
    g_admin = d2[d2["administrasi"].astype(float) > 0][y_var].astype(float)
    g_no = d2[d2["administrasi"].astype(float) == 0][y_var].astype(float)

    if len(g_admin) == 0 or len(g_no) == 0:
        st.warning("Tidak cukup data untuk membagi grup admin > 0 vs admin = 0.")
    else:
        u, p_mw = mannwhitneyu(g_admin, g_no, alternative="two-sided")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("U", f"{u:.3f}")
        c2.metric("p-value", f"{p_mw:.4f}")
        c3.metric("Median (Admin ada)", f"{float(np.median(g_admin)):.3f}")
        c4.metric("Median (Admin tidak)", f"{float(np.median(g_no)):.3f}")

        if p_mw < 0.05:
            st.success("Ada perbedaan signifikan antara kedua grup (α=5%).")
        else:
            st.warning("Tidak ada perbedaan signifikan (α=5%).")

st.divider()
st.subheader("Dataset Terintegrasi (hasil join)")
st.dataframe(df_int, use_container_width=True)

st.download_button(
    "⬇️ Download dataset terintegrasi (CSV)",
    data=df_int.to_csv(index=False).encode("utf-8"),
    file_name="dataset_terintegrasi_kb.csv",
    mime="text/csv"
)
