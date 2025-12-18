import os
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


# =========================
# PAGE CONFIG + THEME CSS
# =========================
st.set_page_config(page_title="Dashboard KB Terintegrasi", layout="wide")

DASHBOARD_CSS = """
<style>
/* Background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(99,102,241,0.35), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(59,130,246,0.25), transparent 55%),
              linear-gradient(180deg, #0b1020 0%, #0b1020 100%);
  color: #e5e7eb;
}

/* Main container padding */
section.main .block-container{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(17,24,39,0.95) 0%, rgba(17,24,39,0.85) 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] *{
  color: #e5e7eb !important;
}

/* Remove Streamlit default header space */
header {visibility: hidden;}
/* Card look */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card h3, .card h4, .card p {margin: 0;}
.muted{color: rgba(229,231,235,0.70); font-size: 0.9rem;}
.kpi-grid{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.kpi{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
}
.kpi .label{font-size: 0.78rem; color: rgba(229,231,235,0.75);}
.kpi .value{font-size: 1.25rem; font-weight: 700; margin-top: 6px;}
.kpi .delta{font-size: 0.78rem; color: rgba(229,231,235,0.75); margin-top: 6px;}
.badge{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
  font-size: 0.78rem;
  color: rgba(229,231,235,0.85);
}
hr{
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin: 14px 0;
}
.small-table-note{
  font-size: 0.8rem;
  color: rgba(229,231,235,0.7);
}

/* Make charts and tables sit nicer */
[data-testid="stDataFrame"]{
  border-radius: 12px;
  overflow: hidden;
}
</style>
"""
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


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


def norm_str(x):
    return str(x).strip().upper()


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def fmt_float(x, d=0):
    try:
        return f"{float(x):,.{d}f}"
    except Exception:
        return f"{0:,.{d}f}"


# =========================
# LOAD & PREP DB1 (Time series)
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

    df_all[KOLUM_NUMERIK_DB1] = df_all[KOLUM_NUMERIK_DB1].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

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
    out = (
        df_all[["KABUPATEN"] + VAR_STOK]
        .groupby("KABUPATEN", as_index=False)
        .sum()
    )
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


def kpi_cards(items):
    # items: list of dict(label, value, delta)
    cards_html = '<div class="kpi-grid">'
    for it in items:
        cards_html += f"""
        <div class="kpi">
          <div class="label">{it.get("label","")}</div>
          <div class="value">{it.get("value","")}</div>
          <div class="delta">{it.get("delta","")}</div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# =========================
# UI HEADER
# =========================
st.markdown(
    """
    <div class="card">
      <h3>Dashboard KB Terintegrasi</h3>
      <p class="muted">Persediaan kontrasepsi (deret waktu) × SDM & administrasi (people analytics) per kabupaten</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD DATA FROM REPO
# =========================
DB1_PATH = os.path.join("data", "DATA KETERSEDIAAN ALAT DAN OBAT KONTRASEPSI.xlsx")
DB2_PATH = os.path.join("data", "Jumlah tempat pelayanan kb yang memiliki tenaga kesehatan dan administrasi.xlsx")

with st.sidebar:
    st.markdown("### Navigasi")
    menu = st.radio(
        "Pilih menu",
        ["Summary", "Deret Waktu", "People Analytics", "Keterkaitan", "Dataset"],
        label_visibility="collapsed"
    )

    st.markdown("### Data Source")
    st.markdown(f"<span class='badge'>DB1: Excel</span>", unsafe_allow_html=True)
    st.caption(DB1_PATH)
    st.markdown(f"<span class='badge'>DB2: Excel</span>", unsafe_allow_html=True)
    st.caption(DB2_PATH)

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

missing_link = set(df2["KABUPATEN"]) - set(df1_kab["KABUPATEN"])
if len(missing_link) > 0:
    st.warning(f"Ada {len(missing_link)} kabupaten di DB2 yang tidak ketemu di DB1 (beda penulisan).")

kab_list = sorted(df_int["KABUPATEN"].unique().tolist())

with st.sidebar:
    st.markdown("### Filter")
    kab = st.selectbox("Kabupaten", kab_list, index=0)
    st.markdown("---")
    st.caption("Tip: jika join tidak lengkap, samakan penulisan nama kabupaten di kedua file.")

# =========================
# GLOBAL KPIs
# =========================
kpi_cards([
    {"label": "Kabupaten Terhubung", "value": f"{df_int['KABUPATEN'].nunique():,}", "delta": "hasil join DB1 × DB2"},
    {"label": "Total Tempat KB", "value": fmt_int(df_int["tempat_kb"].fillna(0).sum()), "delta": "akumulasi seluruh kabupaten"},
    {"label": "Total Tenaga Kesehatan", "value": fmt_int(df_int["tenaga_kesehatan_total"].fillna(0).sum()), "delta": "dokter + bidan + perawat"},
    {"label": "Total Stok (Setahun)", "value": fmt_float(df_int["TOTAL_STOK"].fillna(0).sum(), 0), "delta": "Suntik+Pil+Implan+Kondom+IUD"},
])

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# LAYOUT: MAIN + RIGHT SUMMARY PANEL
# =========================
main_col, right_col = st.columns([2.2, 1], gap="large")

# Right summary panel (always visible)
with right_col:
    row = df_int[df_int["KABUPATEN"] == kab].iloc[0]

    st.markdown(
        f"""
        <div class="card">
          <h4>Summary</h4>
          <p class="muted">Kabupaten terpilih</p>
          <hr/>
          <p><b>{kab}</b></p>
          <p class="muted">Tempat KB: <b>{fmt_int(row['tempat_kb'])}</b></p>
          <p class="muted">Tenaga Kesehatan: <b>{fmt_int(row['tenaga_kesehatan_total'])}</b></p>
          <p class="muted">Administrasi: <b>{fmt_int(row['administrasi'])}</b></p>
          <hr/>
          <p class="muted">SDM/Tempat: <b>{fmt_float(row['sdm_per_tempat'], 3)}</b></p>
          <p class="muted">Admin/Tempat: <b>{fmt_float(row['admin_per_tempat'], 3)}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top categories = top stok metode di kabupaten terpilih
    stok_comp = pd.DataFrame(
        {"Metode": VAR_STOK, "Total": [float(row[v]) for v in VAR_STOK]}
    ).sort_values("Total", ascending=False)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>Top Metode (Setahun)</h4><p class='muted'>Komposisi stok</p></div>", unsafe_allow_html=True)
    st.bar_chart(stok_comp.set_index("Metode")["Total"], use_container_width=True)

# =========================
# CONTENT PER MENU
# =========================
with main_col:
    if menu == "Summary":
        st.markdown(
            """
            <div class="card">
              <h4>Ringkasan Dashboard</h4>
              <p class="muted">Pilih kabupaten di sidebar untuk melihat detail. Gunakan menu untuk berpindah analisis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Mini leaderboard
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='card'><h4>Top 10 — Total Stok Setahun</h4><p class='muted'>Lintas kabupaten</p></div>", unsafe_allow_html=True)
            top_stok = df_int.sort_values("TOTAL_STOK", ascending=False).head(10)
            st.dataframe(top_stok[["KABUPATEN","TOTAL_STOK","SUNTIK","PIL","IMPLAN","KONDOM","IUD"]], use_container_width=True, height=340)
        with c2:
            st.markdown("<div class='card'><h4>Top 10 — Tenaga Kesehatan</h4><p class='muted'>Lintas kabupaten</p></div>", unsafe_allow_html=True)
            top_sdm = df_int.sort_values("tenaga_kesehatan_total", ascending=False).head(10)
            st.dataframe(top_sdm[["KABUPATEN","tempat_kb","tenaga_kesehatan_total","administrasi"]], use_container_width=True, height=340)

    elif menu == "Deret Waktu":
        st.markdown(
            f"""
            <div class="card">
              <h4>Deret Waktu Persediaan</h4>
              <p class="muted">Grafik bulanan untuk <b>{kab}</b>. Pilih variabel di bawah.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        df_ts = db1_ts_per_kab(df1_all, kab)
        vsel = st.multiselect("Variabel stok", VAR_STOK, default=VAR_STOK)

        st.markdown("<div class='card'><h4>Time Series</h4><p class='muted'>Bulan (Jan–Des)</p></div>", unsafe_allow_html=True)
        if vsel:
            st.line_chart(df_ts.set_index("BULAN")[vsel], use_container_width=True)
        else:
            st.info("Pilih minimal 1 variabel.")

        st.markdown("<div class='card'><h4>ADF Test</h4><p class='muted'>Uji stasioneritas per variabel</p></div>", unsafe_allow_html=True)
        rows = []
        for v in VAR_STOK:
            p, status = adf_status(df_ts[v])
            rows.append({"Variabel": v, "p-value": p, "Kesimpulan": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif menu == "People Analytics":
        st.markdown(
            """
            <div class="card">
              <h4>People Analytics</h4>
              <p class="muted">Ringkasan distribusi SDM, administrasi, dan kapasitas tempat KB.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='card'><h4>Deskriptif Variabel Kunci</h4><p class='muted'>Ringkasan statistik</p></div>", unsafe_allow_html=True)
        st.dataframe(
            df_int[["tempat_kb","tenaga_kesehatan_total","administrasi","sdm_per_tempat","admin_per_tempat","TOTAL_STOK"]].describe(),
            use_container_width=True
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='card'><h4>Distribusi: Tenaga Kesehatan Total</h4><p class='muted'>Bar chart top 15</p></div>", unsafe_allow_html=True)
            top15 = df_int.sort_values("tenaga_kesehatan_total", ascending=False).head(15)
            st.bar_chart(top15.set_index("KABUPATEN")["tenaga_kesehatan_total"], use_container_width=True)

        with c2:
            st.markdown("<div class='card'><h4>Distribusi: Total Stok Setahun</h4><p class='muted'>Bar chart top 15</p></div>", unsafe_allow_html=True)
            top15s = df_int.sort_values("TOTAL_STOK", ascending=False).head(15)
            st.bar_chart(top15s.set_index("KABUPATEN")["TOTAL_STOK"], use_container_width=True)

    elif menu == "Keterkaitan":
        st.markdown(
            """
            <div class="card">
              <h4>Analisis Keterkaitan</h4>
              <p class="muted">Korelasi Spearman & uji beda (Mann–Whitney) untuk menghubungkan SDM/admin dengan stok.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        x_opts = ["tempat_kb", "tenaga_kesehatan_total", "administrasi", "sdm_per_tempat", "admin_per_tempat"]
        y_opts = ["TOTAL_STOK"] + VAR_STOK

        cx1, cx2 = st.columns(2)
        with cx1:
            x_var = st.selectbox("Variabel People (X)", x_opts, index=1)
        with cx2:
            y_var = st.selectbox("Variabel Stok (Y)", y_opts, index=0)

        d = df_int[[x_var, y_var, "KABUPATEN", "administrasi"]].dropna()

        st.markdown("<div class='card'><h4>Spearman Correlation</h4><p class='muted'>Lintas kabupaten</p></div>", unsafe_allow_html=True)
        if len(d) < 5:
            st.warning("Data valid terlalu sedikit untuk analisis korelasi.")
        else:
            rho, p = spearmanr(d[x_var], d[y_var], nan_policy="omit")
            strength = spearman_strength(rho)

            kpi_cards([
                {"label": "Spearman rho", "value": f"{rho:.3f}", "delta": "arah & kekuatan hubungan"},
                {"label": "p-value", "value": f"{p:.4f}", "delta": "signifikan jika < 0.05"},
                {"label": "Kekuatan", "value": strength, "delta": "kategori interpretasi"},
                {"label": "N (kabupaten)", "value": f"{len(d):,}", "delta": "data non-null"},
            ])

            st.scatter_chart(d.set_index("KABUPATEN")[[x_var, y_var]])

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><h4>Mann–Whitney U Test</h4><p class='muted'>Bandingkan nilai Y: Admin ada vs Admin tidak</p></div>", unsafe_allow_html=True)

        d2 = df_int[[y_var, "administrasi"]].dropna()
        g_admin = d2[d2["administrasi"].astype(float) > 0][y_var].astype(float)
        g_no = d2[d2["administrasi"].astype(float) == 0][y_var].astype(float)

        if len(g_admin) == 0 or len(g_no) == 0:
            st.warning("Tidak cukup data untuk membagi grup admin > 0 vs admin = 0.")
        else:
            u, p_mw = mannwhitneyu(g_admin, g_no, alternative="two-sided")
            kpi_cards([
                {"label": "U statistic", "value": f"{u:.3f}", "delta": ""},
                {"label": "p-value", "value": f"{p_mw:.4f}", "delta": "signifikan jika < 0.05"},
                {"label": "Median (Admin ada)", "value": f"{float(np.median(g_admin)):.3f}", "delta": ""},
                {"label": "Median (Admin tidak)", "value": f"{float(np.median(g_no)):.3f}", "delta": ""},
            ])

            if p_mw < 0.05:
                st.success("Ada perbedaan signifikan antara kedua grup (α=5%).")
            else:
                st.info("Tidak ada perbedaan signifikan (α=5%).")

    elif menu == "Dataset":
        st.markdown(
            """
            <div class="card">
              <h4>Dataset Terintegrasi</h4>
              <p class="muted">Hasil join DB1 (stok tahunan) × DB2 (people analytics).</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(df_int, use_container_width=True, height=520)
        st.markdown("<p class='small-table-note'>Tip: gunakan fitur pencarian/scroll pada tabel untuk eksplor.</p>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download dataset terintegrasi (CSV)",
            data=df_int.to_csv(index=False).encode("utf-8"),
            file_name="dataset_terintegrasi_kb.csv",
            mime="text/csv"
        )
