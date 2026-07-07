import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter, maximum_filter, label
from scipy.interpolate import griddata
import io

import kaleido
kaleido.get_chrome_sync()

st.set_page_config(layout="wide", page_title="TM2 Analiz Sistemi v2.0", page_icon="🔍")
if "clear_cache" not in st.session_state:
    st.cache_data.clear()
    st.session_state["clear_cache"] = True

# ─────────────────────────────────────────────────────────
# YARDIMCI FONKSIYONLAR
# ─────────────────────────────────────────────────────────
def detect_columns(df):
    """Sütun isimlerini otomatik tanı / normalize et."""
    cols = {c: c.strip().lower().replace(",", "") for c in df.columns}
    df = df.rename(columns=cols)
    mapping = {}
    for c in df.columns:
        c_base = c.split("(")[0].split("[")[0].strip().rstrip()
        if c_base in ("x", "y", "z", "value", "val", "deger", "değer"):
            mapping[c] = {"x": "x", "y": "y", "z": "z", "value": "value", "val": "value",
                          "deger": "value", "değer": "value"}[c_base]
        elif any(k in c for k in ("koord", "x_koord", "pos_x", "konum_x")):
            mapping[c] = "x"
        elif any(k in c for k in ("y_koord", "pos_y", "konum_y")):
            mapping[c] = "y"
        elif any(k in c for k in ("derinlik", "depth", "z_koord")):
            mapping[c] = "z"
        elif any(k in c for k in ("manyetik", "sinyal", "grad", "magnetics", "anomali")):
            mapping[c] = "value"
    if mapping:
        df = df.rename(columns=mapping)
    return df


def create_grid(x, y, values, method="cubic", resolution=200):
    """Izgara enterpolasyonu (cubic / linear / nearest)."""
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    safe_method = method if len(x) >= 4 else "linear"
    zi_grid = griddata((x, y), values, (xi_grid, yi_grid), method=safe_method)
    zi_grid = np.nan_to_num(zi_grid, nan=0.0)
    return xi, yi, zi_grid


def analytic_signal_amplitude(grid_x, grid_y, grid_z):
    """Gerçek Analitik Sinyal Genliği: sqrt((dF/dx)^2 + (dF/dy)^2)."""
    df_dx = np.gradient(grid_z, grid_x[1] - grid_x[0], axis=1)
    df_dy = np.gradient(grid_z, grid_y[1] - grid_y[0], axis=0)
    return np.sqrt(df_dx**2 + df_dy**2)


def detect_peaks_2d(grid_z, min_distance=5, threshold_rel=0.3):
    """2D grid üzerinde pozitif ve negatif maksimumları bul."""
    abs_grid = np.abs(grid_z)
    threshold = abs_grid.max() * threshold_rel
    local_max = maximum_filter(abs_grid, size=min_distance) == abs_grid
    above_thresh = abs_grid > threshold
    candidates = local_max & above_thresh
    labeled, n_features = label(candidates, structure=np.ones((3, 3)))
    peaks = []
    for i in range(1, n_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        vals = abs_grid[ys, xs]
        best = np.argmax(vals)
        peaks.append((xs[best], ys[best], grid_z[ys[best], xs[best]]))
    peaks.sort(key=lambda p: abs(p[2]), reverse=True)
    return peaks


def compute_confidence(signal, noise, min_conf=50.0, max_conf=99.5):
    """SNR tabanlı güven skoru."""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power > 0 and signal_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        return max(min_conf, min(max_conf, 65.0 + (snr * 1.8)))
    return 75.0


def export_chart(fig, filename, fmt="png", width=1200, height=600):
    """Plotly grafiğini PNG veya PDF olarak dışa aktar."""
    try:
        scale = 2 if fmt == "png" else 1
        return pio.to_image(fig, format=fmt, width=width, height=height, scale=scale)
    except Exception:
        import traceback
        st.error(f"{fmt.upper()} dönüştürme hatası: {traceback.format_exc()}")
        return None


def fft_filter(grid, dx, dy, filter_type="gaussian", sigma=1.0, cutoff=0.15, order=4, height=10.0,
               cutoff_low=0.3, cutoff_high=0.05):
    """2D FFT tabanlı filtreler: butterworth_lp, butterworth_hp, upward, vert_deriv, gaussian, butterworth_bp."""
    ny, nx = grid.shape
    pad = min(ny, nx) // 2
    gp = np.pad(grid, pad, mode="reflect")
    F = np.fft.fftshift(np.fft.fft2(gp))
    fny, fnx = gp.shape
    u = np.fft.fftshift(np.fft.fftfreq(fnx, d=dx)) * 2 * np.pi
    v = np.fft.fftshift(np.fft.fftfreq(fny, d=dy)) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2) + 1e-15

    H = np.ones_like(D)
    if filter_type == "gaussian":
        H = np.exp(-D**2 / (2 * (sigma or 1.0)**2))
    elif filter_type == "butterworth_lp":
        H = 1.0 / (1.0 + (D / (cutoff + 1e-15))**(2 * order))
    elif filter_type == "butterworth_hp":
        H = 1.0 / (1.0 + ((cutoff + 1e-15) / D)**(2 * order))
    elif filter_type == "butterworth_bp":
        H_lp = 1.0 / (1.0 + (D / (cutoff_low + 1e-15))**(2 * order))
        H_hp = 1.0 / (1.0 + ((cutoff_high + 1e-15) / D)**(2 * order))
        H = H_lp * H_hp
    elif filter_type == "upward":
        H = np.exp(-D * height)
    elif filter_type == "vert_deriv":
        H = D * 0.5

    F_filtered = F * H
    gp_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    return gp_filtered[pad:pad + ny, pad:pad + nx]


def estimate_depth_from_data(df, peaks_xy):
    """CSV'deki z (derinlik) sütunundan hedef derinliklerini al."""
    depths = []
    for px, py in peaks_xy:
        idx = ((df["x"] - px).abs() + (df["y"] - py).abs()).idxmin()
        if idx in df.index and "z" in df.columns:
            depths.append(float(df.loc[idx, "z"]))
        else:
            depths.append(None)
    return depths


def classify_target(val, mean_val, std_val):
    """Hedef sınıflandırma."""
    if val > (mean_val + 2.5 * std_val):
        if val > (mean_val + 4.5 * std_val):
            return "İnsan Yapısı / Duvar / Temel", "positive_high"
        return "Metal / Yoğun Ferromanyetik", "positive"
    elif val < (mean_val - 2.5 * std_val):
        if val < (mean_val - 4.5 * std_val):
            return "İnsan Yapısı / Odacık / Mahzen", "negative_high"
        return "Doğal Boşluk / Mezar / Gevşek Zemin", "negative"
    return "Belirsiz / Rutin Toprak", "none"


# ─────────────────────────────────────────────────────────
# BASLIK & DOSYA YUKLEME
# ─────────────────────────────────────────────────────────
st.title("🔍 TM2 Veri Analiz Platformu v2.0")
st.markdown("""
<div style='background:#1a1a2e;padding:12px 18px;border-radius:10px;border-left:5px solid #e94560;margin-bottom:20px'>
<strong>✔ Gerçek Analitik Sinyal | ✔ Çoklu Hedef Tespiti | ✔ Enterpolasyon | ✔ Profil Kesitleri | ✔ SNR Kalibrasyonu</strong>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Gradyometre CSV Dosyasını Seçin (x, y, z, value sütunları)", type=["csv", "txt"], key="csv_uploader")

if uploaded_file is None:
    st.info("📂 Analize başlamak için gradyometre CSV dosyasını yükleyin.")
    if "raw_data" in st.session_state:
        del st.session_state["raw_data"]
    st.stop()

# ─────────────────────────────────────────────────────────
# VERI YUKLEME & TEMIZLEME
# ─────────────────────────────────────────────────────────
with st.spinner("Veri işleniyor..."):
    upload_key = uploaded_file.name + str(uploaded_file.size)
    if "raw_data" in st.session_state and st.session_state.get("upload_key") == upload_key:
        raw_lines = st.session_state["raw_data"]
    else:
        try:
            raw_bytes = uploaded_file.read()
            raw_text = raw_bytes.decode("utf-8-sig", errors="ignore")
            raw_lines = raw_text.splitlines()
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}. Lütfen farklı bir dosya deneyin.")
            st.stop()
        if not raw_lines:
            st.error("Dosya boş veya okunamadı. Lütfen geçerli bir CSV dosyası yükleyin.")
            st.stop()
        st.session_state["raw_data"] = raw_lines
        st.session_state["upload_key"] = upload_key
    cleaned_lines = [l.strip()[:-1] if l.strip().endswith(",") else l.strip() for l in raw_lines if l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)), sep=",")
    df = detect_columns(df)
    req = {"x", "y", "value"}
    found = req & set(df.columns)
    if len(found) < 3:
        st.error(f"Gerekli sütunlar bulunamadı. Mevcut sütunlar: {list(df.columns)}. 'x','y','value' gerekli.")
        st.stop()
    cols_use = ["x", "y", "value"]
    if "z" in df.columns:
        cols_use.append("z")
    df = df[cols_use]
    for col in cols_use:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if "z" not in df.columns:
        df["z"] = 0

    st.success(f"✅ {len(df)} ölçüm noktası yüklendi. Izgara: {df['x'].nunique()}×{df['y'].nunique()}")

# ─────────────────────────────────────────────────────────
# YAN PANEL – FILTRE KONTROLLERI
# ─────────────────────────────────────────────────────────
val_arr = df["value"].values.astype(float)
st.sidebar.header("🎛️ Filtre & Görsel Ayarları")
auto_params = st.sidebar.checkbox("🎯 Otomatik Parametre", False,
                                  help="Filtre parametrelerini veri istatistiğine göre otomatik ayarlar")
grid_resolution = st.sidebar.slider("Grid Çözünürlüğü", 50, 500, 200, 10,
                                    help="Düşük → hızlı, yüksek → detaylı (200 önerilir)")
detrend_on = st.sidebar.checkbox("Detrend (Saha Eğimi Kaldır)", True)
filter_type = st.sidebar.selectbox("Filtre Türü", [
    "Gauss Alçak Geçiren",
    "Butterworth Alçak Geçiren",
    "Butterworth Yüksek Geçiren",
    "Bant Geçiren (Butterworth)",
    "Yukarı Devam (Upward)",
    "Düşey Türev (1. Derece)"
], index=0)
filter_descriptions = {
    "Gauss Alçak Geçiren": "Genel yumuşatma, yüksek frekanslı gürültüyü bastırır, büyük yapıları korur.",
    "Butterworth Alçak Geçiren": "Keskin kesimli alçak geçiren; gürültüyü temizler, geçiş bandını korur.",
    "Butterworth Yüksek Geçiren": "Bölgesel trendi/büyük yapıları kaldırır, küçük anomalileri belirginleştirir.",
    "Bant Geçiren (Butterworth)": "Belirli frekans aralığını geçirir; hem çok büyük yapıları hem gürültüyü bastırır.",
    "Yukarı Devam (Upward)": "Sığ gürültüyü bastırır, derin yapıları vurgular. Yükseklik arttıkça etki artar.",
    "Düşey Türev (1. Derece)": "Kenarları/dar anomalileri keskinleştirir, üst üste binmiş sinyalleri ayırır.",
}
st.sidebar.caption(filter_descriptions.get(filter_type, ""))
if filter_type == "Gauss Alçak Geçiren":
    def_sigma = max(0.3, min(3.0, float(np.std(val_arr) / 30))) if auto_params else 1.0
    sigma = st.sidebar.slider("Sigma (Pürüzsüzlük)", 0.3, 3.0, def_sigma, 0.1)
    filter_params = {"sigma": sigma}
elif filter_type == "Bant Geçiren (Butterworth)":
    ux_bp = df["x"].nunique()
    def_clow = max(0.1, min(0.8, 0.5 - ux_bp * 0.001)) if auto_params else 0.3
    def_chigh = max(0.01, min(0.3, 0.1 - ux_bp * 0.0005)) if auto_params else 0.05
    cutoff_low = st.sidebar.slider("Üst Kesim Frekansı (düşük geçiren)", 0.1, 0.8, def_clow, 0.01)
    cutoff_high = st.sidebar.slider("Alt Kesim Frekansı (yüksek geçiren)", 0.01, 0.3, def_chigh, 0.01)
    order = st.sidebar.slider("Filtre Derecesi", 1, 8, 4, 1)
    filter_params = {"cutoff_low": cutoff_low, "cutoff_high": cutoff_high, "order": order}
elif filter_type.startswith("Butterworth"):
    ux_bw = df["x"].nunique()
    def_cut = max(0.01, min(0.5, 0.25 - ux_bw * 0.0008)) if auto_params else 0.12
    cutoff = st.sidebar.slider("Kesim Frekansı", 0.01, 0.5, def_cut, 0.01)
    order = st.sidebar.slider("Filtre Derecesi", 1, 8, 4, 1)
    filter_params = {"cutoff": cutoff, "order": order}
elif filter_type == "Yukarı Devam (Upward)":
    z_rng = float(df["z"].max() - df["z"].min()) if "z" in df.columns and df["z"].nunique() > 1 else 100
    def_h = max(5, min(100, int(z_rng * 0.15))) if auto_params else 20
    height = st.sidebar.slider("Devam Yüksekliği (cm)", 5, 100, def_h, 5)
    filter_params = {"height": height}
else:
    filter_params = {}
colorscale = st.sidebar.selectbox("Renk Paleti", ["RdBu_r", "Jet", "Viridis", "Plasma", "Turbo", "Earth"], index=0)
peak_threshold = st.sidebar.slider("İkincil Hedef Duyarlılığı", 0.2, 0.99, 0.55, 0.01)
min_peak_distance = st.sidebar.slider("Min. Hedef Mesafesi (piksel)", 5, 20, 10, 1)
show_analytic = st.sidebar.checkbox("Analitik Sinyal Katmanı", True)
show_tilt = st.sidebar.checkbox("Tilt Angle Filtresi", False)
show_raw_compare = st.sidebar.checkbox("Ham-Filtreli Karşılaştırma", True)
show_3d_analytic = st.sidebar.checkbox("3D Analitik Sinyal", False)
st.sidebar.markdown("---")
st.sidebar.markdown("**📏 Derinlik Tahmini**")
show_depth = st.sidebar.checkbox("Derinlik Tahmini (Analitik Sinyal)", True)
st.sidebar.markdown("---")
st.sidebar.markdown("**🧹 Outlier Temizleme**")
remove_outliers = st.sidebar.checkbox("Outlier Temizle", True,
                                      help="IQR yöntemi ile aşırı uç değerleri temizler")
outlier_iqr = st.sidebar.slider("IQR Eşiği", 0.5, 5.0, 1.5, 0.1,
                                help="Düşük → agresif temizlik (çok nokta silinir), yüksek → koruyucu (az nokta silinir)")
outlier_status = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Profil Kesiti**")
profile_dir = st.sidebar.radio("Profil Yönü", ["X", "Y"], horizontal=True, index=0)
profile_min = float(min(df["x"].min(), df["y"].min()))
profile_max = float(max(df["x"].max(), df["y"].max()))
profile_val = st.sidebar.slider("Profil Değeri", profile_min, profile_max, 30.0, 1.0)

# ─────────────────────────────────────────────────────────
# SİNYAL İSLEME
# ─────────────────────────────────────────────────────────
x_arr = df["x"].values.astype(float)
y_arr = df["y"].values.astype(float)
ux, uy = df["x"].nunique(), df["y"].nunique()

# 0. OUTLIER TEMIZLEME (IQR)
if remove_outliers:
    q1, q3 = np.percentile(df["value"], 25), np.percentile(df["value"], 75)
    iqr_v = q3 - q1
    lower = q1 - outlier_iqr * iqr_v
    upper = q3 + outlier_iqr * iqr_v
    before = len(df)
    mask = (df["value"] >= lower) & (df["value"] <= upper)
    after = int(mask.sum())
    removed = before - after
    if after >= 10 and after < before:
        df = df[mask].reset_index(drop=True)
        outlier_status.success(f"🧹 {removed} outlier temizlendi ({before}→{after})")
        val_arr = df["value"].values.astype(float)
        x_arr = df["x"].values.astype(float)
        y_arr = df["y"].values.astype(float)
        ux, uy = df["x"].nunique(), df["y"].nunique()

# 1. DETREND
if detrend_on and ux > 1 and uy > 1:
    sx = np.polyfit(x_arr, val_arr, 1)[0]
    sy = np.polyfit(y_arr, val_arr, 1)[0]
    filtered_val = val_arr - (x_arr * sx + y_arr * sy)
else:
    filtered_val = val_arr.copy()

df["filtered_value"] = filtered_val

# 2. FILTRELEME (seçilen filtre türü ile, gridde işle, orijinal noktalara enterpole et)
is_2d = ux > 1 and uy > 1
xi = yi = zi_grid = zi_smoothed = as_grid = depth_grid = None

if is_2d:
    xi, yi, zi_grid = create_grid(df["x"], df["y"], df["filtered_value"], method="cubic", resolution=grid_resolution)
    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0
    # Seçilen filtreyi uygula
    ftype_map = {
        "Gauss Alçak Geçiren": "gaussian",
        "Butterworth Alçak Geçiren": "butterworth_lp",
        "Butterworth Yüksek Geçiren": "butterworth_hp",
        "Bant Geçiren (Butterworth)": "butterworth_bp",
        "Yukarı Devam (Upward)": "upward",
        "Düşey Türev (1. Derece)": "vert_deriv",
    }
    zi_smoothed = fft_filter(zi_grid, dx, dy, ftype_map[filter_type], **filter_params)
    as_grid = analytic_signal_amplitude(xi, yi, zi_smoothed)
    # Derinlik grid'i (varsa)
    if "z" in df.columns and df["z"].nunique() > 1:
        depth_grid = create_grid(df["x"], df["y"], df["z"], method="linear", resolution=grid_resolution)[2]
    # Smooth grid değerlerini orijinal (x,y) noktalarına enterpole et
    flat_x, flat_y = np.meshgrid(xi, yi)
    df["filtered_value"] = griddata(
        (flat_x.ravel(), flat_y.ravel()), zi_smoothed.ravel(),
        (df["x"].values, df["y"].values), method="linear"
    )
    # NaN kalmamalı; kalırsa nearest
    nan_mask = df["filtered_value"].isna()
    if nan_mask.any():
        df.loc[nan_mask, "filtered_value"] = griddata(
            (flat_x.ravel(), flat_y.ravel()), zi_smoothed.ravel(),
            (df.loc[nan_mask, "x"].values, df.loc[nan_mask, "y"].values), method="nearest"
        )
else:
    df = df.sort_values("y")
    gauss_sigma = filter_params.get("sigma", 1.0)
    df["filtered_value"] = gaussian_filter(df["filtered_value"].values, sigma=gauss_sigma)

# 3. TILT ANGLE FILTRESI
tilt_grid = None
if is_2d and show_tilt:
    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0
    gx = np.gradient(zi_smoothed, dx, axis=1)
    gy = np.gradient(zi_smoothed, dy, axis=0)
    thg = np.sqrt(gx**2 + gy**2)
    # Yukarı devam yaklaşımı ile düşey türev
    zi_up = gaussian_filter(zi_smoothed, sigma=2.0)
    vert_deriv = (zi_smoothed - zi_up) / (dy * 0.5 + 1e-10)
    tilt_grid = np.arctan2(vert_deriv, thg + 1e-10)

# 4. RAW GRID (karşılaştırma için)
raw_grid = None
if is_2d and show_raw_compare:
    raw_grid = create_grid(df["x"], df["y"], df["value"], method="linear", resolution=grid_resolution)[2]

# 5. ISTATISTIK
max_val = float(df["filtered_value"].max())
min_val = float(df["filtered_value"].min())
mean_val = float(df["filtered_value"].mean())
std_val = float(df["filtered_value"].std())
q25, q75 = df["filtered_value"].quantile(0.25), df["filtered_value"].quantile(0.75)
iqr = q75 - q25

# 6. ANA HEDEF TESPITI (orijinal veri noktalarında, mutlak sapması en büyük)
df["abs_dev"] = (df["filtered_value"] - mean_val).abs()
main_peak = df.loc[df["abs_dev"].idxmax()]
c_x = float(main_peak["x"])
c_y = float(main_peak["y"])
c_val = float(main_peak["filtered_value"])

# 7. IKINCIL HEDEFLER (grid üzerinde, sadece ek bilgi)
peaks = [(c_x, c_y, c_val)]
if is_2d:
    r_x = df["x"].max() - df["x"].min()
    r_y = df["y"].max() - df["y"].min()
    pad_x_grid = max(6, r_x * 0.05) if r_x > 0 else 6
    pad_y_grid = max(6, r_y * 0.05) if r_y > 0 else 6
    raw_peaks = detect_peaks_2d(zi_smoothed, min_distance=min_peak_distance, threshold_rel=peak_threshold)
    for px, py, pv in raw_peaks:
        real_x = float(xi[px])
        real_y = float(yi[py])
        real_v = float(pv)
        if abs(real_x - c_x) > pad_x_grid * 0.5 or abs(real_y - c_y) > pad_y_grid * 0.5:
            peaks.append((real_x, real_y, real_v))
    peaks = [peaks[0]] + sorted(peaks[1:], key=lambda p: abs(p[2]), reverse=True)[:3]
closest_idx = ((df["x"] - c_x).abs() + (df["y"] - c_y).abs()).idxmin()
highest_anomaly_z = int(df.loc[closest_idx, "z"]) if "z" in df.columns and closest_idx in df.index else 0

# 8. HEDEF DERiNLIKLERI (z sütunundan)
peak_depths = []
if "z" in df.columns and df["z"].nunique() > 1:
    peak_xy = [(p[0], p[1]) for p in peaks[:5]]
    peak_depths = estimate_depth_from_data(df, peak_xy)

# Kutu
r_x = df["x"].max() - df["x"].min()
r_y = df["y"].max() - df["y"].min()
pad_x = max(6, r_x * 0.05) if r_x > 0 else 6
pad_y = max(6, r_y * 0.05) if r_y > 0 else 6
min_anom_x = max(df["x"].min(), c_x - pad_x)
max_anom_x = min(df["x"].max(), c_x + pad_x)
min_anom_y = max(df["y"].min(), c_y - pad_y)
max_anom_y = min(df["y"].max(), c_y + pad_y)

# Güven skoru
confidence_score = compute_confidence(df["filtered_value"], df["value"] - df["filtered_value"])

# Hedef sınıflandırma
target_type, target_class = classify_target(c_val, mean_val, std_val)

# ─────────────────────────────────────────────────────────
# SEKMELI ARAYÜZ
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Manyetik Harita", "📐 3D Model", "📊 İstatistik", "🤖 Rapor", "📋 Veri"
])

# ──────── TAB 1: 2D HARITA ────────
with tab1:
    st.subheader("Manyetik Dağılım Haritası")
    col_meta = st.columns(5)
    col_meta[0].metric("Nokta", f"{len(df)}")
    col_meta[1].metric("Sinyal (min/max)", f"{min_val:.1f} / {max_val:.1f}")
    col_meta[2].metric("Ortalama ± Std", f"{mean_val:.1f} ± {std_val:.1f}")
    col_meta[3].metric("Güven Skoru", f"%{confidence_score:.1f}")
    col_meta[4].metric("Hedef Sayısı", f"{len(peaks)}")

    if is_2d:
        # Hangi ikinci panel gösterilecek?
        second_title = None
        second_z = None
        if show_raw_compare and raw_grid is not None:
            second_title = "Ham Veri (Filtresiz)"
            second_z = raw_grid
        elif show_analytic:
            second_title = "Analitik Sinyal (Toplam Gradyan)"
            second_z = as_grid
        elif show_tilt and tilt_grid is not None:
            second_title = "Tilt Angle"
            second_z = tilt_grid

        ncols = 2 if second_z is not None else 1
        sub_titles = ("Filtrelenmiş Sinyal", second_title) if second_z is not None else ("Filtrelenmiş Sinyal",)
        fig1 = make_subplots(rows=1, cols=ncols, subplot_titles=sub_titles, horizontal_spacing=0.12)

        fig1.add_trace(go.Contour(
            x=xi, y=yi, z=zi_smoothed, colorscale=colorscale,
            contours=dict(coloring="heatmap", showlabels=True),
            hovertemplate="<b>X:</b> %{x:.1f}<br><b>Y:</b> %{y:.1f}<br><b>Filtreli:</b> %{z:.2f}<extra></extra>",
            colorbar=dict(title="Filtrelenmiş", x=0.46 if second_z is not None else 1.02)
        ), row=1, col=1)

        target_x, target_y, target_hover, target_labels = [], [], [], []
        for pi, (px, py, pv) in enumerate(peaks):
            if pi > 4: break
            prx = max(4, r_x * 0.04)
            pry = max(4, r_y * 0.04)
            c = "Lime" if pi == 0 else "Cyan"
            fig1.add_shape(type="rect", x0=px - prx, y0=py - pry, x1=px + prx, y1=py + pry,
                           line=dict(color=c, width=2, dash="dash"), row=1, col=1)
            target_x.append(px); target_y.append(py)
            depth_h = ""
            if pi < len(peak_depths) and peak_depths[pi] is not None:
                depth_h = f"<b>Derinlik:</b> ~{peak_depths[pi]:.0f} cm<br>"
            ttype, _ = classify_target(pv, mean_val, std_val)
            target_hover.append(
                f"<b>HEDEF #{pi+1}</b><br>"
                f"<b>X:</b> {px:.1f} cm<br>"
                f"<b>Y:</b> {py:.1f} cm<br>"
                f"<b>Sinyal:</b> {pv:.2f}<br>"
                f"{depth_h}"
                f"<b>Sınıf:</b> {ttype}"
            )
            target_labels.append(f" H{pi+1}")
        if target_x:
            fig1.add_trace(go.Scatter(
                x=target_x, y=target_y, mode="markers+text",
                marker=dict(
                    color=["Lime" if i == 0 else "Cyan" for i in range(len(target_x))],
                    size=12, symbol="x", line=dict(color="white", width=1)
                ),
                text=target_labels, textposition="top center",
                hovertext=target_hover, hoverinfo="text",
                showlegend=False
            ), row=1, col=1)

        # Kazı alanı dikdörtgeni (ana hedef merkezli, min 100cm, harita sınırlarına kırpılmış)
        kazi_yarim = max((max_anom_x - min_anom_x) / 2, 50)
        kazi_x0 = max(df["x"].min(), c_x - kazi_yarim)
        kazi_x1 = min(df["x"].max(), c_x + kazi_yarim)
        kazi_y0 = max(df["y"].min(), c_y - kazi_yarim)
        kazi_y1 = min(df["y"].max(), c_y + kazi_yarim)
        fig1.add_shape(type="rect",
                       x0=kazi_x0, y0=kazi_y0, x1=kazi_x1, y1=kazi_y1,
                       line=dict(color="red", width=2, dash="dot"),
                       fillcolor="rgba(255,0,0,0.05)", row=1, col=1)
        fig1.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                       line=dict(color="red", width=2, dash="dot"),
                       name="Kazı Alanı", showlegend=True), row=1, col=1)

        if second_z is not None:
            cs = "RdBu_r" if second_z is tilt_grid else colorscale
            fig1.add_trace(go.Contour(
                x=xi, y=yi, z=second_z, colorscale=cs,
                contours=dict(coloring="heatmap", showlabels=True),
                hovertemplate="<b>X:</b> %{x:.1f}<br><b>Y:</b> %{y:.1f}<br><b>Değer:</b> %{z:.3f}<extra></extra>",
                colorbar=dict(title=second_title, x=1.02)
            ), row=1, col=2)

        fig1.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
        fig1.update_xaxes(title_text="X (cm)", row=1, col=1)
        fig1.update_yaxes(title_text="Y (cm)", row=1, col=1)
        if second_z is not None:
            fig1.update_xaxes(title_text="X (cm)", row=1, col=2)
            fig1.update_yaxes(title_text="Y (cm)", row=1, col=2)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["y"], y=-df["z"], mode="markers",
            marker=dict(size=8, color=df["filtered_value"], colorscale=colorscale, colorbar=dict(title="Sinyal")),
            hovertemplate="<b>Y:</b> %{x:.0f}<br><b>Z:</b> %{y:.0f}<br><b>Sinyal:</b> %{marker.color:.2f}<extra></extra>"))
        fig1.add_shape(type="rect", x0=df["y"].min(), y0=-df["z"].max(),
                       x1=df["y"].max(), y1=-df["z"].min(), line=dict(color="Red", width=2, dash="dashdot"))
        fig1.update_layout(height=500, xaxis_title="Y (cm)", yaxis_title="Derinlik -Z (cm)")
        st.plotly_chart(fig1, use_container_width=True)

    # Profil kesiti (X veya Y yönünde)
    if is_2d:
        if profile_dir == "X":
            uniq_vals = df["x"].unique()
            nearest = uniq_vals[np.abs(uniq_vals - profile_val).argmin()]
            prof_data = df[df["x"] == nearest]
            x_axis, x_label = "y", "Y (cm)"
        else:
            uniq_vals = df["y"].unique()
            nearest = uniq_vals[np.abs(uniq_vals - profile_val).argmin()]
            prof_data = df[df["y"] == nearest]
            x_axis, x_label = "x", "X (cm)"
        st.subheader(f"📈 {profile_dir} = {nearest:.0f} cm Profil Kesiti")
        if len(prof_data) > 2:
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(x=prof_data[x_axis], y=prof_data["filtered_value"],
                mode="lines+markers", name="Filtrelenmiş", line=dict(color="#e94560", width=2), marker=dict(size=5)))
            fig_prof.add_trace(go.Scatter(x=prof_data[x_axis], y=prof_data["value"],
                mode="lines", name="Ham", line=dict(color="gray", width=1, dash="dot")))
            fig_prof.add_hline(y=mean_val, line=dict(color="green", dash="dash"), annotation_text="Ortalama")
            fig_prof.add_hline(y=mean_val + 2 * std_val, line=dict(color="orange", dash="dash"), annotation_text="+2\u03c3")
            fig_prof.add_hline(y=mean_val - 2 * std_val, line=dict(color="orange", dash="dash"), annotation_text="-2\u03c3")
            fig_prof.update_layout(height=300, xaxis_title=x_label, yaxis_title="Sinyal", hovermode="x unified")
            st.plotly_chart(fig_prof, use_container_width=True)

    # Dışa aktar butonları
    exp_cols = st.columns(3)
    if is_2d:
        png_bytes = export_chart(fig1, "manyetik_harita", "png")
        if png_bytes:
            exp_cols[0].download_button("📷 Harita PNG", png_bytes,
                                          "manyetik_harita.png", "image/png")
        pdf_bytes = export_chart(fig1, "manyetik_harita", "pdf")
        if pdf_bytes:
            exp_cols[1].download_button("📄 Harita PDF", pdf_bytes, "manyetik_harita.pdf", "application/pdf")
        else:
            exp_cols[1].info("PDF dönüştürme başarısız")
        has_profile = len(prof_data) > 2
        if has_profile:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=prof_data[x_axis], y=prof_data["filtered_value"], mode="lines", name="Filtrelenmiş"))
            fig_p.update_layout(height=300)
            prof_png = export_chart(fig_p, "profil", "png")
            if prof_png:
                exp_cols[2].download_button("📈 Profil PNG", prof_png, "profil.png", "image/png")

# ──────── TAB 2: 3D MODEL ────────
with tab2:
    st.subheader("Pürüzsüzleştirilmiş 3D Manyetik Rölyef")
    if is_2d:
        zi_clean = np.nan_to_num(zi_smoothed, nan=0.0)
        fig3 = go.Figure()
        fig3.add_trace(go.Surface(
            x=xi, y=yi, z=zi_clean, colorscale=colorscale,
            colorbar=dict(title="Sinyal", x=1.05),
            customdata=depth_grid[..., None] if depth_grid is not None else None,
            hovertemplate=(
                "<b>X:</b> %{x:.1f}<br><b>Y:</b> %{y:.1f}<br><b>Sinyal:</b> %{z:.2f}"
                + ("<br><b>Derinlik:</b> %{customdata:.1f} cm<extra></extra>" if depth_grid is not None else "<extra></extra>")
            )
        ))
        if show_3d_analytic and as_grid is not None:
            as_clean = np.nan_to_num(as_grid, nan=0.0)
            fig3.add_trace(go.Surface(
                x=xi, y=yi, z=as_clean, colorscale="Hot",
                opacity=0.5, showscale=False
            ))
        # Hedef işaretleri
        for pi, (px, py, pv) in enumerate(peaks):
            if pi > 4:
                break
            pz = float(pv)
            color = "lime" if pi == 0 else "cyan"
            depth_tag = ""
            if pi < len(peak_depths) and peak_depths[pi] is not None:
                depth_tag = f" ~{peak_depths[pi]:.0f}cm"
            fig3.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz], mode="markers+text",
                marker=dict(size=8, color=color, symbol="diamond"),
                text=[f"H{pi+1}{depth_tag}"], textposition="top center",
                showlegend=False
            ))
        fig3.update_layout(
            scene=dict(
                xaxis_title="X (cm)", yaxis_title="Y (cm)", zaxis_title="Sinyal",
                xaxis=dict(backgroundcolor="rgb(230,230,230)", showbackground=True),
                yaxis=dict(backgroundcolor="rgb(230,230,230)", showbackground=True),
                zaxis=dict(backgroundcolor="rgb(210,210,210)", showbackground=True)
            ),
            height=580, margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        ds = df.sort_values("y")
        fig3 = go.Figure(data=go.Scatter3d(
            x=np.zeros(len(ds)), y=ds["y"], z=ds["filtered_value"],
            mode="lines+markers",
            line=dict(color="red", width=4),
            marker=dict(size=5, color=ds["filtered_value"], colorscale=colorscale)
        ))
        fig3.update_layout(height=580, scene=dict(xaxis_title="", yaxis_title="Y (cm)", zaxis_title="Sinyal"))
        st.plotly_chart(fig3, use_container_width=True)
        # 3D dışa aktar
        e3 = st.columns(2)
        e3[0].download_button("📷 3D PNG", export_chart(fig3, "3d_model", "png"), "3d_model.png", "image/png")
        try:
            pdf3 = pio.to_image(fig3, format="pdf", width=1200, height=600)
            e3[1].download_button("📄 3D PDF", pdf3, "3d_model.pdf", "application/pdf")
        except Exception:
            pass

# ──────── TAB 3: İSTATİSTİK ────────
with tab3:
    st.subheader("İstatistiksel Analiz")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ortalama", f"{mean_val:.3f}")
    c2.metric("Standart Sapma", f"{std_val:.3f}")
    c3.metric("Çarpıklık (Skewness)", f"{df['filtered_value'].skew():.3f}")
    c4.metric("Basıklık (Kurtosis)", f"{df['filtered_value'].kurtosis():.3f}")

    fig_stat = make_subplots(rows=1, cols=3, subplot_titles=("Histogram", "Q-Q Plot", "Kutu Grafiği"),
                             horizontal_spacing=0.12)
    fv = df["filtered_value"].values
    # Histogram
    fig_stat.add_trace(go.Histogram(x=fv, nbinsx=40, marker_color="#3a86ff",
                                     hovertemplate="Değer: %{x:.2f}<br>Frekans: %{y}<extra></extra>"),
                       row=1, col=1)
    # Q-Q plot
    sorted_fv = np.sort(fv)
    n = len(sorted_fv)
    theoretical = np.random.normal(mean_val, std_val, n) if n > 1 else sorted_fv
    theoretical.sort()
    fig_stat.add_trace(go.Scatter(x=theoretical, y=sorted_fv, mode="markers",
                                   marker=dict(color="#e94560", size=4),
                                   hovertemplate="Teorik: %{x:.2f}<br>Gözlem: %{y:.2f}<extra></extra>"),
                       row=1, col=2)
    max_qq = max(theoretical.max(), sorted_fv.max())
    min_qq = min(theoretical.min(), sorted_fv.min())
    fig_stat.add_trace(go.Scatter(x=[min_qq, max_qq], y=[min_qq, max_qq],
                                   mode="lines", line=dict(color="gray", dash="dash"),
                                   showlegend=False),
                       row=1, col=2)
    # Kutu grafiği
    fig_stat.add_trace(go.Box(y=fv, name="Sinyal", marker_color="#8338ec",
                               boxmean=True),
                       row=1, col=3)

    fig_stat.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_stat, use_container_width=True)

    # Anomali tablosu
    st.subheader("Çoklu Hedef Listesi")
    if is_2d and peaks:
        rows = []
        for pi, (px, py, pv) in enumerate(peaks):
            if pi > 9:
                break
            ttype, _ = classify_target(pv, mean_val, std_val)
            depth_str = ""
            if pi < len(peak_depths) and peak_depths[pi] is not None:
                depth_str = f"~{peak_depths[pi]:.0f} cm"
            else:
                depth_str = "—"
            rows.append({
                "Hedef #": pi + 1,
                "X (cm)": f"{px:.1f}",
                "Y (cm)": f"{py:.1f}",
                "Sinyal": f"{pv:.2f}",
                "Derinlik": depth_str,
                "Sınıflandırma": ttype
            })
        st.table(pd.DataFrame(rows))
    else:
        ttype, _ = classify_target(peaks[0][2], mean_val, std_val) if peaks else ("-", "-")
        rows = [{"Hedef #": 1, "X (cm)": f"{peaks[0][0]:.1f}" if peaks else "-",
                 "Y (cm)": f"{peaks[0][1]:.1f}" if peaks else "-",
                 "Sinyal": f"{peaks[0][2]:.2f}" if peaks else "-",
                 "Sınıflandırma": ttype}]
        st.table(pd.DataFrame(rows))

# ──────── TAB 4: AI RAPORU ────────
with tab4:
    st.subheader("Yapay Zeka Otomatik Analiz Raporu")

    s_std = ""
    if std_val >= 40:
        s_std = "Yüksek manyetik değişkenlik – metal/cüruf/yapı duvarı olasılığı yüksek."
    elif std_val >= 15:
        s_std = "Orta heterojen yapı – mineral kırılmaları veya dolgu."
    else:
        s_std = "Homojen stabil mineral yapısı – doğal zemin."

    hedef_listesi_str = "\n".join(
        [f"  • Hedef {i+1}: X={px:.1f}, Y={py:.1f}, Sinyal={pv:.2f}"
         + (f", Derinlik≈{peak_depths[i]:.0f}cm" if i < len(peak_depths) and peak_depths[i] is not None else "")
         for i, (px, py, pv) in enumerate(peaks[:5])]
    ) if peaks else "  • Belirgin hedef saptanmadı."

    # Çalışma uygunluğu değerlendirmesi
    if not peaks:
        calisma_uygunlugu = "Belirgin hedef saptanmadı. Detaylı çalışma önerilmez."
    elif confidence_score >= 80 and len(peaks) <= 3:
        calisma_uygunlugu = "Yüksek güven skoru ve az sayıda hedef. Sahada çalışma yapılabilir, hedeflere odaklanılmalı."
    elif confidence_score >= 60:
        calisma_uygunlugu = "Orta güven skoru. Hedefler doğrulama kazısı ile teyit edildikten sonra çalışma yapılabilir."
    else:
        calisma_uygunlugu = "Düşük güven skoru. Sahada çalışma yapılmadan önce ek veri toplanması önerilir."
    if len(peaks) > 3:
        calisma_uygunlugu += " Çok sayıda hedef tespit edildi, önceliklendirme yapılmalı."

    report_text = f"""==================================================
GRADYOMETRE ANALİZ RAPORU (v2.0)
==================================================

1. TOPRAK MINERALIZASYON DURUMU
{s_std}

2. ÇOKLU HEDEF ANALİZİ
Toplam tespit: {len(peaks)} hedef
{hedef_listesi_str}

3. ANA HEDEF KUTUSU
X: [{min_anom_x:.0f} - {max_anom_x:.0f}] cm
Y: [{min_anom_y:.0f} - {max_anom_y:.0f}] cm
Derinlik: {highest_anomaly_z} cm

4. SINIFLANDIRMA
{target_type}

5. GÜVEN SKORU
%{confidence_score:.1f} (SNR tabanlı)

6. İSTATISTIK ÖZET
Ortalama: {mean_val:.3f} | Std: {std_val:.3f}
Çarpıklık: {df['filtered_value'].skew():.3f} | Basıklık: {df['filtered_value'].kurtosis():.3f}
Çeyreklikler: Q1={q25:.3f}, Q3={q75:.3f}, IQR={iqr:.3f}

7. OPERASYONEL TAVSİYE
Ana hedefe odaklanın. İkincil hedefleri doğrulama kazısı ile test edin.

8. ÇALIŞMA UYGUNLUĞU
{calisma_uygunlugu}
==================================================
"""
    st.info(report_text.replace("\n", "  \n"))
    c_rap = st.columns(2)
    c_rap[0].download_button("📄 Raporu İndir (.txt)", report_text, "gradyometre_raporu.txt", "text/plain")
    c_rap[1].download_button("📊 Filtrelenmiş Veriyi İndir (.csv)",
                              df[["x", "y", "z", "value", "filtered_value"]].to_csv(index=False).encode("utf-8"),
                              "gradyometre_filtrelenmis.csv", "text/csv")

# ──────── TAB 5: VERİ ────────
with tab5:
    st.subheader("Ham & Filtrelenmiş Veri")
    show_df = df[["x", "y", "z", "value", "filtered_value"]].copy()
    show_df.columns = ["X (cm)", "Y (cm)", "Z (cm)", "Ham Sinyal", "Filtrelenmiş Sinyal"]
    st.dataframe(show_df, use_container_width=True, height=400)
    st.caption(f"Toplam {len(df)} satır, {len(show_df.columns)} sütun")

# ─────────────────────────────────────────────────────────
# HEDEF ÖZET TABLOSU (footer)
# ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Ana Hedef Kimlik Kartı")
summary_data = {
    "Parametre": [
        "Hedef Merkez X", "Hedef Merkez Y", "Odak Derinlik (Z)",
        "Filtrelenmiş Zirve Değer", "Güven Skoru (SNR)",
        "Sınıflandırma", "Hedef Sayısı", "Sinyal Aralığı"
    ],
    "Değer": [
        f"{c_x:.1f} cm", f"{c_y:.1f} cm", f"{highest_anomaly_z} cm",
        f"{c_val:.2f}", f"%{confidence_score:.1f}",
        f"🎯 {target_type}", f"{len(peaks)}", f"{min_val:.1f} – {max_val:.1f}"
    ]
}
st.table(pd.DataFrame(summary_data))
