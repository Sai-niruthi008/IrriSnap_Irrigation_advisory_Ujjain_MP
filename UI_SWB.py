# app.py
import os
import io
import numpy as np
import pandas as pd
import xarray as xr
import folium
import matplotlib.pyplot as plt
import branca
import streamlit as st
from PIL import Image
from base64 import b64encode
from streamlit_folium import st_folium

# === 1) PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Soil Water Balance Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Pixel-wise Soil Water Balance")

# === 2) DATA LOADING & CONSTANTS ===
nc_path = "swb_pixel_wheat_clipped.nc"
try:
    ds = xr.open_dataset(nc_path, engine="netcdf4")
except Exception as e:
    st.error(f"Failed to load `{nc_path}`:\n>{e}")
    st.stop()

FC = float(ds.attrs.get("FieldCapacity_mm", 120))
WP = float(ds.attrs.get("WiltingPoint_mm", 60))

# === 3) SIDEBAR CONTROLS ===
with st.sidebar:
    st.header("Map Controls")
    temporal_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    layer_to_map = st.selectbox("Variable (mm)", temporal_vars, index=temporal_vars.index("ETc") if "ETc" in temporal_vars else 0)
    dates = pd.to_datetime(ds.time.values)
    selected_time = st.slider("Date", value=dates[-1], min_value=dates[0], max_value=dates[-1], format="YYYY-MM-DD")
    st.markdown("---")
    st.markdown(f"**Field Capacity:** {FC} mm  \n**Wilting Point:** {WP} mm")

# === 4) PREP MAP LAYER ===
da_map = ds[layer_to_map].sel(time=selected_time, method="nearest")
raster = np.nan_to_num(da_map.values)
lat = ds.y.values; lon = ds.x.values
bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
center = [float(lat.mean()), float(lon.mean())]

# Render as PNG base64
vmin, vmax = float(da_map.min()), float(da_map.max())
norm = (raster - vmin) / (vmax - vmin)
rgba = (plt.get_cmap("viridis")(norm) * 255).astype(np.uint8)
buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
overlay_data = "data:image/png;base64," + b64encode(buf.getvalue()).decode()

# Prepare colormap
cmaps = {
    "ETc": branca.colormap.linear.YlGnBu_09,
    "SoilWater": branca.colormap.linear.YlOrBr_09,
    "Precip": branca.colormap.linear.Blues_09,
    "Irrigation": branca.colormap.linear.GnBu_09
}
cmap = cmaps.get(layer_to_map, branca.colormap.linear.viridis).scale(vmin, vmax)
cmap.caption = f"{layer_to_map} (mm)"

# === 5) LAYOUT: CHART ON TOP, MAP BELOW ===
click_info = None

# Placeholder for chart & download
chart_placeholder = st.container()

with chart_placeholder:
    st.subheader("📊 Click the map to view time-series")
    chart_col1, chart_col2 = st.columns([3,1])
    with chart_col2:
        # empty space for download button later
        st.write("")

# Separator
st.markdown("---")

# Map
st.subheader(f"🗺️ {layer_to_map} on {selected_time.strftime('%Y-%m-%d')}")
m = folium.Map(location=center, zoom_start=8, tiles=None)
folium.TileLayer("CartoDB positron", name="OSM").add_to(m)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Satellite", attr="Esri"
).add_to(m)
folium.LayerControl(position="topright").add_to(m)
folium.raster_layers.ImageOverlay(
    image=overlay_data,
    bounds=bounds,
    opacity=0.7,
    name=layer_to_map
).add_to(m)
cmap.add_to(m)
m.fit_bounds(bounds)

map_output = st_folium(m, height=600, width="100%")

# === 6) ON CLICK: SHOW TIME-SERIES ===
if map_output and map_output.get("last_clicked"):
    click_info = map_output["last_clicked"]
    lat_c, lon_c = click_info["lat"], click_info["lng"]

    pixel = ds.sel(x=lon_c, y=lat_c, method="nearest")
    df = pd.DataFrame({var: pixel[var].values for var in temporal_vars}, index=pd.to_datetime(ds.time.values))

    # Re-render chart in placeholder
    with chart_placeholder:
        st.subheader(f"📍 Time-Series at ({lat_c:.4f}, {lon_c:.4f})")
        fig, ax1 = plt.subplots(figsize=(10,4))
        ax2 = ax1.twinx()

        # SoilWater
        ax1.plot(df.index, df["SoilWater"], color="green", lw=2.5, label="SoilWater")
        ax1.fill_between(df.index, 0, WP, color="lightcoral", alpha=0.3)
        ax1.fill_between(df.index, WP, FC, color="lightgreen", alpha=0.3)
        ax1.fill_between(df.index, FC, FC*1.1, color="powderblue", alpha=0.3)
        ax1.axhline(WP, ls="--", color="darkred", lw=1)
        ax1.axhline(FC, ls="--", color="darkblue", lw=1)
        ax1.set_ylabel("Soil Water (mm)", color="green")

        # Fluxes
        ax2.bar(df.index, df["Precip"], alpha=0.3, label="Precip")
        ax2.plot(df.index, df["ETc"], color="blue", lw=1.5, label="ETc")
        ax2.plot(df.index, df["Irrigation"], color="orange", ls="--", lw=1.5, label="Irrigation")
        ax2.set_ylabel("Fluxes (mm/day)")

        ax1.set_xlabel("Date"); ax1.grid(True, ls="--", alpha=0.5)
        fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.9))
        st.pyplot(fig, use_container_width=True)

        # Download
        @st.cache_data
        def to_csv(dataframe):
            return dataframe.to_csv().encode("utf-8")
        csv = to_csv(df)
        chart_col1, chart_col2 = st.columns([3,1])
        with chart_col2:
            st.download_button("📥 Download CSV", csv, file_name="timeseries.csv")

