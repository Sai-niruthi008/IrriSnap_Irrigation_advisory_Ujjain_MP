# app.py

import streamlit as st
import xarray as xr
import folium
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from PIL import Image
from base64 import b64encode
import io
import branca

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Soil Water Balance Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Pixel-wise Soil Water Balance Dashboard")
st.markdown(
    "**📌 Tip:** Click on the map below to generate a water balance time-series chart for that location."
)

# 2. LOAD DATA
try:
    nc_path = "swb_pixel_wheat_clipped.nc"
    ds = xr.open_dataset(nc_path)
except FileNotFoundError:
    st.error(f"Data file not found: '{nc_path}'")
    st.stop()

# --- Constants
FC = float(ds.attrs.get("FieldCapacity_mm", 120))
WP = float(ds.attrs.get("WiltingPoint_mm", 60))

# 3. SIDEBAR CONTROLS
st.sidebar.header("🗺️ Map Controls")
with st.sidebar.expander("Select Parameters", expanded=True):
    temporal_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    default_idx = temporal_vars.index("ETc") if "ETc" in temporal_vars else 0
    layer_to_map = st.selectbox("Select map variable (mm):", temporal_vars, index=default_idx)
    date_list = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d").tolist()
    selected_date_str = st.select_slider("Select map date:", options=date_list, value=date_list[-1])
    selected_time = pd.to_datetime(selected_date_str)

# 4. HANDLE MAP CLICK
map_output = None
lat_click = lon_click = None
pixel = df = None

# PREVIEW CHART ABOVE MAP
with st.container():
    st.subheader("📊 Water Balance Chart (Click map to update)")
    placeholder = st.empty()

# 5. MAP SETUP
da_map = ds[layer_to_map].sel(time=selected_time, method='nearest')
raster_data = np.nan_to_num(da_map.values)
lat = ds.y.values
lon = ds.x.values
bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
map_center = [float(np.mean(lat)), float(np.mean(lon))]

m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

# Overlay
vmin, vmax = float(da_map.min()), float(da_map.max())
cmap = plt.get_cmap('viridis')
normed = (raster_data - vmin) / (vmax - vmin)
img_arr = (cmap(normed) * 255).astype(np.uint8)
img = Image.fromarray(img_arr, 'RGBA')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
encoded = b64encode(buffer.getvalue()).decode()

folium.raster_layers.ImageOverlay(
    image=f"data:image/png;base64,{encoded}",
    bounds=bounds,
    opacity=0.7,
    name=f"{layer_to_map} (mm)"
).add_to(m)

# Legend
colormaps = {
    "ETc": branca.colormap.linear.YlGnBu_09,
    "SoilWater": branca.colormap.linear.YlOrBr_09,
    "Precip": branca.colormap.linear.Blues_09,
    "Irrigation": branca.colormap.linear.GnBu_09,
}
colormap = colormaps.get(layer_to_map, branca.colormap.linear.viridis).scale(vmin, vmax)
colormap.caption = f"{layer_to_map} (mm)"
colormap.add_to(m)

m.fit_bounds(bounds)
folium.LayerControl().add_to(m)

# 6. DISPLAY MAP
st.subheader(f"🗺️ Map of '{layer_to_map}' on {selected_date_str}")
map_output = st_folium(m, height=600, width="100%")

# 7. HANDLE CLICKED PIXEL DATA
if map_output and map_output.get("last_clicked"):
    clicked = map_output["last_clicked"]
    lat_click, lon_click = clicked["lat"], clicked["lng"]
    pixel = ds.sel(x=lon_click, y=lat_click, method="nearest")

    df = pd.DataFrame(index=pd.to_datetime(ds.time.values))
    for var in temporal_vars:
        df[var] = pixel[var].values

    # Chart Plot
    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["SoilWater"], label="Soil Water (mm)", color='green', linewidth=2.5)
    ax2.plot(df.index, df["ETc"], label="ETc (mm)", color='blue', linewidth=2)
    ax2.plot(df.index, df["Irrigation"], label="Irrigation (mm)", color='orange', linestyle='--', linewidth=2)
    ax2.bar(df.index, df["Precip"], label="Precip (mm)", color='skyblue', alpha=0.5)

    ax1.axhline(FC, linestyle='--', color='black', linewidth=2, label="Field Capacity")
    ax1.axhline(WP, linestyle='--', color='gray', linewidth=2, label="Wilting Point")
    ax1.fill_between(df.index, FC, FC * 1.1, color='powderblue', alpha=0.3)
    ax1.fill_between(df.index, WP, FC, color='lightgreen', alpha=0.3)
    ax1.fill_between(df.index, 0, WP, color='lightcoral', alpha=0.3)

    ax1.set_ylabel("Soil Water (mm)", color='green', fontsize=12)
    ax2.set_ylabel("ETc / Precip / Irrigation (mm)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig.suptitle(f"Daily Soil-Water Balance at ({lat_click:.4f}, {lon_click:.4f})", fontsize=16, weight='bold')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(0.9, 0.9))

    placeholder.pyplot(fig)

    # Download button
    @st.cache_data
    def convert_df_to_csv(df_in):
        return df_in.to_csv(index=True).encode('utf-8')

    csv_data = convert_df_to_csv(df.round(2))
    st.download_button(
        label="📥 Download Time Series as CSV",
        data=csv_data,
        file_name=f"swb_timeseries_{lat_click:.4f}_{lon_click:.4f}.csv",
        mime="text/csv"
    )

else:
    placeholder.info("🖱️ Click on the map below to generate the water balance chart.")
