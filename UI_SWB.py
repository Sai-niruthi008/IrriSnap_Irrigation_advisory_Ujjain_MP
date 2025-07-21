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

# ======================================================================================
# 1. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(
    page_title="Soil Water Balance Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Pixel-wise Soil Water Balance Dashboard")
st.markdown(
    "Select a variable and date to view on the map. "
    "**Click on the map** to generate a detailed time-series water balance chart for that location."
)

# ======================================================================================
# 2. DATA LOADING & CONSTANTS
# ======================================================================================
try:
    nc_path = "swb_pixel_wheat_clipped.nc"
    ds = xr.open_dataset(nc_path)
except FileNotFoundError:
    st.error(f"Data file not found. Make sure '{nc_path}' is in the same directory as the script.")
    st.stop()
except ImportError as e:
    st.error(f"""
    **Failed to import a required library, which is causing an error.**
    
    **Error:** {e}
    
    This is very likely an environment issue. Please follow the instructions to create a clean conda environment.
    The key is to install packages from the 'conda-forge' channel.
    """)
    st.stop()

# --- Define Constants from Dataset Attributes ---
FC = float(ds.attrs.get("FieldCapacity_mm", 120))
WP = float(ds.attrs.get("WiltingPoint_mm", 60))

# ======================================================================================
# 3. SIDEBAR CONTROLS (USER INPUT)
# ======================================================================================
st.sidebar.header("🗺️ Map Controls")
with st.sidebar.expander("Select Parameters", expanded=True):
    temporal_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    default_idx = temporal_vars.index("ETc") if "ETc" in temporal_vars else 0
    layer_to_map = st.selectbox(
        "Select map variable (mm):", temporal_vars,
        index=default_idx, key="var_select"
    )

    date_list = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d").tolist()
    selected_date_str = st.select_slider(
        "Select map date:", options=date_list,
        value=date_list[-1], key="date_select"
    )
    selected_time = pd.to_datetime(selected_date_str)

# ======================================================================================
# 4. INTERACTIVE MAP DISPLAY (FOLIUM)
# ======================================================================================
st.subheader(f"📍 Map of '{layer_to_map}' for {selected_date_str}")
da_map = ds[layer_to_map].sel(time=selected_time, method='nearest')
raster_data = np.nan_to_num(da_map.values)

lat = ds.y.values
lon = ds.x.values
bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
map_center = [float(np.mean(lat)), float(np.mean(lon))]

m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

# Colorize and overlay
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

# Add color scale legend
colormaps = {
    "ETc": branca.colormap.linear.YlGnBu_09,
    "SoilWater": branca.colormap.linear.YlOrBr_09,
    "Precip": branca.colormap.linear.Blues_09,
    "Irrigation": branca.colormap.linear.GnBu_09,
}

default_cmap = branca.colormap.linear.viridis.scale(vmin, vmax) 
color_map_base = colormaps.get(layer_to_map, default_cmap)
colormap = color_map_base.scale(vmin, vmax)
colormap.caption = f"{layer_to_map} (mm)"
colormap.add_to(m)


# Fit map to bounds and add controls
m.fit_bounds(bounds)
folium.LayerControl().add_to(m)
# ======================================================================================
# 5. PIXEL-WISE TIME-SERIES CHART
# ======================================================================================
with st.container():
    map_output = st_folium(m, height=600, width='100%')

    if map_output and map_output.get("last_clicked"):
        clicked = map_output["last_clicked"]
        lat_click, lon_click = clicked["lat"], clicked["lng"]

        st.markdown("---")
        st.subheader(
            f"📊 Daily Water Balance at Location ({lat_click:.4f}, {lon_click:.4f})"
        )

        pixel = ds.sel(x=lon_click, y=lat_click, method="nearest")
        df = pd.DataFrame(index=pd.to_datetime(ds.time.values))
        for var in temporal_vars:
            df[var] = pixel[var].values

        fig, ax1 = plt.subplots(figsize=(18, 6))
        ax2 = ax1.twinx()

        ax1.plot(df.index, df["SoilWater"], label="Soil Water (mm)", linewidth=2.5)
        ax2.plot(df.index, df["ETc"], label="ETc (mm/day)", linewidth=2)
        ax2.plot(df.index, df["Irrigation"], label="Irrigation (mm)", linestyle='--', linewidth=2)
        bars = ax2.bar(df.index, df["Precip"], width=0.8, label="Precip (mm)", alpha=0.6)

        for x, y in zip(df.index, df["Irrigation"]):
            if y > 0:
                ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        ax1.axhline(FC, linestyle='--', linewidth=2, label="Field Capacity (mm)")
        ax1.axhline(WP, linestyle='--', linewidth=2, label="Wilting Point (mm)")
        ax1.fill_between(df.index, FC, FC * 1.1, color='powderblue', alpha=0.3)
        ax1.fill_between(df.index, WP, FC, color='lightgreen', alpha=0.3)
        ax1.fill_between(df.index, 0, WP, color='lightcoral', alpha=0.3)

        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Soil Water (mm)", fontsize=12, color="darkgreen")
        ax2.set_ylabel("ETc / Precip / Irrigation (mm)", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="darkgreen")
        ax1.grid(True, linestyle='--', alpha=0.6)
        fig.suptitle("Daily Soil-Water Balance & Moisture Zones (Rabi 2023-24)", fontsize=16, weight='bold')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(0.9, 0.9))

        st.pyplot(fig)

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
        st.info("🖱️ **Click on the map** to see the time-series chart for that pixel.")

