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
    page_title="Irrisnap_Irrigation_Advisories",
    page_icon="🌾",
    layout="wide"
)

# ======================================================================================
# 2. DATA LOADING & CACHING
# ======================================================================================
@st.cache_data
def load_data(path):
    """Loads and caches the NetCDF dataset."""
    try:
        ds = xr.open_dataset(path)
        FC = float(ds.attrs.get("FieldCapacity_mm", 120))
        WP = float(ds.attrs.get("WiltingPoint_mm", 60))
        return ds, FC, WP
    except FileNotFoundError:
        st.error(f"Data file not found. Make sure '{path}' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

nc_path = "swb_pixel_wheat_clipped.nc"
ds, FC, WP = load_data(nc_path)


# ======================================================================================
# 3. HELPER FUNCTIONS (for Map and Chart Creation)
# ======================================================================================
def create_map(data_array, variable_name):
    """Creates a Folium map with a transparent raster overlay for NoData values."""
    lat = data_array.y.values
    lon = data_array.x.values
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
    map_center = [lat.mean(), lon.mean()]

    m = folium.Map(location=map_center, zoom_start=8, tiles=None)

    folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Satellite Hybrid',
        overlay=False,
        control=True
    ).add_to(m)

    # ===================================================================
    # --- TRANSPARENCY FIX FOR NODATA APPLIED HERE ---
    # ===================================================================
    # 1. Get the raw numpy array, keeping NaNs
    raster_data = data_array.values

    # 2. Create a colormap and set 'bad' (NaN) values to be transparent
    #    Use .copy() to avoid modifying the global colormap instance
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(alpha=0)

    # 3. Calculate vmin/vmax ignoring NaNs for accurate color scaling
    vmin = np.nanmin(raster_data)
    vmax = np.nanmax(raster_data)

    # 4. Normalize the data. NaNs will be handled by the colormap.
    normed_data = (raster_data - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else np.zeros_like(raster_data)
    
    # 5. Apply the colormap. Matplotlib now correctly maps NaNs to transparent.
    img_arr = (cmap(normed_data) * 255).astype(np.uint8)
    # ===================================================================

    img = Image.fromarray(img_arr, 'RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode()
    image_url = f"data:image/png;base64,{encoded}"

    folium.raster_layers.ImageOverlay(
        image=image_url,
        bounds=bounds,
        opacity=0.8, # Opacity can be slightly higher now
        name=f"{variable_name} Layer"
    ).add_to(m)

    # Use the vmin/vmax from the actual data for the legend
    colormap = branca.colormap.linear.viridis.scale(vmin, vmax)
    colormap.caption = f"{variable_name} (mm)"
    m.add_child(colormap)

    m.fit_bounds(bounds)
    folium.LayerControl().add_to(m)
    return m

def create_timeseries_chart(df, lat, lon):
    """Creates a Matplotlib time-series chart with irrigation labels."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["SoilWater"], label="Soil Water (mm)", color='darkgreen', linewidth=2.5, zorder=10)
    ax2.plot(df.index, df["ETc"], label="ETc (mm/day)", color='orangered', linewidth=1.5, zorder=5)
    ax2.bar(df.index, df["Precip"], width=0.8, label="Precip (mm)", color='royalblue', alpha=0.6)
    
    irrig_mask = df["Irrigation"] > 0
    ax2.stem(df.index[irrig_mask], df["Irrigation"][irrig_mask], 
             linefmt='c-', markerfmt='co', basefmt=" ", 
             label="Irrigation (mm)")

    for date, value in df.loc[irrig_mask, "Irrigation"].items():
        ax2.annotate(f'{value:.1f}',
                     xy=(date, value),
                     textcoords="offset points",
                     xytext=(0, 8),
                     ha='center',
                     fontsize=9,
                     color='darkcyan')

    ax1.axhline(FC, color='blue', linestyle='--', linewidth=1.5, label="Field Capacity")
    ax1.axhline(WP, color='red', linestyle='--', linewidth=1.5, label="Wilting Point")
    ax1.fill_between(df.index, WP, FC, color='lightgreen', alpha=0.4, label='Optimal Zone')
    ax1.fill_between(df.index, 0, WP, color='lightcoral', alpha=0.4, label='Stress Zone')
    
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Soil Water (mm)", fontsize=12, color="darkgreen")
    ax2.set_ylabel("Water Flux (mm)", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="darkgreen")
    ax1.set_ylim(bottom=0)
    # Dynamic y-axis for ax2 to ensure labels fit
    ax2_max = max(df["Irrigation"].max(), df["Precip"].max(), df["ETc"].max())
    ax2.set_ylim(bottom=0, top=ax2_max * 1.25) 
    
    fig.suptitle(f"Daily Water Balance at ({lat:.4f}, {lon:.4f})", fontsize=16, weight='bold')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=True)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig

# ======================================================================================
# 4. MAIN APP INTERFACE
# ======================================================================================
st.title = ("Irrisnap Pixel wise Irrigation Advisories Dashboard")

st.markdown(
    "Select a variable and date from the sidebar to view the spatial map. "
    "**Click on any pixel** on the map to generate a detailed time-series water balance chart for that location."
)

st.sidebar.header("🗺️ Map Controls")
temporal_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
default_idx = temporal_vars.index("ETc") if "ETc" in temporal_vars else 0
layer_to_map = st.sidebar.selectbox(
    "Select map variable:", temporal_vars,
    index=default_idx,
    help="Choose the data layer to display on the map."
)

date_list = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d").tolist()
selected_date_str = st.sidebar.select_slider(
    "Select map date:", options=date_list,
    value=date_list[-1],
    help="Slide to choose the date for the map view."
)
selected_time = pd.to_datetime(selected_date_str)

map_col, chart_col = st.columns([3, 2])

with map_col:
    st.subheader(f"📍 Map of '{layer_to_map}' for {selected_date_str}")
    da_map = ds[layer_to_map].sel(time=selected_time, method='nearest')
    
    folium_map = create_map(da_map, layer_to_map)
    map_output = st_folium(
        folium_map, 
        height=600, 
        width='100%', 
        returned_objects=['last_clicked']
    )

with chart_col:
    st.subheader("📊 Time-Series Analysis")
    if map_output and map_output.get("last_clicked"):
        clicked = map_output["last_clicked"]
        lat_click, lon_click = clicked["lat"], clicked["lng"]

        st.markdown(f"**Selected Location:** `{lat_click:.4f}°N, {lon_click:.4f}°E`")

        pixel = ds.sel(x=lon_click, y=lat_click, method="nearest")
        df = pd.DataFrame(index=pd.to_datetime(ds.time.values))
        for var in temporal_vars:
            df[var] = pixel[var].values

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Precip.", f"{df['Precip'].sum():.1f} mm")
        metric_col2.metric("Mean ETc", f"{df['ETc'].mean():.1f} mm/day")
        metric_col3.metric("Total Irrig.", f"{df['Irrigation'].sum():.1f} mm")

        ts_chart = create_timeseries_chart(df, lat_click, lon_click)
        st.pyplot(ts_chart)

        with st.expander("📂 View and Download Pixel Data"):
            st.dataframe(df.round(2))
            
            @st.cache_data
            def convert_df_to_csv(df_in):
                return df_in.to_csv(index=True).encode('utf-8')

            csv_data = convert_df_to_csv(df.round(2))
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"swb_timeseries_{lat_click:.4f}_{lon_click:.4f}.csv",
                mime="text/csv"
            )

    else:
        st.info("🖱️ **Click a location on the map** to view its detailed water balance.")
        st.image("https://i.imgur.com/gYy8n4f.gif", caption="Click on the map to get started!")
