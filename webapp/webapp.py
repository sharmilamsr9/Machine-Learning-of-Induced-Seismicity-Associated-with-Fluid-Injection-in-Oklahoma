import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


from utils import plot_pred, plot_input

cur_dir = Path(__file__).parent

page_title = (
    "Machine Learning of Induced Seismicity Associated with Fluid Injection in Oklahoma"
)
st.set_page_config(
    layout="centered", page_title=page_title, initial_sidebar_state="expanded"
)
# Sidebar options
model = st.sidebar.selectbox("Model", ("MLP", "Base"))
param = st.sidebar.selectbox("Parameter", ("mu", "lambda"))

# load df
df = pd.read_pickle(cur_dir.joinpath("df_final_pred_%s.pkl" % (param)))
# convert lat lon to string
df["lat"] = df.lat.apply(lambda x: "%.2f" % (x))
df["lon"] = df.lon.apply(lambda x: "%.2f" % (x))
df = df.groupby(["lat", "lon"])

lat_lon_options = sorted(df.groups.keys())
index = lat_lon_options.index(("98.40", "36.80"))
lat_lon = st.sidebar.selectbox(
    "Select Latitude, Longitude", lat_lon_options, index=index
)
print(lat_lon)

# lat = st.sidebar.selectbox("Select Latitude", set(df_temp["lat"].to_list()))
# lon = st.sidebar.selectbox("Select Longitude", set(df_temp["lon"].to_list()))

df_temp = df.get_group(lat_lon)


st.markdown("#### " + page_title)

left, right = st.columns(2)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
plot_input(axes[0], df_temp)
plot_pred(axes[1], df_temp, param)
st.pyplot(fig)
