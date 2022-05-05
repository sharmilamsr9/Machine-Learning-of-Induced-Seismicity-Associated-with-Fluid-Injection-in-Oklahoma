import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

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

# select year_month
min_value = datetime.date(2012, 1, 1)
max_value = datetime.date(2018, 1, 31)
default_value = datetime.date(2016, 1, 31)
date_month = st.sidebar.date_input(
    "Enter the Year and Month to predict",
    default_value,
    min_value=min_value,
    max_value=max_value,
)

year_month = date_month.year + round((date_month.month - 1) / 12, 2)
print("**** year_month", year_month)

# processing
df_temp = df.get_group(lat_lon)


st.markdown("#### " + page_title)

left, right = st.columns(2)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
plot_input(axes[0], df_temp)
plot_pred(axes[1], df_temp, param)
st.pyplot(fig)

row = df_temp[df_temp.year_month == year_month].iloc[0]
month_str = date_month.strftime("%B")
st.markdown(f"## **{param}** for {date_month.year} {month_str}:")
st.markdown(f"### Actual = {row[param]:.4f} , Predicted = {row['Pred']:.4f} ")
