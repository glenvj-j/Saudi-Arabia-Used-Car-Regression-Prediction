import pandas as pd
import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="favicon.ico",
    layout="wide")


df = pd.read_csv('clean_dataset_arab_used_car.csv').loc[:,'Type':]

st.write(f'Total Data Trained : {df.shape[0]}')


col1, col2,col3 = st.columns([4, 1, 4])  # Relative widths: 2:1:3
with col1:
    st.subheader('Average Price of Each Type')
    Type_price_average = df.groupby('Type')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False).head(10)
    st.bar_chart(data=Type_price_average,  x='Type', y='Price', x_label=None, y_label=None, color='#ffaa00',horizontal=False, stack=None, width=1080, height=300, use_container_width=True)

with col3:
    st.subheader('Average Price of Each Region')
    region_price_average = df.groupby('Region')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
    st.bar_chart(data=region_price_average,  x='Region', y='Price', x_label=None, y_label=None, color='#ffaa00',horizontal=False, stack=None, width=1080, height=300, use_container_width=True)