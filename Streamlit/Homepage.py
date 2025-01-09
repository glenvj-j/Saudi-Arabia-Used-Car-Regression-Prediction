import pandas as pd
import streamlit as st
import os

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="favicon.ico",
    layout="wide")

st.sidebar.success("Select a page above")
st.image("cover.png")

st.markdown(
    """
    <h1 style='text-align: center;'>Welcome to Syarah.com Tools</h1>
    
    <p style='text-align: center;'> In this page you can predict a price for used car by filling the specification of the car.   </p>
    
    """,
    unsafe_allow_html=True
)
st.info(" To use this tools go to the left and click Calculator")

col1, col2,col3 = st.columns([2, 2, 2])  # Relative widths: 2:1:3

with col1:
    st.markdown("#### 📊 Comprehensive Vehicle Analysis")
    st.write("Our tool evaluates multiple factors to ensure precise pricing insights tailored to the Saudi market.")

with col2:
    st.markdown("#### 🔬 Powered by CatBoost")
    st.write("Leveraging CatBoost, a cutting-edge machine learning algorithm, to deliver highly accurate predictions.")

with col3:
    st.markdown("💡 Informed Decisions")
    st.write("Make confident buying or selling choices backed by data-driven insights specifically designed for the Saudi Arabian car market.")

st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #7F8C8D;">
    <p>Disclaimer: This tool is not 100% accurate, need to always be maintained.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('''
    For detail how the model work you can visit : [Link text Here](https://github.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/tree/main)
''')

# Automatically run Streamlit app from terminal
if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUN"):
        os.environ["STREAMLIT_RUN"] = "1"  # Set a flag to indicate Streamlit is running
        os.system("streamlit run Homepage.py")