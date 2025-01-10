import pandas as pd
import streamlit as st
import pickle
import requests
# import os

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="https://raw.githubusercontent.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/refs/heads/main/Streamlit/favicon.ico",
    layout="wide")


# st.sidebar.header("Menu List")
# st.sidebar.success("Select a page above")


# ==============

# Judul
st.title("🚘 Predict Used Car Price for Batch Data")

# ========
uploaded_file = st.sidebar.file_uploader(
    label="Upload your file", 
    type=["csv"],  # Format file yang didukung
    help="Upload file format .csv only."
)

# df = pd.read_csv(uploaded_file, index_col=None).loc[:,'Type':'Price']

if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' successfully uploaded!")
    st.write("Here are the preview of your data:")
    
    # Membaca file berdasarkan format
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, index_col=None).loc[:,'Type':'Mileage']
        # elif uploaded_file.name.endswith('.xlsx'):
        #     data = pd.read_excel(uploaded_file)
        
        st.dataframe(data,height=125)  # Menampilkan data dalam tabel
        
        # Pastikan data memiliki kolom yang sesuai untuk prediksi
        required_columns = ['Type',	'Region','Make','Gear_Type','Origin','Options','Year','Engine_Size','Mileage']
        if all(col in data.columns for col in required_columns):
            st.success(f"Dataset got all the required column needed to start prediction. Total rows of data : {data.shape[0]}")
            
            # Tombol untuk melanjutkan ke tahap prediksi
            if st.button("Predict the Price"):
                # Melakukan prediksi

                url = "https://github.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/raw/refs/heads/main/Model_Saudi_Arabia_Used_Cars.sav"
                response = requests.get(url)
                model = pickle.loads(response.content)

                # model = pickle.load(open('Model_Saudi_Arabia_Used_Cars.sav','rb'))

                predictions = model.predict(data[required_columns])
                data['Prediction'] = predictions.round().astype(int)  # Menambahkan kolom prediksi
                
                st.write("Prediction Result:")
                st.dataframe(data[['Prediction']])  # Menampilkan kolom prediksi
                
                # Simpan hasil prediksi ke file untuk diunduh
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Result",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
        else:
            st.error(f"Your file dont had column needed: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"An Error Occured when reading the file: {e}")
else:
    st.info("👈 Please upload your file first to start.")
