import pandas as pd
import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="favicon.ico",
    layout="wide")


# st.sidebar.header("Menu List")
# st.sidebar.success("Select a page above")


# ==============

# Judul
st.title("Predict Used Car Price")

# ========
uploaded_file = st.file_uploader(
    label="Pilih file Anda", 
    type=["csv"],  # Format file yang didukung
    help="Unggah file dalam format CSV atau Excel."
)

# df = pd.read_csv(uploaded_file, index_col=None).loc[:,'Type':'Price']

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil diunggah!")
    st.write("Berikut adalah isi file yang Anda unggah:")
    
    # Membaca file berdasarkan format
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, index_col=None).loc[:,'Type':'Mileage']
        # elif uploaded_file.name.endswith('.xlsx'):
        #     data = pd.read_excel(uploaded_file)
        
        st.dataframe(data)  # Menampilkan data dalam tabel
        
        # Pastikan data memiliki kolom yang sesuai untuk prediksi
        required_columns = ['Type',	'Region','Make','Gear_Type','Origin','Options','Year','Engine_Size','Mileage']
        if all(col in data.columns for col in required_columns):
            st.success("Data memiliki semua kolom yang diperlukan untuk prediksi.")
            
            # Tombol untuk melanjutkan ke tahap prediksi
            if st.button("Lakukan Prediksi"):
                # Melakukan prediksi
                model = pickle.load(open('Model_Saudi_Arabia_Used_Cars.sav','rb'))

                predictions = model.predict(data[required_columns])
                data['Prediction'] = predictions  # Menambahkan kolom prediksi
                
                st.write("Hasil Prediksi:")
                st.dataframe(data[['Prediction']])  # Menampilkan kolom prediksi
                
                # Simpan hasil prediksi ke file untuk diunduh
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Unduh Hasil Prediksi",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
        else:
            st.error(f"File Anda tidak memiliki kolom yang diperlukan: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file untuk memulai.")




# Create a horizontal layout using columns
# Adjust the width of columns using ratios
col1, col2,col3 = st.columns([10, 1, 4])  # Relative widths: 2:1:3

with col1:
    st.write("")
    # st.write("Fill the Detail")

    # model_loaded = pickle.load(open('Model_Saudi_Arabia_Used_Cars.sav','rb'))
    # price = model_loaded.predict(df)
    # price

with col2:
    st.write("")

with col3:
    st.write("")
    # st.write("Final Prediction")
    # # st.write(f'''Prediksi harga : 
    # #          {str(price[0])}''')
    # range_error = 18
    # price_formated = str("{:,}".format(int(price[0])))
    # price_down =  str("{:,}".format((int(price[0]))-int(price[0])*(range_error/100)))
    # price_up =  str("{:,}".format((int(price[0]))+int(price[0])*(range_error/100)))

    # st.title('SAR' + ' ' + price_formated)
    # st.markdown("---")
    # st.write(f"Estimation (±{range_error}%)")
    # st.write(f"SAR {price_down} - {price_up}" )




# Automatically run Streamlit app from terminal
if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUN"):
        os.environ["STREAMLIT_RUN"] = "1"  # Set a flag to indicate Streamlit is running
        os.system("streamlit run Used_Car.py")
