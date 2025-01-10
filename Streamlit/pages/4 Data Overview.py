import pandas as pd
import streamlit as st
import pickle
# import os

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="https://raw.githubusercontent.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/refs/heads/main/Streamlit/favicon.ico",
    layout="wide")

st.title("📊 Data Overview")
st.write('With this tool, you can see overview of the dataset and the average of each column')
st.markdown('---')

# df = pd.read_csv('clean_dataset_arab_used_car.csv').loc[:,'Type':]


uploaded_file = st.sidebar.file_uploader(
    label="Upload your file", 
    type=["csv"],  # Format file yang didukung
    help="Upload file format .csv only."
)

if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' successfully uploaded!")
    st.write("Here are the preview of your data:")

    # Membaca file berdasarkan format
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, index_col=None).loc[:,'Type':'Price']
        # elif uploaded_file.name.endswith('.xlsx'):
        #     data = pd.read_excel(uploaded_file)
        
        st.dataframe(data,height=125)  # Menampilkan data dalam tabel
        
        # Pastikan data memiliki kolom yang sesuai untuk prediksi
        required_columns = ['Type',	'Region','Make','Gear_Type','Origin','Options','Year','Engine_Size','Mileage','Price']
        if all(col in data.columns for col in required_columns):
            st.sidebar.success(f"Dataset got all the required column needed to start prediction. Total rows of data : {data.shape[0]}")
            
            # Tombol untuk melanjutkan ke tahap prediksi
            selection = st.selectbox('Choose what you want to see',['Type','Region','Make','Gear Type','Origin','Options'])
            if selection == 'Type' : #st.button("Lakukan Prediksi")
                st.subheader('Average Price of Each Type')
                Type_price_average = data.groupby('Type')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False).head(10)
                st.bar_chart(data=Type_price_average,  x='Type', y='Price', x_label=None, y_label=None, color='#ffaa00',horizontal=False, stack=None, width=1080, height=300, use_container_width=True)
            elif selection == 'Region'  :
                st.subheader('Average Region of Each Type')
                region_price_average = data.groupby('Region')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
                st.bar_chart(data=region_price_average,  x='Region', y='Price', x_label=None, y_label=None, color='#028976',horizontal=False, stack=None, width=1080, height=300, use_container_width=True)
            elif selection == 'Make'  :
                st.subheader('Average Make of Each Type')
                region_price_average = data.groupby('Make')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
                st.bar_chart(data=region_price_average,  x='Make', y='Price', x_label=None, y_label=None, color='#486DE8',horizontal=False, stack=None, width=1080, height=300, use_container_width=True)
            elif selection == 'Gear Type'  :
                st.subheader('Average Gear_Type of Each Type')
                region_price_average = data.groupby('Gear_Type')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
                st.bar_chart(data=region_price_average,  x='Gear_Type', y='Price', x_label=None, y_label=None, color='#E07F9D',horizontal=True, stack=None, width=500, height=300, use_container_width=True)
            elif selection == 'Origin'  :
                st.subheader('Average Origin of Each Type')
                region_price_average = data.groupby('Origin')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
                st.bar_chart(data=region_price_average,  x='Origin', y='Price', x_label=None, y_label=None, color='#33C056',horizontal=True, stack=None, width=500, height=300, use_container_width=True)
            elif selection == 'Options'  :
                st.subheader('Average Options of Each Type')
                region_price_average = data.groupby('Options')[['Price']].mean().reset_index().sort_values(by='Price',ascending=False)
                st.bar_chart(data=region_price_average,  x='Options', y='Price', x_label=None, y_label=None, color='#F82B52',horizontal=True, stack=None, width=500, height=300, use_container_width=True)

        else:
            st.error(f"Your file dont had column needed: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"An Error Occured when reading the file: {e}")
else:
    st.info("👈 Please upload your file first to start.")
