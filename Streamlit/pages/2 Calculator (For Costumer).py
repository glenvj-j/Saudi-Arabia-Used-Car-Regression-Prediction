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


# df = pd.read_csv('clean_dataset_arab_used_car.csv').loc[:,'Type':]

def user_input_features():
    df = pd.read_csv('clean_dataset_arab_used_car.csv', index_col=None)
    # df = pd.read_csv('clean_dataset_arab_used_car.csv').loc[:,'Type':]
    # Make & Type
    list_brand = []
    list_type = []

    for brand in sorted(df['Make'].unique()) :
        type = sorted(list(df[df['Make']==brand]['Type'].unique()))
        list_type.append(type)
        list_brand.append(brand)

    df_brand_type = pd.DataFrame()
    df_brand_type['Make'] = list_brand
    df_brand_type['Type'] = list_type

    # ==Make==
    Make_allowed_values = df_brand_type['Make'].tolist()

    Make = st.selectbox("Select Make (Brand of Car)", options=Make_allowed_values)

    # ==Type==
    # Filter Type based on the selected Make and convert it to a list
    Type_allowed_values = df_brand_type[df_brand_type['Make'] == Make]['Type'].tolist()[0]

    # Use the list in the selectbox
    Type = st.selectbox("Select Type", options=Type_allowed_values)




    # Origin & Region
    list_Origin = []
    list_Region = []

    for origin in sorted(df['Origin'].unique()) :
        region = sorted(list(df[df["Origin"]==origin]['Region'].unique()))
        list_Region.append(region)
        list_Origin.append(origin)

    df_origin_region = pd.DataFrame()
    df_origin_region['Origin'] = list_Origin
    df_origin_region['Region'] = list_Region
    df_origin_region = df_origin_region.loc[[0,3,2]]


    origin_allowed_values = df_origin_region['Origin'].tolist()

    Origin = st.selectbox("Select Origin", options=origin_allowed_values)

    # ==Region==
    # Filter Type based on the selected Make and convert it to a list
    Region_allowed_values = df_origin_region[df_origin_region['Origin'] == Origin]['Region'].tolist()[0]

    # Use the list in the selectbox
    Region = st.selectbox("Select Region", options=Region_allowed_values)


    # ==Gear Type==
    Gear_Type = st.radio('Fill Gear Type:', df['Gear_Type'].unique().tolist(), horizontal=True)

    # ==Options==
    Options = st.radio('Fill Options:', df['Options'].unique().tolist(), horizontal=True)

    # ==Engine_Size==
    Engine_Size = st.number_input('Fill Engine Size',min_value=1.0, max_value=9.0,step=0.1,value=5.0)

    # ==Year==
    Year = st.number_input('Fill Year (2003 - 2021)',min_value=2003, max_value=2021,step=1,value=2010)

    # ==Mileage==
    Mileage = st.number_input('Fill Mileage (in KM per hour)',min_value=0, max_value=376000,step=100,value=0)
    
    df_new = pd.DataFrame()
    df_new['Type'] = [Type]
    df_new['Region'] = [Region]
    df_new['Make'] = [Make]
    df_new['Gear_Type'] = [Gear_Type]
    df_new['Origin'] = [Origin]
    df_new['Options'] = [Options]
    df_new['Year'] = [Year]
    df_new['Engine_Size'] = [Engine_Size]
    df_new['Mileage'] = [Mileage]

    return df_new

# Create a horizontal layout using columns
# Adjust the width of columns using ratios
col1, col2,col3 = st.columns([10, 1, 4])  # Relative widths: 2:1:3

with col1:
    st.write("Fill the Detail")

    df_customer = user_input_features()

    model_loaded = pickle.load(open('Model_Saudi_Arabia_Used_Cars.sav','rb'))
    price = model_loaded.predict(df_customer)

with col2:
    st.write("")

with col3:
    st.write("Final Prediction")
    # st.write(f'''Prediksi harga : 
    #          {str(price[0])}''')
    range_error = 18
    price_formated = str("{:,}".format(int(price[0])))
    price_down =  str("{:,}".format((int(price[0]))-int(price[0])*(range_error/100)))
    price_up =  str("{:,}".format((int(price[0]))+int(price[0])*(range_error/100)))

    st.title('SAR' + ' ' + price_formated)
    st.markdown("---")
    st.write(f"Estimation (±{range_error}%)")
    st.write(f"SAR {price_down} - {price_up}" )




# Automatically run Streamlit app from terminal
if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUN"):
        os.environ["STREAMLIT_RUN"] = "1"  # Set a flag to indicate Streamlit is running
        os.system("streamlit run Used_Car.py")
