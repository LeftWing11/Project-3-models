import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Data Cleanup and Preprocessing")

header = st.container()
dataset = st.container()


with header:
    
    st.subheader("Summary")
    st.markdown('''

The data exploration phase involved meticulous scrutiny of various macroeconomic indicators and asset class prices. Challenges emerged in reconciling diverse data sources and normalizing data for different time periods. Advanced data cleaning techniques were applied to address missing values, outliers, and inconsistencies, ensuring the integrity of the dataset.
''')

    st.subheader("Data Cleaning and Preprocessing Techniques Used")

    st.markdown("""
1. Missing Values Handling

2. Outlier Detection and Treatment

3. Time-Series Alignment

4. Scaling
""")

with dataset:
    st.text("Dataset quick look")
    df = pd.read_csv('NH_Data.csv')
    st.write(df.head(20))
    
    st.subheader("Model Summary")

    st.markdown("""
We employed a machine learning approach utilizing methods such as Random Forests and linear regression. The model was chosen for its ability to handle complex relationships within the data and offer robust predictions. The flexibility of the methods allows for capturing both linear and non-linear patterns, making them suitable for the intricate dynamics of macroeconomic indicators and asset prices.

Several techniques were employed to assess the performance of our predictive models. These included standard metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared. Additionally, we conducted cross-validation to ensure the generalizability of our models to new data.
""")
    

    
    
    


    


