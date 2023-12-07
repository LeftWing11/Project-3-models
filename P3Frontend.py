import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# cache csv file so as not be ran repeatedly when interacting or modufying the code
@st.cache
def get_data(filename):
    df = pd.read_csv(filename)
    
    return df

# Page 1: Title, Authors, and Summary
def page_1():
    st.title("Project 3")
    st.subheader("By: Ashwin Rao, Nicholas Hugh, Hardy Ty, Jenny Vhong")

    st.header("Project Purpose")
    st.write("""
    The project aims to evaluate the correlation between changes in macroeconomic indicators 
    and fluctuations in asset class prices. It addresses the need to comprehend the extent to 
    which variations in macroeconomic factors impact diverse asset classes.
    """)
    
    st.subheader("Objectives")
    st.write("""
    - Identify and categorize major macroeconomic indicators affecting asset prices.
    - Establish a quantifiable methodology to measure the impact of these indicators on asset class price changes.
    - Analyze historical data to assess the correlation between macroeconomic indicators and asset price movements.
    - Develop predictive models to forecast potential changes in asset prices based on macroeconomic shifts.
    """)

    st.subheader("Macroeconomic Indicators")
    st.write("""
    - Gross Domestic Product (GDP)
    - Inflation Rate
    - Unemployment Rate
    - Interest Rates -> AUS, US
    - Consumer Price Index (CPI) to measure inflation
    - Currency Prices
       - We will start with a wide range of macro variables, but through PCA, we will look to identify the most important factors.
    """)

    st.subheader("Asset Classes")
    st.write("""
    - Bonds
    - Stocks
    - Property
    - Crypto Currencies
       - Bitcoin
       - ETH
    """)

    st.subheader("Data Sources")
    st.write("""
    - Nick’s data source -> Still assessing whether we can get access
    - Government agencies such as ABS
    - Central Bank
    - Kaggle
    - Data World
    - Data.gov
    - Public APIs
    """)


# Page 2: Presentation of the Project
def page_2():
    st.title("Presentation")
    # Add presentation content here

# Page 3: Solution of the Project
def page_3():
    st.title("Solution")

    # Sidebar slider for the graph
    slider_value = st.sidebar.slider("Select a value", 0, 100, 50)

    # Generate a simple graph using the slider value
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * slider_value
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)
    

    # Sidebar radio button for selecting Asset Class
    selected_asset_class = st.sidebar.radio("Select an Asset Class", ["Bonds", "Stocks", "Property", "Crypto Currencies"])

    # Sidebar multi-select dropdown for Crypto Currencies
    if selected_asset_class == "Crypto Currencies":
        selected_crypto_currencies = st.sidebar.multiselect("Select Crypto Currencies", ["Bitcoin", "ETH"])

# Page 4: Conclusion
def page_4():
    st.title("Conclusion")
    # Add conclusion content here

# Page 5: Acknowledgements or References
def page_5():
    st.title("Acknowledgements or References")
    # Add acknowledgements or references content here

# Page 6: Thank You!
def page_6():
    st.title("Thank You!")
    # Add message here

# Page Navigation
pages = {
    "Page 1": page_1,
    "Page 2": page_2,
    "Page 3": page_3,
    "Page 4": page_4,
    "Page 5": page_5,
    "Page 6": page_6,
}

selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
pages[selected_page]()
