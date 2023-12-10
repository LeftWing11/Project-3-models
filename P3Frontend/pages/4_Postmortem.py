import streamlit as st

st.set_page_config(page_title = "Postmortem")

conclusion = st.container()
recommendations = st.container()

success_col, challenges_col = st.columns(2)

success_col.subheader("Successes:")
success_col.success("""


1. Identified and categorized major macroeconomic indicators, including but not limited to GDP growth, inflation rates, interest rates, and unemployment rates; 

2. Developed methodology to measure the impact of macroeconomic indicators on asset class price changes;

3. Created a standardized framework that allowed for consistent measurement and comparison across various indicators and asset classes;

4. Leveraged statistical methods and data visualization tools to identify patterns and trends;

5. Predictive models were developed to forecast potential changes in asset prices based on macroeconomic shifts.

""")

challenges_col.subheader("Challenges:")
challenges_col.warning("""

1. Determining the most relevant macroeconomic indicators for specific asset classes;

2. Reconciling diverse data sources and normalizing data for different time periods;

3. Data cleaning and handling missing data points, especially when dealing with historical datasets;

4. Balancing model complexity with interpretability.

""")

with conclusion:

    st.subheader(" Conclusion")
    st.markdown("""
The project successfully achieved its objectives, providing valuable insights into the correlation between macroeconomic indicators and asset prices. The identification of relevant indicators, establishment of a robust methodology, analysis of historical data, and development of predictive models collectively contributed to a comprehensive understanding of the dynamics between macroeconomics and asset classes.
""")

with recommendations:
    st.subheader("Recommendations for Future Projects:")
    st.markdown("""
Continuously update and refine the macroeconomic indicator selection process to adapt to changing economic landscapes.

Explore additional data sources and consider incorporating alternative data sets to enhance the accuracy and robustness of predictive models.

Conduct regular reviews of the methodology to incorporate advancements in statistical techniques and machine learning algorithms.

Foster collaboration between experts in related fields to ensure a multidisciplinary approach to problem-solving.
""")
