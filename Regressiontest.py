import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('NH_Data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date',inplace = True)

df = df.drop([df.index[0], df.index[-1]])
df = df.dropna()

df['Aus Real GDP mom%'] = df["Aus Real GDP mom%"].astype(float)
df['Aus Nominal GDP mom%'] = df["Aus Nominal GDP mom%"].astype(float)



# setting the y-variables
df['S&P/ASX 200 - Energy Returns'] = df["S&P/ASX 200  Energy   ( TR)m"].pct_change()
df['S&P/ASX 200 - Materials Returns'] = df["S&P/ASX 200 - Materials   ( TR)m"].pct_change()
df['S&P/ASX 200 - Telecommunication Returns'] = df["S&P/ASX 200 - Telecommunication Services   ( TR)m"].pct_change()


# setting the x-variables
types = ['Unemployment','CPI - Inflation','Terms of Trade','Total Assests to Disposable Income','Underemployment','Real GDP','Nominal GDP']
x_variables = ['Aus Employed - Unemployment Rate', 'Aus CPI All Groups','Aus Terms of Trade','Aus Total Assets to Disposable Income','Aust Underemployment rate','Aus Real GDP mom%','Aus Nominal GDP mom%']
X = df[x_variables]


y_SPASX_Energy = df['S&P/ASX 200 - Energy Returns'].dropna()
y_SPASX_Materials = df['S&P/ASX 200 - Materials Returns'].dropna()
y_SPASX_Telecommunication = df['S&P/ASX 200 - Telecommunication Returns'].dropna()        
               
               
training_begin = X.index.min()
training_end = X.index.min() + pd.DateOffset(months=60)

X_train, X_test = X[training_begin:training_end], X[training_end:]
X_train  = X_train.drop([X_train.index[0]])

#setting y variables
y_SPASX_Energy_train, y_SPASX_Energy_test = y_SPASX_Energy[training_begin:training_end], y_SPASX_Energy[training_end:]
y_SPASX_Materials_train, y_SPASX_Materials_test = y_SPASX_Materials[training_begin:training_end], y_SPASX_Materials[training_end:]
y_SPASX_Telecommunication_train, y_SPASX_Telecommunication_test = y_SPASX_Telecommunication[training_begin:training_end], y_SPASX_Telecommunication[training_end:]


#model creating
model_Energy = LinearRegression().fit(X_train, y_SPASX_Energy_train)
model_Materials = LinearRegression().fit(X_train, y_SPASX_Materials_train)
model_Telecommunication = LinearRegression().fit(X_train, y_SPASX_Telecommunication_train)
               
#model predictions               
Energy_predictions = model_Energy.predict(X_test)
Materials_predictions = model_Materials.predict(X_test)
Telecommunication_predictions = model_Telecommunication.predict(X_test)

#Getting Coeffecients

Energy_intercept = model_Energy.intercept_

Energy_variables = ['unemployment', 'inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Energy_coefficients = model_Energy.coef_

for i, variable in enumerate(Energy_variables):
    globals()[f"Energy_{variable}"] = Energy_coefficients[i-1]



Materials_intercept = model_Materials.intercept_

Materials_variables = ['inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Materials_coefficients = model_Materials.coef_

for i, variable in enumerate(Materials_variables):
    globals()[f"Materials_{variable}"] = Materials_coefficients[i]


Telecommunication_intercept = model_Telecommunication.intercept_

Telecommunication_variables = ['unemployment', 'inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Telecommunication_coefficients = model_Telecommunication.intercept_, *model_Telecommunication.coef_

for i, variable in enumerate(Telecommunication_variables):
    globals()[f"Telecommunication_{variable}"] = Telecommunication_coefficients[i]


    
# Streamlit app

st.title('Effect of Variables on Returns')
st.sidebar.header('Adjust Variables')


unemployment_slider = st.sidebar.slider('Unemployment', df['Aus Employed - Unemployment Rate'].min(), df['Aus Employed - Unemployment Rate'].max())
inflation_slider = st.sidebar.slider('Inflation', df['Aus CPI All Groups'].min(), df['Aus CPI All Groups'].max())
terms_of_trade_slider = st.sidebar.slider('Terms of trade', df['Aus Terms of Trade'].min(), df['Aus Terms of Trade'].max())
assests_to_income = st.sidebar.slider('Assests to Disposable Income', df['Aus Total Assets to Disposable Income'].min(), df['Aus Total Assets to Disposable Income'].max())
underemployment_slider = st.sidebar.slider('Underemployment', df['Aust Underemployment rate'].min(), df['Aust Underemployment rate'].max())
real_gdp_slider = st.sidebar.slider('Real GDP', df['Aus Real GDP mom%'].min(), df['Aus Real GDP mom%'].max())
nominal_gdp_slider = st.sidebar.slider('Nominal GDP', df['Aus Nominal GDP mom%'].min(), df['Aus Nominal GDP mom%'].max())


Energy_predicted_return = Energy_intercept + Energy_unemployment * unemployment_slider + Energy_inflation * inflation_slider + Energy_terms_of_trade * terms_of_trade_slider + Energy_assests_to_income * assests_to_income + Energy_underemployment * underemployment_slider + Energy_real_GDP * real_gdp_slider + Energy_nominal_GDP * nominal_gdp_slider


Material_predicted_return = Materials_intercept + Materials_unemployment*unemployment_slider + Materials_inflation*inflation_slider + Materials_terms_of_trade*terms_of_trade_slider + Materials_assests_to_income*assests_to_income+Materials_underemployment*underemployment_slider


Telecommunication_predicted_return = Telecommunication_intercept + Telecommunication_unemployment * unemployment_slider + Telecommunication_inflation * inflation_slider + Telecommunication_terms_of_trade * terms_of_trade_slider + Telecommunication_assests_to_income * assests_to_income + Telecommunication_underemployment * underemployment_slider


st.write('Predicted Returns:')
st.write(f"Materials: {Material_predicted_return}")
st.write(f"Telecommunication: {Telecommunication_predicted_return}")
st.write(f"Energy: {Energy_predicted_return}")


#Error Scores

# Calculate error scores for Energy model
mse_energy = mean_squared_error(y_SPASX_Energy_test, Energy_predictions)
r2_energy = r2_score(y_SPASX_Energy_test, Energy_predictions)

# Calculate error scores for Materials model
mse_materials = mean_squared_error(y_SPASX_Materials_test, Materials_predictions)
r2_materials = r2_score(y_SPASX_Materials_test, Materials_predictions)

# Calculate error scores for Telecommunication model
mse_telecommunication = mean_squared_error(y_SPASX_Telecommunication_test, Telecommunication_predictions)
r2_telecommunication = r2_score(y_SPASX_Telecommunication_test, Telecommunication_predictions)

# Display error scores in a table
error_data = {
    'Model': ['Energy', 'Materials', 'Telecommunication'],
    'Mean Squared Error': [mse_energy, mse_materials, mse_telecommunication],
    'R-squared': [r2_energy, r2_materials, r2_telecommunication]
}

error_df = pd.DataFrame(error_data)

st.write('Error Scores:')
st.table(error_df)
