import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

st.set_page_config()


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
X_1 = df[x_variables]


y_SPASX_Energy = df['S&P/ASX 200 - Energy Returns'].dropna()
y_SPASX_Materials = df['S&P/ASX 200 - Materials Returns'].dropna()
y_SPASX_Telecommunication = df['S&P/ASX 200 - Telecommunication Returns'].dropna()        
               
               
training_begin = X_1.index.min()
training_end = X_1.index.min() + pd.DateOffset(months=180)

X_train, X_test = X_1[training_begin:training_end], X_1[training_end:]
X_train  = X_train.drop([X_train.index[0]])

Scaler = StandardScaler()
X_scaler = Scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)



#setting y variables
y_SPASX_Energy_train, y_SPASX_Energy_test = y_SPASX_Energy[training_begin:training_end], y_SPASX_Energy[training_end:]
y_SPASX_Materials_train, y_SPASX_Materials_test = y_SPASX_Materials[training_begin:training_end], y_SPASX_Materials[training_end:]
y_SPASX_Telecommunication_train, y_SPASX_Telecommunication_test = y_SPASX_Telecommunication[training_begin:training_end], y_SPASX_Telecommunication[training_end:]


#model creating
model_Energy = LinearRegression().fit(X_train_scaled, y_SPASX_Energy_train)
model_Materials = LinearRegression().fit(X_train_scaled, y_SPASX_Materials_train)
model_Telecommunication = LinearRegression().fit(X_train_scaled, y_SPASX_Telecommunication_train)
               
#model predictions               
Energy_predictions = model_Energy.predict(X_test_scaled)
Materials_predictions = model_Materials.predict(X_test_scaled)
Telecommunication_predictions = model_Telecommunication.predict(X_test_scaled)

#Printing Coeffecients Value

st.title('Coeffecients for each asset class')


#Getting Coeffecients for energy model

Energy_intercept = model_Energy.intercept_

Energy_variables = ['unemployment','inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Energy_coefficients = model_Energy.coef_

for i, variable in enumerate(Energy_variables):
    globals()[f"Energy_{variable}"] = Energy_coefficients[i-1]

#printing these coeffs in a table

energy_coeff_dict = {
    'Variable': ['Intercept'] + Energy_variables,  
    'Coefficient': [Energy_intercept] + list(Energy_coefficients)
}

energy_coeff_df = pd.DataFrame(energy_coeff_dict)



    
#Getting Coeffecients for materials model

Materials_intercept = model_Materials.intercept_

Materials_variables = ['unemployment','inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Materials_coefficients = model_Materials.coef_

for i, variable in enumerate(Materials_variables):
    globals()[f"Materials_{variable}"] = Materials_coefficients[i]

    
#Printing these coeffs in a table

materials_coeff_dict = {
    'Variable': ['Intercept'] + Materials_variables,  
    'Coefficient': [Materials_intercept] + list(Materials_coefficients)
}

materials_coeff_df = pd.DataFrame(materials_coeff_dict)




#Getting Coeffecients for Telecommunications model

Telecommunication_intercept = model_Telecommunication.intercept_

Telecommunication_variables = ['unemployment', 'inflation', 'terms_of_trade', 'assests_to_income', 'underemployment', 'real_GDP', 'nominal_GDP']
Telecommunication_coefficients = model_Telecommunication.coef_

for i, variable in enumerate(Telecommunication_variables):
    globals()[f"Telecommunication_{variable}"] = Telecommunication_coefficients[i]

    
    
#Printing these coeffs in a table


telecommunication_coeff_dict = {
    'Variable': ['Intercept'] + Telecommunication_variables,
    'Coefficient': [Telecommunication_intercept] + list(Telecommunication_coefficients)
}

telecommunication_coeff_df = pd.DataFrame(telecommunication_coeff_dict)


    
col1, col2, col3 = st.columns(3)

# Display Energy Model Coefficients
with col1:
    st.write('Energy Model Coefficients:')
    st.write(energy_coeff_df)

# Display Materials Model Coefficients
with col2:
    st.write('Materials Model Coefficients:')
    st.write(materials_coeff_df)

# Display Telecommunication Model Coefficients
with col3:
    st.write('Telecommunication Model Coefficients:')
    st.write(telecommunication_coeff_df)
    
    
# Streamlit app

st.title('Effect of Variables on Returns')
st.sidebar.header('Adjust Variables')

#Streamlit Slider Creation
unemployment_slider = st.sidebar.slider('Unemployment', df['Aus Employed - Unemployment Rate'].min(), df['Aus Employed - Unemployment Rate'].max())
inflation_slider = st.sidebar.slider('Inflation', df['Aus CPI All Groups'].min(), df['Aus CPI All Groups'].max())
terms_of_trade_slider = st.sidebar.slider('Terms of trade', df['Aus Terms of Trade'].min(), df['Aus Terms of Trade'].max())
assests_to_income = st.sidebar.slider('Assests to Disposable Income', df['Aus Total Assets to Disposable Income'].min(), df['Aus Total Assets to Disposable Income'].max())
underemployment_slider = st.sidebar.slider('Underemployment', df['Aust Underemployment rate'].min(), df['Aust Underemployment rate'].max())
real_gdp_slider = st.sidebar.slider('Real GDP', df['Aus Real GDP mom%'].min(), df['Aus Real GDP mom%'].max())
nominal_gdp_slider = st.sidebar.slider('Nominal GDP', df['Aus Nominal GDP mom%'].min(), df['Aus Nominal GDP mom%'].max())



Energy_predicted_return = Energy_intercept + Energy_unemployment * unemployment_slider + Energy_inflation * inflation_slider + Energy_terms_of_trade * terms_of_trade_slider + Energy_assests_to_income * assests_to_income + Energy_underemployment * underemployment_slider + Energy_real_GDP * real_gdp_slider + Energy_nominal_GDP * nominal_gdp_slider


Material_predicted_return = Materials_intercept + Materials_unemployment*unemployment_slider + Materials_inflation*inflation_slider + Materials_terms_of_trade*terms_of_trade_slider + Materials_assests_to_income*assests_to_income+Materials_underemployment*underemployment_slider + Materials_real_GDP * real_gdp_slider + Materials_nominal_GDP * nominal_gdp_slider


Telecommunication_predicted_return = Telecommunication_intercept + Telecommunication_unemployment * unemployment_slider + Telecommunication_inflation * inflation_slider + Telecommunication_terms_of_trade * terms_of_trade_slider + Telecommunication_assests_to_income * assests_to_income + Telecommunication_underemployment * underemployment_slider+Telecommunication_real_GDP * real_gdp_slider + Telecommunication_nominal_GDP * nominal_gdp_slider



# Display predicted returns for each sector
st.subheader('Predicted Returns Over a Month Period')
st.write(f"Materials: {Material_predicted_return:.2f}%")
st.write(f"Telecommunication: {Telecommunication_predicted_return:.2f}%")
st.write(f"Energy: {Energy_predicted_return:.2f}%")

# Streamlit sidebar sliders for investment amounts
st.sidebar.header("Your investment portfolio")
investment_energy = st.sidebar.number_input('Investment in Energy', min_value=0)
investment_materials = st.sidebar.number_input('Investment in Materials', min_value=0)
investment_telecom = st.sidebar.number_input('Investment in Telecommunication', min_value=0)

# Calculating returns considering percentage returns
return_energy = investment_energy * (1 + (Energy_predicted_return / 100))
return_materials = investment_materials * (1 + (Material_predicted_return / 100))
return_telecom = investment_telecom * (1 + (Telecommunication_predicted_return / 100))

# Total return on investment
total_return = return_energy + return_materials + return_telecom


# Display investment returns
st.subheader('Investment Returns Over a Month Period')
st.write(f"Energy Return: {return_energy:.2f}")
st.write(f"Materials Return: {return_materials:.2f}")
st.write(f"Telecommunication Return: {return_telecom:.2f}")
st.write(f"Total Return on Investment: {total_return:.2f}")


# Create a bar graph to visualize initial investment and returns
labels = ['Energy', 'Materials', 'Telecommunication']
initial_investment = [investment_energy, investment_materials, investment_telecom]
predicted_returns = [return_energy, return_materials, return_telecom]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, initial_investment, width, label='Initial Investment')
rects2 = ax.bar([i + width for i in x], predicted_returns, width, label='Predicted Returns Month on Month')

ax.set_xlabel('Sector')
ax.set_ylabel('Amount')
ax.set_title('Initial Investment vs Predicted Returns')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(labels)
ax.legend()

st.pyplot(fig)



#Error Scores

st.title(" Error Scores")
st.subheader("Poor Error Results is A Cause of Cocern")

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

st.title("Methods Used to Try and Improve This")
st.subheader("Method 1: Use Random Forest For Energy Model To Determine Important Features")

rf_energy = RandomForestRegressor(n_estimators=500, random_state=78)
rf_energy.fit(X_train, y_SPASX_Energy_train)

predictions_energy = rf_energy.predict(X_test)

#feature importance plot

feature_importance = rf_energy.feature_importances_

feature_names = X_1.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Energy Returns')
plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
plt.show()
st.pyplot(plt)

st.subheader("Results using this")



types = ['Unemployment','CPI - Inflation','Terms of Trade','Total Assests to Disposable Income','Underemployment','Real GDP','Nominal GDP']
x_variables = ['Aus CPI All Groups','Aust Underemployment rate','Aus Terms of Trade']
X = df[x_variables]


y_SPASX_Energy = df['S&P/ASX 200 - Energy Returns'].dropna()
y_SPASX_Materials = df['S&P/ASX 200 - Materials Returns'].dropna()
y_SPASX_Telecommunication = df['S&P/ASX 200 - Telecommunication Returns'].dropna()        
               
               
training_begin = X.index.min() 
training_end = X.index.min() + pd.DateOffset(months=180)

X_train, X_test = X[training_begin:training_end], X[training_end:]
X_train  = X_train.drop([X_train.index[0]])


scaler = StandardScaler()

X_scaler = scaler.fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
 

#setting y variables
y_SPASX_Energy_train, y_SPASX_Energy_test = y_SPASX_Energy[training_begin:training_end], y_SPASX_Energy[training_end:]
y_SPASX_Materials_train, y_SPASX_Materials_test = y_SPASX_Materials[training_begin:training_end], y_SPASX_Materials[training_end:]
y_SPASX_Telecommunication_train, y_SPASX_Telecommunication_test = y_SPASX_Telecommunication[training_begin:training_end], y_SPASX_Telecommunication[training_end:]


#model creating
model_Energy = LinearRegression().fit(X_train_scaled, y_SPASX_Energy_train)
model_Materials = LinearRegression().fit(X_train_scaled, y_SPASX_Materials_train)
model_Telecommunication = LinearRegression().fit(X_train_scaled, y_SPASX_Telecommunication_train)
               
#model predictions               
Energy_predictions = model_Energy.predict(X_test_scaled)
Materials_predictions = model_Materials.predict(X_test_scaled)
Telecommunication_predictions = model_Telecommunication.predict(X_test_scaled)

r2_energy = r2_score(y_SPASX_Energy_test, Energy_predictions)
st.write("The R^2 for the energy model only using the three main varibables identified from the Random Forest Regressor")
st.write(f"There was minimal improvment in the R^2 as it became {r2_energy} which was only a very slight improvment")



st.subheader("Method 2: Using Correlation Matrix")

# Assuming 'df' is your DataFrame containing the data
correlation_matrix = X_1.corr()

# Display the correlation matrix
st.table(correlation_matrix)



# setting the x-variables
types = ['Unemployment','CPI - Inflation','Terms of Trade','Total Assests to Disposable Income','Underemployment','Real GDP','Nominal GDP']
x_variables = ['Aus Employed - Unemployment Rate', 'Aus CPI All Groups','Aus Terms of Trade','Aus Total Assets to Disposable Income','Aus Real GDP mom%']
X = df[x_variables]


y_SPASX_Energy = df['S&P/ASX 200 - Energy Returns'].dropna()
y_SPASX_Materials = df['S&P/ASX 200 - Materials Returns'].dropna()
y_SPASX_Telecommunication = df['S&P/ASX 200 - Telecommunication Returns'].dropna()        
               
               
training_begin = X.index.min()
training_end = X.index.min() + pd.DateOffset(months=180)

X_train, X_test = X[training_begin:training_end], X[training_end:]
X_train  = X_train.drop([X_train.index[0]])

Scaler = StandardScaler()
X_scaler = Scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)



#setting y variables
y_SPASX_Energy_train, y_SPASX_Energy_test = y_SPASX_Energy[training_begin:training_end], y_SPASX_Energy[training_end:]
y_SPASX_Materials_train, y_SPASX_Materials_test = y_SPASX_Materials[training_begin:training_end], y_SPASX_Materials[training_end:]
y_SPASX_Telecommunication_train, y_SPASX_Telecommunication_test = y_SPASX_Telecommunication[training_begin:training_end], y_SPASX_Telecommunication[training_end:]


#model creating
model_Energy = LinearRegression().fit(X_train_scaled, y_SPASX_Energy_train)
model_Materials = LinearRegression().fit(X_train_scaled, y_SPASX_Materials_train)
model_Telecommunication = LinearRegression().fit(X_train_scaled, y_SPASX_Telecommunication_train)
               
#model predictions               
Energy_predictions = model_Energy.predict(X_test_scaled)
Materials_predictions = model_Materials.predict(X_test_scaled)
Telecommunication_predictions = model_Telecommunication.predict(X_test_scaled)

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
