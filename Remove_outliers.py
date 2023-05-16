import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# importing the data
dataset = pd.read_excel('AB_NYC_2019.xlsx')

dataset

# checking for null values
dataset.info() 

dataset.boxplot(by='Area', column=['Price/night($)'], grid=True, return_type=None)
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.show()

# for interactive insight I'll use plotly library
fig = px.box(dataset, x='Area', y='Price/night($)')
fig.show()

# Now remove the outliers as following
df_new = dataset.drop(dataset[(dataset['Area'] == 'Brooklyn') & (dataset['Price/night($)'] > 285)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Manhattan') & (df_new['Price/night($)'] > 350)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Queens') & (df_new['Price/night($)'] > 140)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Staten Island') & (df_new['Price/night($)'] > 70)].index)

# plot again the update
fig2 = px.box(df_new, x='Area', y='Price/night($)')
fig2.show()

# For calculattion of mean, std deviation and coefficient of variation
# mean
def mean_price(location):
    city = df_new[df_new['Area'] == location]
    result = np.mean(city['Price/night($)'])
    return result


# standard deviation
def std_devi(location):
    city = df_new[df_new['Area'] == location]
    result = np.std(city['Price/night($)'], ddof=1)
    return result

    
# coefficient of variation
def coeff_var(location):
    city = df_new[df_new['Area'] == location]
    s = std_devi(location)
    m = mean_price(location)
    result = (s / m) * 100 
    return result
  
# Compare
print("mean price in Bronx is ", mean_price('Bronx'))
print("mean price in Brooklyn is ", mean_price('Brooklyn'))
print("mean price in Manhattan is ", mean_price('Manhattan'))
print("mean price in Queens is ", mean_price('Queens'))
print("mean price in Staten Island is ", mean_price('Staten Island'))
print("standard deviation of price is ", std_devi('Bronx'))
print("standard deviation of price is ", std_devi('Brooklyn'))
print("standard deviation of price is ", std_devi('Manhattan'))
print("standard deviation of price is ", std_devi('Queens'))
print("standard deviation of price is ", std_devi('Staten Island'))
print("coefficient of variation of price is ", coeff_var('Bronx'))
print("coefficient of variation of price is ", coeff_var('Brooklyn'))
print("coefficient of variation of price is ", coeff_var('Manhattan'))
print("coefficient of variation of price is ", coeff_var('Queens'))
print("coefficient of variation of price is ", coeff_var('Staten Island'))
