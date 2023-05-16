import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# importing the data
dataset = pd.read_excel('AB_NYC_2019.xlsx')

# to look how the dataset looks
dataset

# checking for null values
dataset.info()

# plot outliers in the data
dataset.boxplot(by='Area', column=['Price/night($)'], grid=True, return_type=None)
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.show()

# for interactive insight will use plotly library
fig = px.box(dataset, x='Area', y='Price/night($)')
fig.show()

# Now remove the outliers as follows
df_new = dataset.drop(dataset[(dataset['Area'] == 'Brooklyn') & (dataset['Price/night($)'] > 285)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Manhattan') & (df_new['Price/night($)'] > 350)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Queens') & (df_new['Price/night($)'] > 140)].index)
df_new = df_new.drop(df_new[(df_new['Area'] == 'Staten Island') & (df_new['Price/night($)'] > 70)].index)

# check if there is still outliers present
fig2 = px.box(df_new, x='Area', y='Price/night($)')
fig2.show()

# compare means, standard dev and coefficient of variation
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
