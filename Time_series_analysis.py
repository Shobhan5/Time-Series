#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install download')


# In[2]:


from __future__ import division, print_function, unicode_literals

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from download import download
mpl.rcParams['figure.figsize']=(8,6)
mpl.rcParams['axes.grid']=False


# In[3]:


path=download('https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip','/tmp/aq',kind="zip",replace=True)


# In[4]:


df=pd.read_csv('/tmp/aq/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv',encoding='ISO-8859-1')


# In[5]:


get_ipython().system('ls -alrt /tmp/aq/PRSA_Data_20130301-20170228')


# In[6]:


import os

# Specify the directory path where the data files are stored
data_directory = '/tmp/aq/PRSA_Data_20130301-20170228'

# List all files in the directory
file_list = os.listdir(data_directory)

# Print the file names
for filename in file_list:
    print(filename)


# In[7]:


df


# In[8]:


df.info()


# In[9]:


def convert_to_date(x):
    return datetime.strptime(x,'%Y %m %d %H')


# In[10]:


aq_df=pd.read_csv('/tmp/aq/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv',parse_dates=[['year','month','day','hour']],date_parser=convert_to_date,keep_date_col=True)


# In[11]:


aq_df.info()


# In[12]:


aq_df.head()


# In[13]:


aq_df['month']=pd.to_numeric(aq_df['month'])


# In[14]:


print("Rows:",aq_df.shape[0])


# In[15]:


print("Columns:",aq_df.shape[1])


# In[16]:


print("\nFeatures:\n",aq_df.columns.tolist())


# In[17]:


print("\nMissingvalue:",aq_df.isnull().any())


# In[18]:


print("\nUniquevalues:\n",aq_df.nunique())


# In[19]:


aq_df.describe()


# In[20]:


aq_df_non_indexed=aq_df.copy()


# In[21]:


aq_df=aq_df.set_index('year_month_day_hour')


# In[22]:


aq_df.index


# In[23]:


aq_df.head()


# In[24]:


aq_df.loc['2013-03-01':'2013-03-05']


# In[25]:


aq_df.loc['2013':'2015']


# In[26]:


pm_data=aq_df['PM2.5']


# In[27]:


pm_data.head()


# In[28]:


pm_data.plot(grid=True)


# In[29]:


aq_df_2015=aq_df.loc['2015']
pm_data_2015=aq_df_2015['PM2.5']
pm_data_2015.plot(grid=True)


# In[30]:


aq_df_2016=aq_df.loc['2016']
pm_data_2016=aq_df_2016['PM2.5']
pm_data_2016.plot(grid=True)


# In[31]:


import plotly.express as px
fig= px.line(aq_df_non_indexed,x='year_month_day_hour',y='PM2.5',title='PM2.5 with slider')

fig.update_xaxes(rangeslider_visible=True)
fig.show()


# In[32]:


fig=px.line(aq_df_non_indexed,x='year_month_day_hour',y='PM2.5',title='PM2.5 with Slider')
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
    buttons=list([
        dict(count=1,label="1y",step="year",stepmode="backward"),
        dict(count=2,label="2y",step="year",stepmode="backward"),
        dict(count=3,label="3y",step="year",stepmode="backward"),
        dict(step="all")
    ])
    )
)
fig.show()


# In[39]:


aq_df


# In[45]:


df_2014= aq_df.loc['2014'].reset_index()
df_2015= aq_df.loc['2015'].reset_index()
df_2014['month_day_hour']=df_2014.apply(lambda x:str(x['month'])+"-"+x['day'],axis=1)
df_2015['month_day_hour']=df_2015.apply(lambda x:str(x['month'])+"-"+x['day'],axis=1)
plt.plot(df_2014['month_day_hour'],df_2014['PM2.5'])
plt.plot(df_2015['month_day_hour'],df_2015['PM2.5'])
plt.legend(['2014','2015'])
plt.xlabel('Month')
plt.ylabel('PM2.5')
plt.title("Air quality plot for the year 2014 and 2015")


# In[46]:


aq_df['2014':'2016'][['month','PM2.5']].groupby('month').describe()


# In[47]:


aq_df['2014':'2016'][['month','PM2.5','TEMP']].groupby('month').agg({'PM2.5':['max'],'TEMP':['min','max']})


# In[51]:


aq_df_2015=aq_df.loc['2015']
pm_data_2015=aq_df_2015[['PM2.5','TEMP']]
pm_data_2015.plot(subplots=True)


# In[52]:


aq_df[['PM2.5','TEMP']].hist() #its important to see the distribution of data


# In[53]:


aq_df[['TEMP']].plot(kind='density') #it is important to see if the data is bimodal or multimodal


# # By lag,we can see the randomness.Lag means basically autocorrelation.If it's centric then it means high correlation and if it's spread out then it means less correlation.

# In[54]:


pd.plotting.lag_plot(aq_df['TEMP'],lag=1)


# In[55]:


pd.plotting.lag_plot(aq_df['TEMP'],lag=24)


# In[56]:


pd.plotting.lag_plot(aq_df['TEMP'],lag=8640) #it means one year


# In[57]:


pd.plotting.lag_plot(aq_df['TEMP'],lag=4320) #4320 means 6 month so opposite i.e.,negative correlation


# In[58]:


pd.plotting.lag_plot(aq_df['TEMP'],lag=2150) #2150 hours mean just 3 months...so there is less correlation


# In[60]:


aq_df_2015=aq_df.loc['2015']
pm_data_2015=aq_df_2015[['PM2.5','TEMP','PRES']]
pm_data_2015.plot(subplots=True)


# In[61]:


multi_data=aq_df[['TEMP','PRES','DEWP','RAIN','PM2.5']]
multi_data.plot(subplots=True)


# In[62]:


multi_data=aq_df[['SO2','NO2','O3','CO','PM2.5']]
multi_data.plot(subplots=True)


# In[67]:


import matplotlib.pyplot as plt

# Assuming 'aq_df' is your DataFrame
aq_df.loc['2014':'2015', ['PM2.5', 'O3']].plot(figsize=(15, 8), linewidth=3, fontsize=15)
plt.xlabel('year_month_day_hour', fontsize=20)
plt.show()  # Display the plot


# In[68]:


aq_df_2015['PM2.5']


# In[69]:


aq_df_2015


# In[70]:


aq_df.isnull().values.any()


# In[71]:


aq_df.isnull().any()


# In[72]:


df.isnull().sum()


# In[73]:


g=sns.pairplot(aq_df[['SO2','NO2','O3','CO','PM2.5']])


# In[75]:


aq_corr=aq_df[['SO2','NO2','O3','CO','PM2.5']].corr(method='pearson')
aq_corr


# # Heatmap basically shows the correlation among the variables; Red denotes high correlation and blue means less correlation

# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'aq_corr' is your correlation matrix
plt.figure(figsize=(10, 10))
g = sns.heatmap(aq_corr, vmax=0.6, center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.5}, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# In[78]:


aq_df.groupby('wd').agg(median=('PM2.5','median'),mean=('PM2.5','mean'),max=('PM2.5','max'),min=('PM2.5','min')).reset_index()


# In[79]:


aq_df_na=aq_df.copy()


# In[80]:


aq_df_na=aq_df_na.dropna()


# In[81]:


pd.plotting.autocorrelation_plot(aq_df_na['2014':'2016']['TEMP'])#See the horizontal axis in hours so 5000 , 10000 are hours


# In[82]:


aq_df_na['TEMP'].resample("1m").mean()


# In[83]:


pd.plotting.autocorrelation_plot(aq_df_na['2014':'2016']['TEMP'].resample("1m").mean()) #The solid line shows the 90% confidence interval and dotted line shows 95% confidence interval


# In[84]:


pd.plotting.autocorrelation_plot(aq_df_na['2014':'2016']['PM2.5'].resample("1m").mean())


# # Handling missing data 

# In[85]:


aq_df.isnull().sum()


# In[87]:


aq_df.query('TEMP!=TEMP').count()


# In[88]:


aq_df[aq_df['PM2.5'].isnull()]


# In[89]:


aq_df.query('TEMP!=TEMP')


# In[90]:


aq_df[aq_df['PM2.5'].isnull()]


# In[91]:


aq_df['2015-02-21 10':'2015-02-21 20']


# In[93]:


aq_df_imp=aq_df['2015-02-21 10':'2015-02-21 23'][['TEMP']]
aq_df_imp


# # Technique to handle missing value: forward fill method (replacing the missing value the previous value as the values are linearly correlated)

# In[94]:


aq_df_imp['TEMP_FFILL']=aq_df_imp['TEMP'].fillna(method='ffill')
aq_df_imp


# # Backward fill method (Not used widely)

# In[95]:


aq_df_imp['TEMP_BFILL']=aq_df_imp['TEMP'].fillna(method='bfill')
aq_df_imp #backward filling is not generally used


# # Rolling method : mean of the previous two values / three values

# In[96]:


aq_df_imp['TEMP'].rolling(window=2,min_periods=1).mean()


# In[99]:


aq_df_imp['TEMP_ROllING']=aq_df_imp['TEMP'].rolling(window=2,min_periods=1).mean()
aq_df_imp


# In[100]:


aq_df.loc[aq_df_imp.index+pd.offsets.DateOffset(years=-1)]['TEMP']


# In[101]:


aq_df_imp=aq_df_imp.reset_index()


# In[102]:


aq_df_imp


# In[103]:


aq_df_imp['TEMP_PREVY']=aq_df_imp.apply(lambda x:aq_df.loc[x['year_month_day_hour']-pd.offsets.DateOffset(years=-1)]['TEMP']if pd.isna(x['TEMP'])else x['TEMP'],axis=1)


# In[104]:


aq_df_imp


# In[ ]:




