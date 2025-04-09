#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np
import matplotlib.pylab as plt 
from matplotlib import pyplot 
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6


# In[11]:


data=pd.read_csv('AirPassengers.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)


# In[16]:


data = pd.read_csv('Airpassengers.csv', parse_dates=['Month'],index_col='Month')
print('\n Parsed Data:')
print(data.head())


# In[17]:


data.index


# In[18]:


#convert to timseries
ts=data['#Passengers']
ts.head(10)


# In[19]:


#indexing time series arrays 
ts['1949-04-01']


# In[20]:


#import datetime library and use datetime function 
from datetime import datetime
ts[datetime(1949,1,1)]


# In[21]:


#check sstationary
plt.plot(ts)


# In[30]:


# ho : TS is non stationary
# H1: TS in stationary
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Determing rolling statistics
    #rolmean pd.rolling_mean(timeseries, window=12)
    rolmean = pd.Series (timeseries).rolling(window=12).mean()
    #rolstd = pd.rolling_std(timeseries, window=12)
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black',label='Rolling Std')
    plt.legend(loc='best')
    plt.title('rolling Mean & Standard Deviation')
    plt.show(block=False)
    # Perform Dickey-Fuller test:
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observation used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput) #Fixed indentation here


# # MAKING  TS STATIONARY

# In[31]:


ts.head


# In[32]:


#log lgane k liye 
ts_log=np.log(ts)
#inverse krne ke liye 
ts_org=np.exp(ts_log)


# In[33]:


plt.plot(ts_log)


# In[34]:


test_stationarity(ts_log)


# In[ ]:





# In[35]:


#double log 
ts_log=np.log(ts)
ts_d_log=np.log(ts_log)
#inverse
ts_orig1 = np.exp(ts_d_log)
ts_orig=np.exp(ts_orig1)


# In[38]:


test_stationarity(ts_d_log)


# In[37]:


ts_log=np.log(ts)
#difference between the 2 consecutive log values
ts_log_diff= ts_log -ts_log.shift()
#inverse transformations
#add the shifted log values to the first log value to get the cumulative sum 
ts_log_cumsum=ts_log_diff.cumsum()
#add the cumultive sum to the first log value to get the restored log time series 
ts_log_restored = ts_log.iloc[0]+ts_log_cumsum
#take the exponential of the restore log
ts_orig=np.exp(ts_log_restored)


# In[41]:


#log+ms+diff
moving_avg=pd.Series(ts_log).rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg,color='red')
ts_log_moving_avg_diff=ts_log-moving_avg
ts_log_moving_avg_diff.head(5)
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head(5)


# In[42]:


#test stationary 
test_stationarity(ts_log_moving_avg_diff)


# In[ ]:


-


# In[ ]:





# In[ ]:




