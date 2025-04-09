#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[3]:


#load the shampoo sales dataset
data=pd.read_csv('shampoo.xls',usecols=[1],names=["sales"],header=0)


# In[4]:


data


# In[5]:


#covert time series format
data.index=pd.date_range(start="1901-01",periods=len(data),freq="M")
#converts the datsets into a time series format with a monthly frequency


# In[6]:


data


# In[7]:


#visulaize the data 
plt.figure(figsize=(10,5))
plt.plot(data,marker='o',linestyle=("-"))
plt.title("shampoo sales over time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()


# In[8]:


result = adfuller(data['sales'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
if result[1] <= 0.05:
    print('The series is stationary')
else:
    print('The series is non-stationary')


# In[9]:


#differencing  to make te data sationary 
data_diff=data.diff().dropna()


# In[10]:


#plot ACF and PACF 
fig,axes=plt.subplots(1,2,figsize=(12,5))
plot_acf(data_diff,lags=15, ax=axes[0])
plot_pacf(data_diff,lags=15,method='ywm', ax=axes[1])
axes[0].set_title("ACF Plot")
axes[1].set_title("PACF Plot")
plt.show()


# In[11]:


#fit arima model(p,d,q) =(5,1,0) based on acf and pacf analysis
model = ARIMA(data,order=(5,1,0))
model_fit=model.fit()


# In[12]:


#arima(5,1,0) is chosen based on acf and pacf analysis 


# In[13]:


#print model summary 
print(model_fit.summary())


# In[14]:


forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)
# forecasts 12 future time points(next12 mothns )
#forecast enerates predicted values baesd on the trained arima model


# In[ ]:




plt.figure(figsize=(10, 5))
plt.plot(data, label='Actual Sales')
plt.plot(pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='M')[1:], forecast, label='Forecast', color='red')
plt.title("Shampoo Sales Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()
# #WALK FORWARD ARIMA

# In[15]:


df=pd.read_csv('shampoo.xls', header=0,parse_dates=[0])


# In[16]:


data=df['Sales'].values


# In[17]:


train_size=int(len(data)*0.8)
train,test=data[:train_size],data[train_size:]


# In[19]:


# walk forward validation 
history = train.tolist() #Convert train set toa list for dynamic update
predictions= []
for t in test:
    #fit AR model
    model = ARIMA(data, order=(5,1,0)) #using last 7days for adtoregression
    model_fit = model.fit()
    
    # predict next value
    y_pred = model_fit.predict(start=len(history),end=len(history))[0]
    predictions.append(y_pred)
    # update history with actual observation
    history.append(t)


# In[21]:


#evaluate performance 
from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(test,predictions))
print(f'Walk Forward Validation RMSE:{rmse:.4f}')


# In[26]:


#plot actual vs predicted values 
plt.figure(figsize=(10,5))
plt.plot(test,label='Actual Temprature',marker='o')
plt.plot(predictions,label='Predicted Temprature',marker='x',linestyle='dotted')
plt.xlabel('Days')
plt.ylabel('Temprature')
plt.title('AR Model-Walk Forward Validation')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




