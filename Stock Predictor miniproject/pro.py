import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as df
from keras.models import Sequential
from keras.layers import Dense,LSTM
import streamlit as st
import yfinance as yf

import yfinance as yf
start_date = '2011-01-01'
end_date = '2019-01-01'
ticker = 'AAPL'
df = yf.download(ticker, start_date, end_date)
#data["Date"] = data.index
#data = data[["Date", "Open", "High","Low", "Close", "Adj Close", "Volume"]]
#data.reset_index(drop=True, inplace=True)
#print(data.head())

#Describe data
st.subheader('data from 2011-2019')
st.write(df.describe())

#visualizations
st.subheader('closing price vs time chart ')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#st.set_option('deprecation.showPyplotGlobalUse', False)

#visualizations 100days
st.subheader('closing price vs time chart with 100MA ')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)


#visualizations  200days
st.subheader('closing price vs time chart with 100MA & 200MA ')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(ma200,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

#Spliting data into traning and testing
data_traning=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#print(data_testing.shape)
#print(data_testing.shape)
#data_traning.head()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_traning_array=scaler.fit_transform(data_traning)
#data_traning_array

#splitting data into x_train and y_train
x_train =[]
y_train =[]

for i in range(100,data_traning_array.shape[0]):
    x_train.append(data_traning_array[i-100:i])
    y_train.append(data_traning_array[i,0])

x_train=np.array(x_train)
y_train=np.array(y_train)

#ML Model
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))


model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

model.save('keras_mode.h5')

#load my model

#model=Sequential('keras_mode.h5') 

#testing part
past_100_days=data_traning.tail(100)
final_df =past_100_days.append(data_testing , ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Predications vs Original')
fig2=plt.figure(figsize=(12,6))
st.subheader('Original-Blue, Predicted-Red')
plt.plot(y_test,'b')
plt.plot(y_predicted,'r')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)








