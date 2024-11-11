#import packages
import pandas as pd
import numpy as np
import os
from fastai.tabular.all import  *
from matplotlib.pylab import rcParams
import quandl
from pathlib import Path

# Path of the output data folder
os.makedirs(str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data/processed", exist_ok=True)
prepared_folder_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data/processed"
input_dataset_folder = str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data"

#setting figure size
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

sp_dataset = input_dataset_folder + "/stock_dataset.csv"

#read the file
df = quandl.get("NSE/TATAGLOBAL", authtoken="WuwsYYzjWT_ogDGgRpSS")
df.to_csv(sp_dataset, index=True)
df.to_pickle(input_dataset_folder + "/stock_dataset.pkl")
print("Writing file {} to disk.".format(sp_dataset))

#print the head
# print(df.head())


#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

#setting index as date values
df['Date'] = pd.to_datetime(df.index.values,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
new_data['Date'] = data['Date'].values
new_data['Close'] = data['Close'].values

# print(new_data.head())


add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data.loc[i,['mon_fri']] = 1
        
    else:
        new_data.loc[i,['mon_fri']] = 0

#split into train and validation
train = new_data.iloc[:987]
valid = new_data.iloc[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']



X_train_path = prepared_folder_path + "/X_train.csv"
y_train_path = prepared_folder_path + "/y_train.csv"
X_valid_path = prepared_folder_path + "/X_valid.csv"
y_valid_path = prepared_folder_path + "/y_valid.csv"

x_train.to_csv(X_train_path, index=False)
print("Writing file {} to disk.".format(X_train_path))

y_train.to_csv(y_train_path, index=False)
print("Writing file {} to disk.".format(y_train_path))

x_valid.to_csv(X_valid_path, index=False)
print("Writing file {} to disk.".format(X_valid_path))

y_valid.to_csv(y_valid_path, index=False)
print("Writing file {} to disk.".format(y_valid_path))
