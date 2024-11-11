#implement linear regression
from sklearn.linear_model import LinearRegression
#importing libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from train import new_data, valid, train, df
# %run train.ipynb import new_data, valid, train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# path of Predicted output folder
os.makedirs(str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data/predicted", exist_ok=True)
predicted_folder_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data/predicted"
out_path = predicted_folder_path+"/predicted_value.csv"

# Read yaml file 
params = str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/params.yaml"

# Data preparation parameters
with open (params, "r") as param_files:
    try:
        params_ = yaml.safe_load(param_files)
        params_ = params_["evaluate"]
    except yaml.YAMLError as error:
        print(error)



# Path of the prepared data folder
input_folder_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())+"/data/processed"

# Read training dataset
x_train = pd.read_csv(input_folder_path + "/X_train.csv")
y_train = pd.read_csv(input_folder_path + "/y_train.csv")
# Read validation dataset
x_valid = pd.read_csv(input_folder_path + "/X_valid.csv")
y_valid = pd.read_csv(input_folder_path + "/y_valid.csv")



model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_train,y_train)

###Result
#make predictions and find the rmse
preds = model.predict(x_valid)
LR_rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))


#plot
valids = []
data = valid.to_dict(orient='records')
for i in range(0, len(data)):
    data[i]['Predictions'] = preds[i]
    valids.append(data[i])
validing = pd.DataFrame(valids)

validing.index = new_data[987:].index
train.index = new_data[:987].index

    
plt.plot(train['Close'])
plt.plot(validing[['Close', 'Predictions']])


# k-Nearest Neighbours Regression

### Implementation :

# We will use the same dataset (train and validation)
# scaling data

scaler = MinMaxScaler(feature_range=(0, 1)) 

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

###Result
#rmse
KNN_rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))



#plot
validate = []
datas = valid.to_dict(orient='records')
for i in range(0, len(datas)):
    datas[i]['Predictions'] = preds[i]
    validate.append(datas[i])
    
valid_knn = pd.DataFrame(validate)

valid_knn.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(valid_knn[['Close', 'Predictions']])
plt.plot(train['Close'])



# Long Short Term Memory (LSTM) - RNN

### Implementation :

# For now, let us implement LSTM and check it’s performance on our particular data.

#creating dataframe
new_df = df.sort_index(ascending=True, axis=0)
new_datas = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
new_datas['Date'] = new_df['Date'].values
new_datas['Close'] = new_df['Close'].values

#setting index
new_datas.index = new_datas.Date
new_datas.drop('Date', axis=1, inplace=True)

#creating train and test sets
datasets = new_datas.values

new_train = datasets[0:987,:]
nu_valid = datasets[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(datasets)

x_trains, y_trains = [], []
for i in range(60,len(train)):
    x_trains.append(scaled_data[i-60:i,0])
    y_trains.append(scaled_data[i,0])
x_trains, y_trains = np.array(x_trains), np.array(y_trains)

x_trains = np.reshape(x_trains, (x_trains.shape[0],x_trains.shape[1],1))

# create and fit the LSTM network
models = Sequential()
models.add(LSTM(units=50, return_sequences=True, input_shape=(x_trains.shape[1],1)))
models.add(LSTM(units=50))
models.add(Dense(1))

models.compile(loss='mean_squared_error', optimizer='adam')
models.fit(x_trains, y_trains, epochs=params_["epochs"], batch_size=params_["batch_size"], verbose=params_["verbose"])

#predicting 246 values, using past 60 from the train data
inputs = new_datas[len(new_datas) - len(nu_valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

new_test = []
for i in range(60,inputs.shape[0]):
    new_test.append(inputs[i-60:i,0])
new_tests = np.array(new_test)

new_tests = np.reshape(new_tests, (new_tests.shape[0],new_tests.shape[1],1))
closing_price = models.predict(new_tests)
closing_price = scaler.inverse_transform(closing_price)

###Result
rms=np.sqrt(np.mean(np.power((nu_valid-closing_price),2)))

#for plotting
new_train = new_datas[:987]
new_valid = new_datas[987:]

val = []
ls_datas = new_valid.to_dict(orient='records')
for i in range(0, len(ls_datas)):
    ls_datas[i]['Predictions'] = closing_price[i][0]
    val.append(ls_datas[i])
    
new_validate = pd.DataFrame(val)
new_validate.to_csv(out_path, index=False)
val[:] = []

plt.plot(new_train['Close'])
plt.plot(new_validate[['Close','Predictions']])

