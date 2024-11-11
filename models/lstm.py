#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import os
import sys
paths = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, paths)


# Data preparation parameters
with open (paths+'/params.yaml', "r") as param_files:
    try:
        params_ = yaml.safe_load(param_files)
        params_ = params_["evaluate"]
    except yaml.YAMLError as error:
        print(error)



def Lstm(datasets):

    df = pd.DataFrame(pd.read_csv("{0}".format(datasets)))
    os.remove("dataset.csv")

    #creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    new_data['Date'] = data['Date'].values
    new_data['Close'] = data['Close'].values

    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    #creating train and test sets
    dataset = new_data.values

    train = dataset[0:987,:]
    valid = dataset[987:,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #implement Sequential Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=params_["epochs"], batch_size=params_["batch_size"], verbose=params_["verbose"])


    #predicting values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    # LSTM Accuracy
    rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))

    train = new_data[:987]
    valid = new_data[987:]
    
    val = []
    datas = valid.to_dict(orient='records')
    for i in range(0, len(datas)):
        datas[i]['Predictions'] = closing_price[i][0]
        val.append(datas[i])


    valid_pred = pd.DataFrame(val)
    val[:] = []


    return rms, valid_pred

