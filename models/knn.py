#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import numpy as np
from fastai.tabular.all import  *
import os


def Knn(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.DataFrame(pd.read_csv("{0}".format(dataset)))
    df.index = df['Date']
    os.remove("dataset.csv")


    #sorting
    data = df.sort_index(ascending=True, axis=0)

    #creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    new_data['Date'] = data['Date'].values
    new_data['Close'] = data['Close'].values

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

    #scaling data
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
    model.score(x_train,y_train)
    preds = model.predict(x_valid)

    #KNN rms
    rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

    
    validate = []
    datas = valid.to_dict(orient='records')
    for i in range(0, len(datas)):
        datas[i]['Predictions'] = preds[i]
        validate.append(datas[i])
        
    valid_pred = pd.DataFrame(validate)
    validate[:] = []

    # valid_pred = pd.DataFrame(preds, columns=["Predictions"])
    # valid_pred['Close'] = valid.values

    return rms, valid_pred


