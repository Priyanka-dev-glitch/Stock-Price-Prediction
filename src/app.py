#importing libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression
from fastai.tabular.all import  *
from http import HTTPStatus
from keras.models import Sequential
from keras.layers import Dense, LSTM
import quandl
from pathlib import Path
import yaml
from prometheus_fastapi_instrumentator import Instrumentator
from models.linear_regression import Linear_Regression 
from models.lstm import Lstm
from models.knn import Knn
from fastapi import FastAPI, File, UploadFile, Query
import shutil
from src.schema import Model_List
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



res_list = []



def prepare_dataset():
    # Read training dataset
    dataset = quandl.get("NSE/TATAGLOBAL", authtoken="WuwsYYzjWT_ogDGgRpSS")


    dataset['Date'] = pd.to_datetime(dataset.index.values,format='%Y-%m-%d')
    dataset.index = dataset['Date']

    #sorting
    data = dataset.sort_index(ascending=True, axis=0)

    #creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(dataset)),columns=['Date', 'Close'])
    new_data['Date'] = data['Date'].values
    new_data['Close'] = data['Close'].values

    new_df = pd.DataFrame(index=range(0,len(dataset)),columns=['Date', 'Close'])
    new_df['Date'] = data['Date'].values
    new_df['Close'] = data['Close'].values




    add_datepart(new_data, 'Date')
    new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

    new_data['mon_fri'] = 0
    for i in range(0,len(new_data)):
        if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
            new_data.loc[i,['mon_fri']] = 1
            
        else:
            new_data.loc[i,['mon_fri']] = 0
    
    #split into train and validation
    split_train = new_data.iloc[:987]
    split_valid = new_data.iloc[987:]

    x_train = split_train.drop('Close', axis=1)
    y_train = split_train['Close']
    x_valid = split_valid.drop('Close', axis=1)
    y_valid = split_valid['Close']
    return {"train": {'X': x_train, 'Y': y_train }, "valid": {'X': x_valid, 'Y': y_valid}, "new_df": new_df}


app = FastAPI(
    title="Stock Prediction Using FastAPI"
    )


origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    expose_headers=["*"],
    allow_headers=origins
)
@app.on_event("startup")
async def instrumentator():
    Instrumentator().instrument(app).expose(app)



@app.get("/", tags=["List Endpoints"])
def Base_url(request: Request):
    url_list = [
        {"path": route.path, "name": route.name} for route in request.app.routes
    ]
    return url_list


 
@app.get("/listmodels", tags=["List Models"])
def List_Models():

    fetch_dataset = prepare_dataset()

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(fetch_dataset['train']['X'], fetch_dataset['train']['Y'])
    # lr_score = lr_model.score(fetch_dataset['train']['X'], fetch_dataset['train']['Y'])

    lr_preds = lr_model.predict(fetch_dataset['valid']['X'])
    
    #Linear Regression rms
    lr_rms = np.sqrt(np.mean(np.power((np.array(fetch_dataset['valid']['Y'])-np.array(lr_preds)),2)))


    res_list.append({
        "Model_Type": "Linear Regression",
        "Model_Accuracy": lr_rms
    })

       # Long Short Term Memory(LSTM)
    # setting index
    fetch_dataset['new_df'].index = fetch_dataset['new_df'].Date
    fetch_dataset['new_df'].drop('Date', axis=1, inplace=True)

    #creating train and test sets
    dataset = fetch_dataset['new_df'].values

    train = dataset[0:987,:]
    valid = dataset[987:,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

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
    inputs = fetch_dataset['new_df'][len(fetch_dataset['new_df']) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    new_test = []
    for i in range(60,inputs.shape[0]):
        new_test.append(inputs[i-60:i,0])
    new_tests = np.array(new_test)

    new_tests = np.reshape(new_tests, (new_tests.shape[0],new_tests.shape[1],1))
    closing_price = models.predict(new_tests)
    closing_price = scaler.inverse_transform(closing_price)

    # LSTM rmse
    lstm_rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))

    res_list.append({
        "Model_Type": "Long Short Term Memory (LSTM)",
        "Model_Accuracy": lstm_rms
    })

    # K-Nearest Neighbours Regression
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    #scaling data
    x_train_scaled = scaler.fit_transform(fetch_dataset['train']['X'])
    X_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(fetch_dataset['valid']['X'])
    X_valid = pd.DataFrame(x_valid_scaled)

    #using gridsearch to find the best parameter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    knn_model = GridSearchCV(knn, params, cv=5)

    #fit the model and make predictions
    knn_model.fit(X_train, fetch_dataset['train']['Y'])
    # knn_score = knn_model.score(X_train, fetch_dataset['train']['Y'])

    preds = knn_model.predict(X_valid)

    #K-NN rms
    knn_rms = np.sqrt(np.mean(np.power((np.array(fetch_dataset['valid']['Y'])-np.array(preds)),2)))

    res_list.append({
        "Model_Type": "K-Nearest Neighbour Regression",
        "Model_Accuracy": knn_rms 
    })
    
    
    available_models = [
        {
            "Model Type": list["Model_Type"],
            "Accuracy": list["Model_Accuracy"],
        }
        for list in res_list
    ]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }
    res_list[:] = []

    return response


@app.post("/models", tags=["Prediction"])
def predict_data(types: Model_List =  Query(""), file: UploadFile = File(...)):
    with open("dataset.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if types == "":
        os.remove("dataset.csv")
        response = {
            "message": "Choose a Model to Train",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
        return response
    
    else:

        if types.value == "Linear Regression Model":
            results = Linear_Regression("dataset.csv")
            final_prediction =[

                {
                    "Close": iter["Close"],
                    "Predictions": iter["Predictions"] 
                }
                for iter in results[1].to_dict(orient='records')
            ]

            response = {
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
                "Model Type": types.value,
                "Accuracy": results[0],
                "data": final_prediction
            }
            return response


        elif types.value == "LSTM Model":
            results = Lstm("dataset.csv")
            final_prediction =[

                {
                    "Close": iter["Close"],
                    "Predictions": iter["Predictions"] 
                }
                for iter in results[1].to_dict(orient='records')
            ]

            response = {
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
                "Model Type": types.value,
                "Accuracy": results[0],
                "data": final_prediction
            }
            return response

        elif types.value == "k-Nearest Neighbour Model":
            results = Knn("dataset.csv")
            final_prediction =[

                {
                    "Close": iter["Close"],
                    "Predictions": iter["Predictions"] 
                }
                for iter in results[1].to_dict(orient='records')
            ]

            response = {
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
                "Model Type": types.value,
                "Accuracy": results[0],
                "data": final_prediction
            }
            return response







