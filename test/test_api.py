import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from src.app import app
from src.schema import Model_List

@pytest.fixture
def source_file():
    fpath = str(Path(Path(__file__).parent.absolute()).parent.absolute())
    pd.read_pickle(fpath+"/data/stock_dataset.pkl").to_csv(fpath+"/data/stock_dataset.csv", index=True)
    fname = fpath+'/data/stock_dataset.csv'
    return {
        "file": ("upload_file", open(fname, "rb"), fname),
    }



with TestClient(app) as myclient:
    
    def test_swagger_ui():
        validate = myclient.get("/docs")
        assert validate is not None

    def test_list_url():
        validate = myclient.get("/")
        assert validate.status_code == 200 
        assert validate is not None
        assert next(item for item in validate.json() if item["name"] == "swagger_ui_html")
   
    def test_models_list():
        validate = myclient.get("/listmodels")
        jsonify = validate.json()
        assert validate.status_code == 200 
        assert jsonify["message"] == "OK"  
        assert type(jsonify['data']) == list  
        assert isinstance(jsonify['data'][0]["Model Type"], str)
        assert isinstance(jsonify['data'][0]["Accuracy"], float)
        assert jsonify['data'][0]["Accuracy"] < jsonify['data'][2]["Accuracy"]

    def test_model_linear_regression(source_file):
        validate = myclient.post("/models?types=Linear Regression Model", files=source_file)
        jsonify = validate.json()
        assert validate.status_code == 200
        assert jsonify['message'] == 'OK'
        assert len(jsonify['data']) != 0
        assert "Close" in jsonify['data'][0]
        assert "Predictions" in jsonify['data'][0]

    def test_model_knn(source_file):
        validate = myclient.post("/models?types=k-Nearest Neighbour Model", files=source_file)
        jsonify = validate.json()
        assert validate.status_code == 200
        assert jsonify['message'] == 'OK'
        assert len(jsonify['data']) != 0
        assert "Close" in jsonify['data'][0]
        assert "Predictions" in jsonify['data'][0]

    def test_model_lstm(source_file):
        validate = myclient.post("/models?types=LSTM Model", files=source_file)
        jsonify = validate.json()
        assert validate.status_code == 200
        assert jsonify['message'] == 'OK'
        assert len(jsonify['data']) != 0
        assert "Close" in jsonify['data'][0]
        assert "Predictions" in jsonify['data'][0]


    @pytest.mark.parametrize("model_list", Model_List.List_params())
    def test_models(model_list):
        fname = str(Path(Path(__file__).parent.absolute()).parent.absolute())+'/data/stock_dataset.csv'
        if model_list != "":
            validate = myclient.post("/models?types="+ str(model_list) , files={"file": ("upload_file", open(fname, "rb"), fname)})
            jsonify = validate.json()
            assert validate.status_code == 200
            assert jsonify['message'] == 'OK'
            assert jsonify['Model Type'] == str(model_list)
            assert len(jsonify['data']) != 0
            assert "Close" in jsonify['data'][0]
            assert "Predictions" in jsonify['data'][0]
        else:
            validate = myclient.post("/models?types="+ str(model_list) , files={"file": ("upload_file", open(fname, "rb"), fname)})
            jsonify = validate.json()
            assert validate.status_code == 200
            assert jsonify['message'] == 'Choose a Model to Train'

