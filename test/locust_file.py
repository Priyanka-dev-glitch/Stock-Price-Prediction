import os
import sys
import pandas as pd
from pathlib import Path
from locust import HttpUser, task, between, tag
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from src.schema import Model_List



class StockPrediction_locust(HttpUser):
    wait_time = between(1, 5)


    @tag('Swagger API')
    @task(1)
    def swagger_ui(self):
        self.client.get('/docs')
    
    
    @tag('List of Endpoints')
    @task(1)
    def list_url(self):
        self.client.get("/")


    @tag('Get All Models')
    @task(6)
    def list_models(self):
        self.client.get("/listmodels")


    @tag('Dynamic Model List')
    @task(10)
    def predict(self):
            self.fpath = str(Path(Path(__file__).parent.absolute()).parent.absolute())
            pd.read_pickle(self.fpath+"/data/stock_dataset.pkl").to_csv(self.fpath+"/data/stock_dataset.csv", index=True)
            self.fname = self.fpath+'/data/stock_dataset.csv'
            for val in Model_List.List_params():
                if val != '':
                    self.client.post(f'/models?types='+ str(val), files={"file": ("upload_file", open(self.fname, "rb"), self.fname)})
