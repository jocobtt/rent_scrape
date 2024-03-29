from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, BaseSettings
import uvicorn
# import mlflow
import joblib
from typing import Dict, Optional, List, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class MlflowRun(BaseModel):
#     run_id: str
#     experiment_name: str
#     experiment_id: str
#     metrics: Dict[str, float]
#     params: Dict[str, str]

class InputData(BaseModel):
    sqr_m: float
    rei_price: float
    shikikin: float
    maintenence_price: float
    year_built: float
    floor: float
    eki_walk: float
    
app = FastAPI(debug = True)

# add middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

def load_model(model_version:str):
    if model_version == "challenger":
         return joblib.load("challenger-model.joblib")
    else:
        return joblib.load("passed-model.joblib")

@app.on_event("startup")
async def startup_event():
    app.state.challenger_model = load_model("challenger")
    app.state.passed_model = load_model("passed")

@app.get("/")
def home():
    return "Welcome to Tokyo Rent Predictor!"

@app.post("/predict")
async def predict_main(data: InputData):
    # load our model 
    model = app.state.passed_model
    # make the prediction
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    # return the prediction
    return {"prediction": pred}

@app.post("/challenger_predict")
async def challenger_predict(data: InputData):
    # make the prediction
    model = app.state.challenger_model
    pred = model.predict([[data.rei_price, data.shikikin, data.maintenence_price, data.year_built, data.floor, data.eki_walk, data.sqr_m]])
    # return the prediction
    return {"prediction": pred}

# @app.get("/api/mlflow-data", response_model=List[MLflowRun])
# async def get_mlflow_data() -> List[MLflowRun]:
#     mlflow.set_tracking_uri(mlflow_data_path)
#     client = mlflow.tracking.MlflowClient()
#     experiments = client.list_experiments()

#     data = []
#     for exp in experiments:
#         runs = client.search_runs(exp.experiment_id)
#         for run in runs:
#             run_data = MLflowRun(
#                 experiment_id=exp.experiment_id,
#                 experiment_name=exp.name,
#                 run_id=run.info.run_id,
#                 metrics=run.data.metrics,
#                 params=run.data.params,
#             )
#             data.append(run_data)

#     return data

# @app.get("/refresh_model")
# async def refresh_model():
#     # load the model
#     model = load_model()
#     # get the run id of the model 
#     run_id = model.run_id
#     # get the metrics of the model 
#     metrics = get_mlflow_metrics(run_id)
#     # check if the model is good 
#     if metrics['mse'] < 4:
#         # if it is good, return the model 
#         return model
#     else:
#         # if it is not good, return a message saying it is not good 
#         return "model is not good"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)