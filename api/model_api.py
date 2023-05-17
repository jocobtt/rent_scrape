from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
import logging
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field
import uvicorn
# import mlflow
import joblib
from typing import Dict, Optional, List, Any
import pandas as pd
import httpx
import os

from utils.scrape_data import ScrapeData

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

class RetrainData(BaseModel):
    sqr_m: float
    rei_price: float
    shikikin: float
    maintenence_price: float
    year_built: float
    floor: float
    eki_walk: float
    model_to_retrain: Optional[List[str]] = Field(default=["challenger", "passed"])

    
app = FastAPI(debug = True)

# add middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], # edit this to allow only certain origins
    allow_credentials = True,
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)

# Update the load_model function
def load_model(model_version:str):
    try:
        return joblib.load(f"{model_version}-model.joblib")
    except FileNotFoundError:
        if model_version == "challenger":
            return joblib.load("challenger-model.joblib")
        else:
            return joblib.load("passed-model.joblib")

@app.middleware("http")
async def load_model_middleware(request: Request, call_next):
    if not hasattr(app.state, 'challenger_model'):
        app.state.challenger_model = load_model("challenger")
    if not hasattr(app.state, 'passed_model'):
        app.state.passed_model = load_model("passed")
    response = await call_next(request)
    return response

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

@app.post("/retrain")
async def retrain(data: RetrainData):
    # rerun data scraping pipeline 
    ScrapeData(data.url, data.wait_time_min, data.wait_time_max, data.pages)

    # run the model training pipeline
    # Retrain the specified model(s)
    for model_name in data.model_to_retrain:
        if model_name == "challenger":
            # Retrain the challenger model
            challenger_model = ...  # You need to provide the code for retraining the model
            joblib.dump(challenger_model, "challenger-model.joblib")
        elif model_name == "passed":
            # Retrain the passed model
            passed_model = ...  # You need to provide the code for retraining the model
            joblib.dump(passed_model, "passed-model.joblib")

    return {"message": f"Retrained model(s): {data.model_to_retrain}"}

# perform health check 
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# MLFLOW_SERVER_URL = "s3_path"
# # get mlflow info
# @app.get("/{path:path}")
# async def proxy(request: Request, path: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.get(f"{MLFLOW_SERVER_URL}{path}")
#         return response.content, response.status_code
    
# @app.post("/{path:path}")
# async def proxy_post(request: Request, path: str):
#     body = await request.body()
#     async with httpx.AsyncClient() as client:
#         response = await client.post(f"{MLFLOW_SERVER_URL}{path}", data=body)
#         return response.content, response.status_code



# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # fix this to work with firebase
# async def authenticate_user(token: str = Depends(oauth2_scheme)):
#     # Validate the token and get the user
#     user = ...  # You need to provide the code for validating the token and getting the user
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid authentication credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return user


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))