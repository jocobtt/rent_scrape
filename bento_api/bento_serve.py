import bentoml
from bentoml.validators import DataframeSchema # https://docs.bentoml.org/en/latest/guides/iotypes.html#tabular
from bentoml.mlflow import load_model 
import mlflow
import polars as pl
from bentoml.io import NumpyNdarray
from typing import Annotated, List, Union
import numpy as np
from bentoml.validators import Shape


@bentoml.env(infer_pip_packages=True)
@bentoml.service(name="model_service", resources={"cpu": "200m", "memory": "512Mi"},traffic={"timeout": 10},) # k8s_resource_yaml="k8s_resource.yaml"
class MyModelService:
    
    # Assuming the imported model names are "reg_model_lgbm" and "reg_model_sklearn" for demonstration
    lgbm_model_info = bentoml.models.get("reg_model_lgbm:latest")
    sklearn_model_info = bentoml.models.get("reg_model_sklearn:latest")

    def __init__(self):
        self.lgbm_model = self.lgbm_model_info.to_runner()
        self.sklearn_model = self.sklearn_model_info.to_runner()

        self.add_runner(self.lgbm_model)
        self.add_runner(self.sklearn_model)

    @bentoml.api(input=DataframeSchema(), output=DataframeSchema(), batch = True)
    def predict_lgbm(self, df: pl.DataFrame):
        return pl.DataFrame(self.lgbm_model.predict(df))
    
    @bentoml.api(input=DataframeSchema(), output=DataframeSchema(), batch = True)
    def predict_sklearn(self, df: pl.DataFrame):
        return pl.DataFrame(self.sklearn_model.predict(df))
    
    @bentoml.api(input=NumpyNdarray(), output=NumpyNdarray())
    def run_lgbm(self, input_data: Annotated[np.ndarray, Shape(None)]):
        return self.lgbm_model.run(input_data)

    @bentoml.api(input=NumpyNdarray(), output=NumpyNdarray())
    def run_sklearn(self, input_data: Annotated[np.ndarray, Shape(None)]):
        return self.sklearn_model.run(input_data)

# Create a BentoML service
svc = MyModelService()

# Save the BentoML service
saved_path = bentoml.models.save(svc)

print(f"Model saved at {saved_path}")

# serve the model 


# Traceback (most recent call last):
#   File "/Users/jabras/rent_scrape/analysis/bento_serve.py", line 4, in <module>
#     from bentoml.adapters import DataframeInput, JsonInput
# ModuleNotFoundError: No module named 'bentoml.adapters'
# serve the model's now based off of which one you want to use
#bento_svc = load_model(saved_path)
#bento_svc.predict_lgbm(df)
#bento_svc.predict_sklearn(df)
