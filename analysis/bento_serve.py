import numpy as np 
import bentoml
from bentoml.io import NumpyNdarrayInput

lgb_model = bentoml.mlflow.get("").to_runner()

svc = bentoml.Service("tokyo_main_svc", runners = [lgb_model])

input_spec = NumpyNdarrayInput(
    dtype = "float64",
    enforce_dtype = True,
    shape = (1, 4),
    enforce_shape = True,
)

@svc.api(input = input_spec, output=NumpyNdarrayInput())
async def classify(input_series: np.ndarray):
    return await lgb_model.predict.async_run(input_series)