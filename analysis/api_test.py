import requests
from pydantic import BaseModel
from typing import Optional

# class Input(BaseModel):
#     rei_price: Optional[int]
#     shikikin: Optional[int]
#     maintenence_price: Optional[int]
#     sqr_m: Optional[int]

url = "http://localhost:8000/predict"
# data = Input(rei_price=1.0, shikikin = 2, maintenence_price=12, sqr_m = 1000).dict()

data = {
    "sqr_m": 30.0,
    "rei_price": 200000.0,
    "shikikin": 100000.0,
    "maintenence_price": 10000.0,
    "year_built": 4.0,
    "floor": 2.0,
    "eki_walk": 5.0

}

response = requests.post(url, json=data)
print("Response status code:", response.status_code)
print("Response content:", response.content)

if response.status_code == 200:
    try:
        print("Predicted rent:", response.json()["prediction"])
    except Exception as e:
        print("Error decoding JSON:", e)
else:
    print("Error:", response.text)