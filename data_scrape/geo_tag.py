import pandas as pd 
import os 
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# load data 
df = pd.read_csv("../data/tokyo_model.csv")

# geocode addresses 
# initialize geolocator
geolocator = Nominatim(user_agent="tokyo_rent_predictor")

# create a rate limiter to avoid getting blocked
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# function for geocoding 
def geocode_address(address):
    try: 
        location = geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(e)
    return None, None

# apply the function
df["lat"], df["lon"] = zip(*df["address"].apply(geocode_address))

# save the data
df.to_csv("../data/tokyo_model_geo.csv", index=False)
    