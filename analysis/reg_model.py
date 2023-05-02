import os 
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import joblib
# import bentoml

# load the data 
df = pd.read_csv("/Users/jabras/rent_scrape/data/tokyo_model.csv")
# will change the datasource to instead be from cloud storage

# apartment type, floor, year built, and house type are categorical variables
# df = pd.get_dummies(df, columns = ["apartment_type", "house_type", "ku_name"])
# drop name_place, address, and eki_name for now 
df = df.drop(['name_place', 'address', 'eki_name', "apartment_type", "house_type", "ku_name"], axis = 1)

X = df.drop('rent_price', axis = 1)
y = df['rent_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# make test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = LinearRegression().fit(X_train, y_train)
# make model include other variables and play around with it a bit more
# log model with MLflow
with mlflow.start_run():
    model_uri = mlflow.get_artifact_uri("model")
    mlflow.sklearn.log_model(model, 'statsmodels-lr')
    mlflow.log_metric('mse', mean_squared_error(y_test, model.predict(X_test)))
    mlflow.log_param('test_size', 0.2)
    mlflow.log_param('random_state', 42)
    mlflow.log_param('model', 'LinearRegression')
    #mlflow.log_param('features', list(X.columns))
    
    # bentoml 
    # bento_model = bentoml.mlflow.import_model("tokyo_model", model_uri, signatures = {"predict": {"batchable": True}})
    # print("model imported to bentoml: %s" % bento_model)

# make predictions and calculate the mean squared error
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print(mse)
if mse < 4:
    print("model is good")
    # and save it 
    # save the model
    path = os.path.join(os.getcwd(), "passed-model.joblib")
    joblib.dump(model, path)    
else:
    print("model is not good enough to ship, consider investigating.")

