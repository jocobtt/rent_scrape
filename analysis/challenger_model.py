# build a challenger model using lightgbm
import joblib
import os 
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import re
# import bentoml

# load the data
df = pd.read_csv("/Users/jabras/rent_scrape/data/tokyo_model.csv")

# rename columns so we don't encounter json error for special characters in column names
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# apartment type, floor, year built, and house type are categorical variables
# drop name_place, address, and eki_name for now 
df = df.drop(['name_place', 'address', 'eki_name', 'ku_name', 'apartment_type', 'house_type'], axis = 1)

# make test and train sets x and y 
X = df.drop('rent_price', axis = 1)
y = df['rent_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Train model
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# Log & store in mlflow 
with mlflow.start_run():
    params = {
    "objective": "regression",
    "metric": "mse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    # Evaluate model
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    model_uri = mlflow.get_artifact_uri("model")
    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_metric("mse", rmse)
    mlflow.log_param("num_boost_round", model.best_iteration)
    mlflow.log_param("early_stopping_rounds", 10)
    mlflow.log_param("objective", "regression")
    mlflow.log_param("metric", "rmse")
    mlflow.log_param("verbosity", -1)
    mlflow.log_param("boosting_type", "gbdt")
    mlflow.log_param("num_leaves", 31)
    mlflow.log_param("max_depth", -1)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)
    #mlflow.log_artifact("analysis/challenger_model.py")
    #mlflow.log_artifact("analysis/model_api.py")
    # Save model with mlem
    joblib.dump(model, "challenger-model.joblib")

    # # bentoml 
    # bento_model = bentoml.mlflow.import_model("tokyo_model_challenger", model_uri, signatures = {"predict": {"batchable": True}})
    # print("model imported to bentoml: %s" % bento_model)