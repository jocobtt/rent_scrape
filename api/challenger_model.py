# build a challenger model using lightgbm
import joblib
import os 
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import re
from datasets import load_dataset
from mlflow import log_metric, log_param, log_artifact
import bentoml

mlflow.set_tracking_uri("s3_path")

# load the data
def load_data(dataset):
    #data = load_dataset(dataset, split="train")
    #df = data.to_pandas()
    df = pd.read_csv(dataset)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # make ku_name a categorical variable and one hot encode it
    df['ku_name'] = df['ku_name'].astype('category')
    df = pd.get_dummies(df, columns=['ku_name'])
    # make apartment_type a categorical variable and one hot encode it
    df['apartment_type'] = df['apartment_type'].astype('category')
    df = pd.get_dummies(df, columns=['apartment_type'])
    # make house_type a categorical variable and one hot encode it
    df['house_type'] = df['house_type'].astype('category')
    df = pd.get_dummies(df, columns=['house_type'])
    # drop the address column
    df = df.drop(['address'], axis=1)
    X = df.drop('rent_price', axis=1)
    y = df['rent_price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run() as run:
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
        gbm = lgb.LGBMRegressor(**params)
        model = gbm.fit(
            X_train,
            y_train,
            eval_set= [(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=100
            #log_evaluation=10
        )
        y_pred = model.predict(X_test, num_iteration=model._best_iteration)
        mse = mean_squared_error(y_test, y_pred, squared=False)
        
        mlflow.lightgbm.log_model(model, "lgbm_model")
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        
        joblib.dump(model, "challenger-model.joblib")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/lgbm_model"
        
        # register the model 
        registered_model_name = "tokyo_rent_lgbm"
        mlflow.register_model(model_uri, registered_model_name)
        
        # save in bentoml format also 
        bento_model = bentoml.mlflow.import_model(
            "tokyo_rent_lgbm", model_uri, signatures = {"predict": {"batchable": True}}
        )
        print("Model imported to BentoML: %s" % bento_model)
    return model

if __name__ == '__main__':
    # eventually move the dataset to huggingface
    X_train, X_test, y_train, y_test = load_data("../data/tokyo_model.csv")
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000
    }
    train_and_log_model(X_train, X_test, y_train, y_test, params)
