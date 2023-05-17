import os 
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import shap
import joblib
import matplotlib.pyplot as plt
import bentoml

mlflow.set_tracking_uri("s3_path")

def load_data():
    data = load_dataset("jbrazzy/tokyo_rent", split="train")
    df = data.to_pandas()
    #df = pd.read_csv("../data/tokyo_model.csv")
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
    df = df.drop(['address'], axis = 1)
    X = df.drop('rent_price', axis = 1)
    y = df['rent_price']
    return train_test_split(X, y, test_size = 0.2, random_state = 42)

def train_model(X_train, y_train, alpha=1.0):
    models = {
        "LinearRegression": LinearRegression().fit(X_train, y_train),
        "Lasso": Lasso(alpha=alpha).fit(X_train, y_train),
        "Ridge": Ridge(alpha=alpha).fit(X_train, y_train)
    }
    return models

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions, squared=False)
    return mse

def get_shap_values(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    return shap_values

def log_mlflow(model_name, model, X_train, X_test, y_test, mlflow_name):
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        params = {
            "test_size": 0.2,
            "random_state": 42,
            "model_type": model_name
        }
        if hasattr(model, 'alpha'):
            params["alpha"] = model.alpha
        mlflow.log_params(params)
        
        # Log MSE
        mse = evaluate_model(model, X_test, y_test)
        mlflow.log_metric('mse', mse)

        # Log model and SHAP values only if the model is the best model
        if mse <= best_mse:
            # Log model
            mlflow.sklearn.log_model(model, mlflow_name)
            
            # Log SHAP values - fix this part
            #shap_values = get_shap_values(model, X_train)
            #shap.summary_plot(shap_values, X_train, show=False)
            #plt.savefig("passed_model_shap_summary.png")
            #mlflow.log_artifact("passed_model_shap_summary.png")
            
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/{mlflow_name}"
            
            # register the model 
            registered_model_name = "tokyo_passed_rent_model"
            mlflow.register_model(model_uri, registered_model_name)
            # save in bentoml format also 
            bento_mlflow_uri = mlflow.get_artifact_uri("reg_passed_model")
            bento_model = bentoml.mlflow.import_model(
                "reg_model", bento_mlflow_uri, signatures = {"predict": {"batchable": True}}, labels=run.data.tags, 
                metadata = {"metrics": run.data.metrics, "params": run.data.params, "tags": run.data.tags, "run_id": run_id}
            )
            print("Model imported to BentoML: %s" % bento_model)

        return mse

if __name__ == "__main__":
    mlflow.set_tracking_uri("s3_path")
    X_train, X_test, y_train, y_test = load_data()
    
    models = train_model(X_train, y_train, alpha=1.0)
    best_mse = float('inf')
    best_model = None
    best_model_name = ""

    for model_name, model in models.items():
        mse = evaluate_model(model, X_test, y_test)
        print(f"{model_name} MSE: {mse}")

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = model_name

    print(f"Best Model: {best_model_name} with MSE: {best_mse}")
    
    # Log the best model to MLflow and save it
    if best_mse < 4:  # or any other threshold you define
        log_mlflow(best_model_name, best_model, X_train, X_test, y_test, mlflow_name="reg_passed_model")
        path = os.path.join(os.getcwd(), "passed-model.joblib")
        joblib.dump(best_model, path)
    else:
        print("No model is good enough to ship, consider investigating further.")