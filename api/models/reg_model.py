import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import shap
import joblib
import matplotlib.pyplot as plt
try:
    import bentoml
except ImportError:
    bentoml = None

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_uri)

def load_data(dataset=None):
    import re
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.feature_preprocessor import FeaturePreprocessor

    if dataset is not None:
        df = pd.read_csv(dataset)
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    else:
        data = load_dataset("jbrazzy/tokyo_rent", split="train")
        df = data.to_pandas()

    preprocessor = FeaturePreprocessor()
    encoded, _ = preprocessor.fit_transform(df)
    preprocessor.save(os.path.join(os.path.dirname(__file__), "passed-features.json"))

    df_for_lineage = encoded.copy()
    X = encoded.drop('rent_price', axis=1)
    y = encoded['rent_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, df_for_lineage

def train_model(X_train, y_train, alpha=1.0):
    models = {
        "LinearRegression": LinearRegression().fit(X_train, y_train),
        "Lasso": Lasso(alpha=alpha).fit(X_train, y_train),
        "Ridge": Ridge(alpha=alpha).fit(X_train, y_train)
    }
    return models

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, predictions))
    return mse

def get_shap_values(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    return shap_values

def log_mlflow(model_name, model, X_train, y_train, X_test, y_test, mlflow_name, best_mse,
               df_for_lineage=None):
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

        # CROSS-VALIDATION EVALUATION
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from services.model_evaluation_service import ModelEvaluationService

        print("\n" + "="*70)
        print(f"EVALUATING {model_name} WITH CROSS-VALIDATION")
        print("="*70)

        # Perform comprehensive evaluation
        evaluator = ModelEvaluationService(n_folds=5, random_state=42)
        evaluation_report = evaluator.evaluate_comprehensive(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=f"{model_name}_Linear"
        )

        # Print summary
        evaluator.print_summary(evaluation_report)

        # Save evaluation report
        report_path = evaluator.save_report(evaluation_report)

        # Log CV metrics to MLflow
        cv_metrics = evaluation_report['cross_validation']['metrics']
        mlflow.log_metric("cv_rmse_mean", cv_metrics['rmse_mean'])
        mlflow.log_metric("cv_rmse_std", cv_metrics['rmse_std'])
        mlflow.log_metric("cv_r2_mean", cv_metrics['r2_mean'])
        mlflow.log_metric("cv_r2_std", cv_metrics['r2_std'])
        mlflow.log_metric("cv_mae_mean", cv_metrics['mae_mean'])
        mlflow.log_metric("cv_overfitting_gap_rmse", cv_metrics['overfitting_gap_rmse'])

        # Log evaluation report as artifact
        mlflow.log_artifact(report_path, "evaluation")

        # Get test metrics from evaluation report
        test_metrics = evaluation_report['test_set']['metrics']
        y_pred = model.predict(X_test)
        mse = test_metrics['mse']
        rmse = test_metrics['rmse']
        mae = test_metrics['mae']
        r2 = test_metrics['r2']
        mape = test_metrics['mape']

        # Log all metrics
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2_score', r2)
        mlflow.log_metric('mape', float(mape))

        # Log model and SHAP values only if the model is the best model
        if mse <= best_mse:
            # Log model
            mlflow.sklearn.log_model(model, mlflow_name)

            # Log feature importance (coefficients for linear models)
            if hasattr(model, 'coef_'):
                import json
                feature_names = X_train.columns.tolist()
                coefficients = model.coef_

                # Create feature importance data (use absolute values for importance)
                importance_data = [
                    {"name": name, "importance": float(abs(coef))}
                    for name, coef in zip(feature_names, coefficients)
                ]
                # Sort by importance (descending)
                importance_data.sort(key=lambda x: x["importance"], reverse=True)

                # Save feature importance as JSON artifact
                temp_file = "/tmp/feature_importance.json"
                with open(temp_file, 'w') as f:
                    json.dump(importance_data, f, indent=2)
                mlflow.log_artifact(temp_file, None)
                os.remove(temp_file)

                # Log SHAP-based global feature importance (LinearExplainer — fast & exact)
                try:
                    sample = X_train.iloc[:min(200, len(X_train))]
                    bg = shap.maskers.Independent(sample, max_samples=100)
                    explainer = shap.LinearExplainer(model, bg)
                    shap_values = explainer.shap_values(sample)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = [
                        {"name": name, "mean_abs_shap": float(v)}
                        for name, v in zip(feature_names, mean_abs_shap)
                    ]
                    shap_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
                    shap_path = "/tmp/shap_importance.json"
                    with open(shap_path, "w") as f:
                        json.dump(shap_importance, f, indent=2)
                    mlflow.log_artifact(shap_path, None)
                    os.remove(shap_path)
                    print(f"SHAP importance logged. Top feature: {shap_importance[0]['name']}")
                except Exception as e:
                    print(f"SHAP logging skipped: {e}")

            run_id = run.info.run_id

            # Dataset lineage logging (non-fatal)
            if df_for_lineage is not None:
                try:
                    import sys as _sys
                    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from services.data_lineage_service import DataLineageService
                    DataLineageService().log_dataset_to_run(
                        df=df_for_lineage, run_id=run_id,
                        source_description="HuggingFace jbrazzy/tokyo_rent split=train",
                        source_type="huggingface",
                        dataset_name="tokyo_rent_training",
                        context="training",
                    )
                except Exception as _e:
                    print(f"Warning: Dataset lineage logging failed (non-fatal): {_e}")

            model_uri = f"runs:/{run_id}/{mlflow_name}"

            # register the model
            registered_model_name = "tokyo_passed_rent_model"
            model_details = mlflow.register_model(model_uri, registered_model_name)

            # Update model version description
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=model_details.version,
                description=f"{model_name} model trained on Tokyo apartment data. Test RMSE: {rmse:.4f}, R²: {r2:.4f} | CV RMSE: {cv_metrics['rmse_mean']:.4f}±{cv_metrics['rmse_std']:.4f}"
            )

            # AUTO-PROMOTION: Check if model should be promoted to production
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from services.auto_promotion_service import AutoPromotionService

                auto_promotion = AutoPromotionService()
                promotion_result = auto_promotion.check_model_for_promotion(
                    model_name=registered_model_name,
                    dry_run=False  # Set to True to test without promoting
                )

                if promotion_result["promoted"]:
                    print(f"\n{'='*60}")
                    print(f"🎉 MODEL AUTO-PROMOTED TO PRODUCTION!")
                    print(f"{'='*60}")
                    print(f"Model: {registered_model_name} v{promotion_result['version']}")
                    print(f"Reason: {promotion_result['reason']}")
                    print(f"Metrics: RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
                    print(f"{'='*60}\n")
                else:
                    print(f"\nℹ️  Model NOT promoted: {promotion_result['reason']}")
                    print(f"   Current metrics: RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%\n")
            except Exception as e:
                print(f"Warning: Auto-promotion check failed: {e}")

            # save in bentoml format also
            try:
                bento_mlflow_uri = mlflow.get_artifact_uri("reg_passed_model")
                bento_model = bentoml.mlflow.import_model(
                    "reg_model", bento_mlflow_uri, signatures = {"predict": {"batchable": True}}, labels=run.data.tags,
                    metadata = {"metrics": run.data.metrics, "params": run.data.params, "tags": run.data.tags, "run_id": run_id}
                )
                print("Model imported to BentoML: %s" % bento_model)
            except Exception as e:
                print(f"Warning: Could not import to BentoML: {e}")

        return mse

if __name__ == "__main__":
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    X_train, X_test, y_train, y_test, df_for_lineage = load_data()

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
        log_mlflow(best_model_name, best_model, X_train, y_train, X_test, y_test, mlflow_name="reg_passed_model", best_mse=best_mse, df_for_lineage=df_for_lineage)
        path = os.path.join(os.path.dirname(__file__), "passed-model.joblib")
        joblib.dump(best_model, path)
    else:
        print("No model is good enough to ship, consider investigating further.")