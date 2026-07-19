# build a challenger model using lightgbm
import joblib
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import re
from datasets import load_dataset
from mlflow import log_metric, log_param, log_artifact
try:
    import bentoml
except ImportError:
    bentoml = None

# Tracking URI is configured via MLFLOW_TRACKING_URI env var

# load the data
def load_data(dataset):
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.feature_preprocessor import FeaturePreprocessor

    df = pd.read_csv(dataset)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    preprocessor = FeaturePreprocessor()
    encoded, _ = preprocessor.fit_transform(df)
    preprocessor.save(os.path.join(os.path.dirname(__file__), "challenger-features.json"))

    df_for_lineage = encoded.copy()
    X = encoded.drop('rent_price', axis=1)
    y = encoded['rent_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, df_for_lineage

def train_and_log_model(X_train, X_test, y_train, y_test, params,
                        df_for_lineage=None,
                        dataset_source_path=None,
                        source_type="local"):
    with mlflow.start_run() as run:
        # Initialize model
        gbm = lgb.LGBMRegressor(**params)

        # CROSS-VALIDATION EVALUATION
        # Import evaluation service
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from services.model_evaluation_service import ModelEvaluationService

        print("\n" + "="*70)
        print("PERFORMING COMPREHENSIVE MODEL EVALUATION")
        print("="*70)

        # Perform cross-validation on training data
        evaluator = ModelEvaluationService(n_folds=5, random_state=42)

        # Fit model for CV evaluation (without early stopping for fair comparison)
        cv_model = lgb.LGBMRegressor(**params)
        evaluation_report = evaluator.evaluate_comprehensive(
            cv_model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name="LightGBM_Challenger"
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

        # NOW TRAIN FINAL MODEL WITH EARLY STOPPING
        print("\nTraining final model with early stopping...")
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
        model = gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )
        y_pred = model.predict(X_test)

        # Calculate test set metrics (from evaluation report)
        test_metrics = evaluation_report['test_set']['metrics']
        rmse = test_metrics['rmse']
        mae = test_metrics['mae']
        r2 = test_metrics['r2']
        mape = test_metrics['mape']
        mse = test_metrics['mse']

        # Log model
        mlflow.lightgbm.log_model(model, "lgbm_model")

        # Log parameters
        mlflow.log_params(params)

        # Log all metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", float(mape))

        # Log feature importance
        feature_importance = model.feature_importances_
        feature_names = X_train.columns.tolist()

        # Create feature importance data
        import json
        importance_data = [
            {"name": name, "importance": float(importance)}
            for name, importance in zip(feature_names, feature_importance)
        ]
        # Sort by importance (descending)
        importance_data.sort(key=lambda x: x["importance"], reverse=True)

        # Save feature importance as JSON artifact
        temp_file = "/tmp/feature_importance.json"
        with open(temp_file, 'w') as f:
            json.dump(importance_data, f, indent=2)
        mlflow.log_artifact(temp_file, None)
        os.remove(temp_file)

        # Log SHAP-based global feature importance
        try:
            explainer = shap.TreeExplainer(model)
            sample = X_train.iloc[:min(200, len(X_train))]
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

        # Save model locally
        joblib.dump(model, os.path.join(os.path.dirname(__file__), "challenger-model.joblib"))

        run_id = run.info.run_id

        # Dataset lineage logging (non-fatal)
        if df_for_lineage is not None:
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from services.data_lineage_service import DataLineageService
                source_desc = (f"CSV: {dataset_source_path}" if source_type == "local"
                               else f"Suumo scrape: {dataset_source_path}")
                DataLineageService().log_dataset_to_run(
                    df=df_for_lineage, run_id=run_id,
                    source_description=source_desc,
                    source_type=source_type,
                    dataset_name="tokyo_rent_training",
                    context="training",
                )
            except Exception as _e:
                print(f"Warning: Dataset lineage logging failed (non-fatal): {_e}")

        model_uri = f"runs:/{run_id}/lgbm_model"

        # register the model
        registered_model_name = "tokyo_rent_lgbm"
        model_details = mlflow.register_model(model_uri, registered_model_name)

        # Update model version description
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.update_model_version(
            name=registered_model_name,
            version=model_details.version,
            description=f"LightGBM model trained on Tokyo apartment data. Test RMSE: {rmse:.4f}, R²: {r2:.4f} | CV RMSE: {cv_metrics['rmse_mean']:.4f}±{cv_metrics['rmse_std']:.4f}"
        )

        # AUTO-PROMOTION: Check if model should be promoted to production
        try:
            # os and sys are already imported at module/function scope above
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
            bento_model = bentoml.mlflow.import_model(
                "tokyo_rent_lgbm", model_uri, signatures = {"predict": {"batchable": True}}
            )
            print("Model imported to BentoML: %s" % bento_model)
        except Exception as e:
            print(f"Warning: Could not import to BentoML: {e}")

    return model

if __name__ == '__main__':
    # eventually move the dataset to huggingface
    X_train, X_test, y_train, y_test, df_for_lineage = load_data("../../data/tokyo_model.csv")
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
    train_and_log_model(X_train, X_test, y_train, y_test, params,
                        df_for_lineage=df_for_lineage,
                        dataset_source_path="../../data/tokyo_model.csv",
                        source_type="local")
