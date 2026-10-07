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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make `services` importable when run as a script

# Tracking URI is configured via MLFLOW_TRACKING_URI env var

# `--no-terms` (or NO_TERMS=1) trains the dedicated variant that never sees deposit / key
# money: separate feature file, model file and registry entry. The API routes requests that
# omit both fields to it.
NO_TERMS = os.getenv("NO_TERMS") == "1" or "--no-terms" in sys.argv
VARIANT = "challenger-noterms" if NO_TERMS else "challenger"
REGISTERED_NAME = "tokyo_rent_lgbm_noterms" if NO_TERMS else "tokyo_rent_lgbm"

# Share of training rows whose deposit / key money is hidden (see mask_optional_terms)
TERMS_MASK_FRAC = 0.0 if NO_TERMS else float(os.getenv("TERMS_MASK_FRAC", "0.25"))

# load the data
def load_data(dataset=None):
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.feature_preprocessor import FeaturePreprocessor, mask_optional_terms
    from services.data_split import group_labels, grouped_train_test_split, attach_groups

    from services.data_cleaning import load_training_frame
    df = load_training_frame(dataset)  # explicit path > DATASET_PATH > latest dataset
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    if NO_TERMS:
        df = df.drop(columns=["rei_price", "shikikin"])

    preprocessor = FeaturePreprocessor()
    encoded, _ = preprocessor.fit_transform(df)
    preprocessor.save(os.path.join(os.path.dirname(__file__), f"{VARIANT}-features.json"))

    X = encoded.drop('rent_price', axis=1)
    y = encoded['rent_price']
    # Hold out whole buildings so near-duplicate units can't leak into the test set
    groups = group_labels(df)
    X_train, X_test, y_train, y_test, _ = grouped_train_test_split(X, y, groups, test_size=0.2, random_state=42)
    df_for_lineage = attach_groups(encoded.copy(), groups)
    # Mask deposit / key money on a quarter of the training rows (0.25 was the best trade-off: see README) so the model also works when the
    # API request omits them. Test data stays complete; no-terms scores are logged separately.
    X_train = mask_optional_terms(X_train, frac=TERMS_MASK_FRAC)
    return X_train, X_test, y_train, y_test, df_for_lineage

def tune_params(X_train, y_train, groups, base_params, n_trials=20, n_folds=3, seed=42):
    """Optuna search for LightGBM hyperparameters.

    Scored by grouped K-fold RMSE on the (term-masked) TRAIN split only, with early stopping
    inside each fold, so the held-out test buildings play no part in the search. Returns
    `base_params` updated with the best settings and `num_iterations` set to the mean number
    of rounds the best trial needed (+10%), so the later fixed-length CV refit is sensible.
    """
    import optuna
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(GroupKFold(n_splits=n_folds).split(X_train, y_train, groups))
    best_iters = {}

    def objective(trial):
        p = {
            **base_params,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "num_iterations": 3000,
            "random_state": seed,
        }
        rmses, iters = [], []
        for tr, va in folds:
            m = lgb.LGBMRegressor(**p).fit(
                X_train.iloc[tr], y_train.iloc[tr],
                eval_set=[(X_train.iloc[va], y_train.iloc[va])], eval_metric="l2",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            rmses.append(m.best_score_["valid_0"]["l2"] ** 0.5)
            iters.append(m.best_iteration_)
        best_iters[trial.number] = float(np.mean(iters))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    print(f"Optuna ({n_trials} trials): best CV RMSE={study.best_value:.3f}  params={study.best_params}")
    tuned = {**base_params, **study.best_params, "bagging_freq": 5,
             "num_iterations": int(best_iters[study.best_trial.number] * 1.1) + 1}
    return tuned, study.best_value


def train_and_log_model(X_train, X_test, y_train, y_test, params,
                        df_for_lineage=None,
                        dataset_source_path=None,
                        source_type="local"):
    params = dict(params)
    tuning_trials = params.pop("tuning_trials", 0)
    with mlflow.start_run() as run:
        # Initialize model
        gbm = lgb.LGBMRegressor(**params)

        # CROSS-VALIDATION EVALUATION
        # Import evaluation service
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from services.model_evaluation_service import ModelEvaluationService
        from services.data_split import groups_for

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
            model_name="LightGBM_Challenger",
            groups_train=groups_for(df_for_lineage, X_train)
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
        # Early-stop on a grouped validation slice of the TRAIN set. Stopping on the
        # test set would leak it into the reported metrics.
        from services.data_split import grouped_train_test_split
        X_fit, X_val, y_fit, y_val, _ = grouped_train_test_split(
            X_train, y_train, groups_for(df_for_lineage, X_train), test_size=0.1, random_state=42
        )
        model = gbm.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        y_pred = model.predict(X_test)

        # Test metrics for the model actually being registered (not the CV refit)
        test_metrics = evaluator.calculate_metrics(y_test.values, y_pred)
        rmse = test_metrics['rmse']
        mae = test_metrics['mae']
        r2 = test_metrics['r2']
        mape = test_metrics['mape']
        mse = test_metrics['mse']

        # Same held-out buildings with deposit / key money hidden: what a caller who omits
        # them can expect from the API
        if not NO_TERMS:
            from services.feature_preprocessor import OPTIONAL_TERMS
            X_test_nt = X_test.copy()
            X_test_nt[[c for c in OPTIONAL_TERMS if c in X_test_nt]] = np.nan
            nt = evaluator.calculate_metrics(y_test.values, model.predict(X_test_nt))
            print(f"Test without deposit/key money: RMSE={nt['rmse']:.2f}  R²={nt['r2']:.4f}  MAPE={nt['mape']:.2f}%")
            mlflow.log_metric("rmse_no_terms", nt['rmse'])
            mlflow.log_metric("r2_no_terms", nt['r2'])
            mlflow.log_metric("mape_no_terms", float(nt['mape']))

        # Log model
        mlflow.lightgbm.log_model(model, "lgbm_model")

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("tuning_trials", tuning_trials)
        mlflow.log_param("terms_mask_frac", TERMS_MASK_FRAC)
        mlflow.log_param("variant", VARIANT)

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
        joblib.dump(model, os.path.join(os.path.dirname(__file__), f"{VARIANT}-model.joblib"))

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
        registered_model_name = REGISTERED_NAME
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
    # Dataset: DATASET_PATH env var, else the newest folder in api/dataset_versions/
    from services.data_cleaning import resolve_dataset_path
    data_path = str(resolve_dataset_path())
    X_train, X_test, y_train, y_test, df_for_lineage = load_data(data_path)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': -1,
        'max_depth': 10,
        'num_leaves': 63,
        'num_iterations': 600,
    }
    # Tuning budget: same as the PyTorch models (20 Optuna trials). LGBM_N_TRIALS=0 uses the
    # hand-set params above.
    n_trials = int(os.getenv("LGBM_N_TRIALS", "20"))
    if n_trials > 0:
        from services.data_split import groups_for
        params, tuned_cv_rmse = tune_params(X_train, y_train, groups_for(df_for_lineage, X_train),
                                            params, n_trials=n_trials)
        params["tuning_trials"] = n_trials  # logged with the other params; ignored by LightGBM
    train_and_log_model(X_train, X_test, y_train, y_test, params,
                        df_for_lineage=df_for_lineage,
                        dataset_source_path=data_path,
                        source_type="local")
