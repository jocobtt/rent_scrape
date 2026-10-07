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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make `services` importable when run as a script

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_uri)

# `--no-terms` (or NO_TERMS=1): dedicated variant that never sees deposit / key money
# (see challenger_model.py). The API routes requests that omit both fields to it.
NO_TERMS = os.getenv("NO_TERMS") == "1" or "--no-terms" in sys.argv
VARIANT = "passed-noterms" if NO_TERMS else "passed"
REGISTERED_NAME = "tokyo_passed_rent_noterms" if NO_TERMS else "tokyo_passed_rent_model"

def load_data(dataset=None):
    import re
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.feature_preprocessor import FeaturePreprocessor
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
    return X_train, X_test, y_train, y_test, df_for_lineage

def train_model(X_train, y_train, alpha=1.0):
    models = {
        "LinearRegression": LinearRegression().fit(X_train, y_train),
        "Lasso": Lasso(alpha=alpha).fit(X_train, y_train),
        "Ridge": Ridge(alpha=alpha).fit(X_train, y_train)
    }
    return models

def _final_estimator(model):
    """The fitted regressor inside a pipeline (or the model itself)."""
    return model[-1] if hasattr(model, "steps") else model


def cv_rmse(model, X, y, groups=None, n_splits=5):
    """Grouped (by building) K-fold RMSE on the training split."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, KFold, cross_val_score
    if groups is not None:
        cv, kw = GroupKFold(n_splits=n_splits), {"groups": groups}
    else:
        cv, kw = KFold(n_splits=n_splits, shuffle=True, random_state=42), {}
    scores = cross_val_score(clone(model), X, y, cv=cv, scoring="neg_root_mean_squared_error", **kw)
    return float(-scores.mean())


def tune_linear(X_train, y_train, groups=None, n_trials=20, seed=42):
    """Optuna search over linear families and regularisation strength.

    Each candidate is scaled first (so one alpha means the same thing for every feature) and
    scored by grouped CV RMSE on the training split only. Returns (fitted pipeline, name, cv_rmse).
    """
    import optuna
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def build(trial):
        family = trial.suggest_categorical("family", ["ridge", "lasso", "elasticnet"])
        alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
        if family == "ridge":
            est = Ridge(alpha=alpha)
        elif family == "lasso":
            est = Lasso(alpha=alpha, max_iter=20000)
        else:
            est = ElasticNet(alpha=alpha, l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95), max_iter=20000)
        return make_pipeline(StandardScaler(), est)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: cv_rmse(build(t), X_train, y_train, groups), n_trials=n_trials)
    best = build(optuna.trial.FixedTrial(study.best_params)).fit(X_train, y_train)
    print(f"Optuna ({n_trials} trials): best CV RMSE={study.best_value:.3f}  params={study.best_params}")
    return best, f"Tuned_{study.best_params['family']}", study.best_value


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
        if hasattr(_final_estimator(model), 'alpha'):
            params["alpha"] = _final_estimator(model).alpha
        mlflow.log_params(params)

        # CROSS-VALIDATION EVALUATION
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from services.model_evaluation_service import ModelEvaluationService
        from services.data_split import groups_for

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
            model_name=f"{model_name}_Linear",
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

        # Get test metrics from evaluation report
        test_metrics = evaluation_report['test_set']['metrics']
        y_pred = model.predict(X_test)
        mse = test_metrics['mse']
        rmse = test_metrics['rmse']
        mae = test_metrics['mae']
        r2 = test_metrics['r2']
        mape = test_metrics['mape']

        # Held-out buildings with deposit / key money replaced by their training medians:
        # this is exactly what the API does for the linear model when a caller omits them.
        if not NO_TERMS:
            from services.feature_preprocessor import OPTIONAL_TERMS
            X_nt = X_test.copy()
            for c in OPTIONAL_TERMS:
                if c in X_nt:
                    X_nt[c] = X_train[c].median()
            nt = evaluator.calculate_metrics(y_test.values, model.predict(X_nt))
            print(f"Test without deposit/key money (median-imputed): RMSE={nt['rmse']:.2f}  R²={nt['r2']:.4f}")
            mlflow.log_metric('rmse_no_terms', nt['rmse'])
            mlflow.log_metric('r2_no_terms', nt['r2'])
            mlflow.log_metric('mape_no_terms', float(nt['mape']))

        # Log all metrics
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2_score', r2)
        mlflow.log_metric('mape', float(mape))

        # Log model and SHAP values only if the model is the best model
        # NB: `best_mse` is the best candidate's RMSE (evaluate_model returns the square root),
        # so compare RMSE with RMSE; comparing it with the squared error never matched.
        if rmse <= best_mse + 1e-9:
            # Log model
            mlflow.sklearn.log_model(model, mlflow_name)

            # Log feature importance (coefficients for linear models)
            if hasattr(_final_estimator(model), 'coef_'):
                import json
                feature_names = X_train.columns.tolist()
                coefficients = _final_estimator(model).coef_  # standardised scale for pipelines

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
                    if hasattr(model, "steps"):
                        raise RuntimeError("pipeline: coefficient importance already logged")
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
                        source_description="Suumo scrape (api/dataset_versions)",
                        source_type="scraped",
                        dataset_name="tokyo_rent_training",
                        context="training",
                    )
                except Exception as _e:
                    print(f"Warning: Dataset lineage logging failed (non-fatal): {_e}")

            model_uri = f"runs:/{run_id}/{mlflow_name}"

            # register the model
            registered_model_name = REGISTERED_NAME
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

    from services.data_split import groups_for
    groups = groups_for(df_for_lineage, X_train)
    models = train_model(X_train, y_train, alpha=1.0)  # untuned baselines
    n_trials = int(os.getenv("LINEAR_N_TRIALS", "20"))
    if n_trials > 0:
        tuned, tuned_name, _ = tune_linear(X_train, y_train, groups, n_trials=n_trials)
        models[tuned_name] = tuned

    # Pick the winner by grouped CV on the training split, not on the test set
    scores = {name: cv_rmse(m, X_train, y_train, groups) for name, m in models.items()}
    best_model_name = min(scores, key=scores.get)
    best_model = models[best_model_name]
    for name, m in models.items():
        print(f"{name:<22} CV RMSE={scores[name]:.3f}  test RMSE={evaluate_model(m, X_test, y_test):.3f}")
    best_mse = evaluate_model(best_model, X_test, y_test)  # test RMSE of the winner

    print(f"Best Model: {best_model_name} with test RMSE: {best_mse}")

    # Always log to MLflow; the auto-promotion service decides whether it ships
    log_mlflow(best_model_name, best_model, X_train, y_train, X_test, y_test, mlflow_name="reg_passed_model", best_mse=best_mse, df_for_lineage=df_for_lineage)
    path = os.path.join(os.path.dirname(__file__), f"{VARIANT}-model.joblib")
    joblib.dump(best_model, path)
