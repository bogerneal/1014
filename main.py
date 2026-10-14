"""50 Startups: CRISP-DM multiple linear regression and feature selection."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
FEATURES = ["R&D Spend", "Administration", "Marketing Spend",
            "State_Florida", "State_New York"]
METHODS = ["Baseline", "LassoCV", "RFECV", "SFS", "ElasticNetCV", "Exhaustive"]


def folds(seed=42):
    return KFold(n_splits=5, shuffle=True, random_state=seed)


def linear_pipeline():
    return Pipeline([("scale", StandardScaler()), ("regression", LinearRegression())])


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Fit every selection procedure using only the current training fold."""

    def __init__(self, method="Baseline", seed=42):
        self.method = method
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        cv = folds(self.seed)
        support = np.ones(X.shape[1], dtype=bool)
        if self.method in ("LassoCV", "ElasticNetCV"):
            scaled = StandardScaler().fit_transform(X)
            # Internal CV tunes regularization; outer CV evaluates the whole selector.
            if self.method == "LassoCV":
                selector = LassoCV(cv=cv, max_iter=100000)
            else:
                selector = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99, 1.0],
                                       cv=cv, max_iter=100000)
            selector.fit(scaled, y)
            support = np.abs(selector.coef_) > 1e-8
            self.regularization_alpha_ = float(selector.alpha_)
        elif self.method == "RFECV":
            selector = RFECV(linear_pipeline(), step=1, min_features_to_select=1,
                             cv=cv, scoring="neg_root_mean_squared_error",
                             importance_getter="named_steps.regression.coef_")
            selector.fit(X, y)
            support = selector.support_
        elif self.method in ("SFS", "Exhaustive"):
            candidates = []
            if self.method == "SFS":
                for count in range(1, X.shape[1]):
                    selector = SequentialFeatureSelector(
                        linear_pipeline(), n_features_to_select=count,
                        direction="forward", cv=cv,
                        scoring="neg_root_mean_squared_error")
                    selector.fit(X, y)
                    candidates.append(np.flatnonzero(selector.get_support()))
                candidates.append(np.arange(X.shape[1]))
            else:
                candidates = [np.array(subset) for count in range(1, X.shape[1] + 1)
                              for subset in itertools.combinations(range(X.shape[1]), count)]
            best_score = -np.inf
            for subset in candidates:
                score = cross_val_score(linear_pipeline(), X[:, subset], y, cv=cv,
                                       scoring="neg_root_mean_squared_error").mean()
                if score > best_score:
                    best_score = score
                    support = np.zeros(X.shape[1], dtype=bool)
                    support[subset] = True
        elif self.method != "Baseline":
            raise ValueError(f"Unknown method: {self.method}")
        # An all-zero penalized fit yields an intercept-only OLS model below.
        self.support_ = support
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.support_.any():
            return np.zeros((len(X), 1))
        return X[:, self.support_]


def make_model(method, seed):
    return Pipeline([("select", FeatureSelector(method, seed)),
                     ("scale", StandardScaler()),
                     ("regression", LinearRegression())])


def load_data(path):
    df = pd.read_csv(path)
    required = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[required].copy()
    if df.isna().any().any():
        raise ValueError("Missing values found; resolve them before analysis.")
    for column in [*required[:3], "Profit"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if not np.isfinite(df[[*required[:3], "Profit"]].to_numpy()).all():
        raise ValueError("Non-finite numeric values found.")
    if not df.State.isin(["California", "Florida", "New York"]).all():
        raise ValueError("Unexpected State category.")
    if len(df) < 20:
        raise ValueError("At least 20 rows are required for this CV configuration.")
    X = df[required[:3]].copy()
    X["State_Florida"] = (df.State == "Florida").astype(float)
    X["State_New York"] = (df.State == "New York").astype(float)
    return df, X.astype(float), df.Profit


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(root_mean_squared_error(y, pred)),
            "R2": float(r2_score(y, pred))}


def save_plot(output, name):
    plt.tight_layout()
    plt.savefig(output / name, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/50_Startups.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output

    print("PHASE 1: BUSINESS UNDERSTANDING", flush=True)
    print("Predict Profit from spending and State; California is the reference category.")
    print("PHASE 2: DATA UNDERSTANDING", flush=True)
    df, X, y = load_data(args.data)
    df.describe(include="all").to_csv(out / "data_summary.csv")
    print(f"Rows: {len(df)}; features: {len(FEATURES)}; duplicate rows: {df.duplicated().sum()}")

    print("PHASE 3: DATA PREPARATION", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed)
    pd.DataFrame({"row_id": df.index + 1,
                  "split": np.where(df.index.isin(X_test.index), "test", "train")}
                 ).to_csv(out / "split.csv", index=False)

    print("PHASE 4: MODELING (training-only nested feature selection)", flush=True)
    rows, cv_rows, fitted = [], [], {}
    for method in METHODS:
        print(f"  Fitting {method}...", flush=True)
        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(folds(args.seed).split(X_train), 1):
            model = make_model(method, args.seed)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            score = metrics(y_train.iloc[valid_idx], model.predict(X_train.iloc[valid_idx]))
            fold_metrics.append(score)
            selected = np.array(FEATURES)[model.named_steps["select"].support_].tolist()
            cv_rows.append({"method": method, "fold": fold, **score,
                            "selected_features": "; ".join(selected)})
        model = make_model(method, args.seed).fit(X_train, y_train)
        fitted[method] = model
        names = np.array(FEATURES)[model.named_steps["select"].support_].tolist()
        row = {"method": method, "n_features": len(names),
               "selected_features": "; ".join(names)}
        for key in ("MAE", "RMSE", "R2"):
            values = [m[key] for m in fold_metrics]
            row[f"cv_{key}_mean"] = np.mean(values)
            row[f"cv_{key}_std"] = np.std(values, ddof=1)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("cv_RMSE_mean")
    comparison.to_csv(out / "feature_selection_comparison.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "cv_fold_metrics.csv", index=False)
    best_method = comparison.iloc[0]["method"]
    best = fitted[best_method]
    best_features = np.array(FEATURES)[best.named_steps["select"].support_].tolist()

    print("PHASE 5: EVALUATION", flush=True)
    print(comparison[["method", "n_features", "cv_RMSE_mean"]].to_string(index=False))
    # Holdout data is first used here, after choosing the procedure by training CV.
    test_pred = best.predict(X_test)
    evaluation = {"selected_method": best_method, "selected_features": best_features,
                  "train_rows": len(X_train), "test_rows": len(X_test), "seed": args.seed,
                  "selected_holdout": metrics(y_test, test_pred),
                  "baseline_holdout": metrics(y_test, fitted["Baseline"].predict(X_test))}
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    pd.DataFrame({"row_id": X_test.index + 1, "actual": y_test,
                  "predicted": test_pred, "residual": y_test - test_pred}
                 ).to_csv(out / "test_predictions.csv", index=False)

    # VIF includes an intercept; report only the five explanatory variables.
    design = sm.add_constant(X_train, has_constant="add")
    pd.DataFrame({"feature": FEATURES,
                  "VIF": [variance_inflation_factor(design.to_numpy(), i)
                          for i in range(1, design.shape[1])]}
                 ).to_csv(out / "vif.csv", index=False)
    selected_design = sm.add_constant(X_train[best_features], has_constant="add")
    ols = sm.OLS(y_train, selected_design).fit()
    (out / "ols_summary.txt").write_text(
        "Exploratory training fit after feature selection; p-values are not selection-adjusted.\n\n"
        + ols.summary().as_text(), encoding="utf-8")
    influence = ols.get_influence()
    diagnostics = df.loc[X_train.index].copy()
    diagnostics.insert(0, "row_id", X_train.index + 1)
    diagnostics["predicted"] = ols.fittedvalues
    diagnostics["residual"] = ols.resid
    diagnostics["studentized_residual"] = influence.resid_studentized_external
    diagnostics["cooks_distance"] = influence.cooks_distance[0]
    diagnostics["leverage"] = influence.hat_matrix_diag
    top10 = diagnostics.sort_values("cooks_distance", ascending=False).head(10)
    top10.to_csv(out / "top10_outliers.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(comparison.method, comparison.cv_RMSE_mean, yerr=comparison.cv_RMSE_std,
            capsize=4, color="steelblue")
    plt.ylabel("Outer CV RMSE (mean +/- fold SD)")
    plt.title("Feature selection procedures on training data")
    save_plot(out, "feature_selection_comparison.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_pred)
    limits = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
    plt.plot(limits, limits, "--", color="gray")
    plt.xlabel("Actual profit")
    plt.ylabel("Predicted profit")
    plt.title(f"Holdout: {best_method}")
    save_plot(out, "actual_vs_predicted.png")
    plt.figure(figsize=(8, 5))
    plt.scatter(test_pred, y_test - test_pred)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Predicted profit")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Holdout residuals")
    save_plot(out, "residuals.png")
    plt.figure(figsize=(9, 5))
    plt.bar(top10.row_id.astype(str), top10.cooks_distance, color="darkorange")
    plt.xlabel("Original data row (1-based, excluding header)")
    plt.ylabel("Cook's distance")
    plt.title("Top 10 influential training observations (not automatically removed)")
    save_plot(out, "top10_outliers_plot.png")

    print("PHASE 6: DEPLOYMENT / NEW-DATA PREDICTION", flush=True)
    # Freeze the selected training feature set, then refit OLS on all available data.
    full_design = sm.add_constant(X[best_features], has_constant="add")
    full_model = sm.OLS(y, full_design).fit()
    new_startup = pd.DataFrame([{"R&D Spend": 10000, "Administration": 120000,
                                 "Marketing Spend": 150000, "State_Florida": 0,
                                 "State_New York": 0}])
    new_design = sm.add_constant(new_startup[best_features], has_constant="add")
    prediction = float(full_model.predict(new_design).iloc[0])
    artifact = {"method": best_method, "features": best_features,
                "intercept": float(full_model.params["const"]),
                "coefficients": {name: float(full_model.params[name]) for name in best_features},
                "reference_state": "California", "training_rows": len(X),
                "example_input": new_startup.iloc[0].to_dict(),
                "example_predicted_profit": prediction,
                "note": "All-data refit; holdout metrics describe the earlier training-only fit."}
    (out / "deployment_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Selected: {best_method}; features: {best_features}")
    print(f"Holdout metrics: {evaluation['selected_holdout']}")
    print(f"Predicted new startup profit: {prediction:,.2f}")
    print(f"Files saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
