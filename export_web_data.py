"""Export reproducible statistics for the browser-only regression explorer."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.model_selection import cross_val_score, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from main import ROOT, FEATURES, folds, linear_pipeline, load_data, metrics


def export(destination):
    df, X, y = load_data(ROOT / "data/50_Startups.csv")
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=42)
    models = {}
    for size in range(1, 6):
        for indices in itertools.combinations(range(5), size):
            names = [FEATURES[i] for i in indices]
            key = "-".join(map(str, indices))
            design = sm.add_constant(train_x[names], has_constant="add")
            fitted = sm.OLS(train_y, design).fit()
            test_pred = fitted.predict(sm.add_constant(test_x[names], has_constant="add"))
            cv = -cross_val_score(linear_pipeline(), train_x[names], train_y,
                                  cv=folds(), scoring="neg_root_mean_squared_error")
            full = sm.OLS(y, sm.add_constant(X[names], has_constant="add")).fit()
            ci = fitted.conf_int()
            influences = fitted.get_influence()
            order = np.argsort(influences.cooks_distance[0])[::-1][:10]
            models[key] = {
                "indices": indices, "features": names,
                "test": metrics(test_y, test_pred),
                "cv_rmse": float(cv.mean()), "cv_sd": float(cv.std(ddof=1)),
                "train_r2": float(fitted.rsquared), "adjusted_r2": float(fitted.rsquared_adj),
                "coefficients": [{"name": name, "estimate": float(fitted.params[name]),
                                  "se": float(fitted.bse[name]), "p": float(fitted.pvalues[name]),
                                  "lower": float(ci.loc[name, 0]), "upper": float(ci.loc[name, 1]),
                                  "vif": None if name == "const" else float(
                                      variance_inflation_factor(design.to_numpy(), j))}
                                 for j, name in enumerate(design.columns)],
                "predictions": [{"id": int(idx+1), "actual": float(test_y.loc[idx]),
                                 "predicted": float(test_pred.loc[idx])} for idx in test_x.index],
                "outliers": [{"id": int(train_x.index[i]+1),
                              "cook": float(influences.cooks_distance[0][i]),
                              "residual": float(fitted.resid.iloc[i])} for i in order],
                "deployment": {"coefficients": full.params.tolist(),
                               "covariance": full.cov_params().to_numpy().tolist(),
                               "variance": float(full.mse_resid),
                               "t95": float(t.ppf(.975, full.df_resid))},
            }
    methods = pd.read_csv(ROOT / "outputs/feature_selection_comparison.csv").to_dict("records")
    result = {"seed": 42, "train_rows": 40, "test_rows": 10,
              "features": FEATURES, "models": models, "methods": methods,
              "rows": [{"id": i+1, **row, "split": "test" if i in test_x.index else "train"}
                       for i, row in enumerate(df.to_dict("records"))]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("window.STARTUP_DATA = " + json.dumps(result, ensure_ascii=False,
                                                                  allow_nan=False) + ";\n",
                           encoding="utf-8")
    print(f"Exported {len(models)} models and {len(df)} observations to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "website/dist/data.js")
    export(parser.parse_args().output)
