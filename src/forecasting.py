"""Forecasting utilities: baselines, Holt-Winters, a direct multi-horizon gradient boosting
model with lag features, and accuracy metrics.

All functions operate on a wide weekly demand matrix ``Y`` (rows = SKU-location series,
columns = consecutive week numbers 0..T-1) plus aligned exogenous matrices (promotion
discount). Forecasts are made from an *origin* week ``o`` (last observed week) for
horizons h = 1..H, i.e. target weeks o+1..o+H, using only information available at o
(promotions are planned in advance, so their calendar is treated as known).
"""
from __future__ import annotations

import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

SEASON = 52


# ----------------------------------------------------------------------------- metrics
def wape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return np.abs(actual - forecast).sum() / actual.sum()


def mape(actual, forecast) -> float:
    """MAPE over observations with non-zero actuals."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    m = actual > 0
    return np.mean(np.abs(actual[m] - forecast[m]) / actual[m])


def bias(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return (forecast - actual).sum() / actual.sum()


# ----------------------------------------------------------------------------- baselines
def moving_average(Y: np.ndarray, origin: int, H: int, window: int = 8) -> np.ndarray:
    """Flat trailing moving average (the incumbent spreadsheet method)."""
    level = Y[:, origin - window + 1 : origin + 1].mean(axis=1, keepdims=True)
    return np.repeat(level, H, axis=1)


def seasonal_naive(Y: np.ndarray, origin: int, H: int) -> np.ndarray:
    """Same week last year, rescaled by the trailing 8-week year-over-year ratio."""
    idx = np.arange(origin + 1, origin + H + 1) - SEASON
    recent = Y[:, origin - 7 : origin + 1].sum(axis=1)
    year_ago = Y[:, origin - 7 - SEASON : origin + 1 - SEASON].sum(axis=1)
    ratio = np.clip(np.where(year_ago > 0, recent / np.maximum(year_ago, 1), 1.0), 0.5, 2.0)
    return Y[:, idx] * ratio[:, None]


def _hw_one(y: np.ndarray, H: int) -> np.ndarray:
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(
                np.log1p(y), trend="add", damped_trend=True, seasonal="add",
                seasonal_periods=SEASON, initialization_method="heuristic",
            ).fit()
            fc = np.expm1(fit.forecast(H))
        except Exception:  # pragma: no cover - fall back to moving average
            fc = np.repeat(y[-8:].mean(), H)
    return np.clip(fc, 0, None)


def holt_winters(Y: np.ndarray, origins, H: int, workers: int = 4) -> dict:
    """Damped-trend additive Holt-Winters on log1p demand, fitted per series and origin.

    Returns {origin: forecast matrix (n_series x H)}. Fits run in a short-lived process pool
    that is shut down before returning, so it does not compete with later model training.
    """
    jobs = [(o, i) for o in origins for i in range(Y.shape[0])]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_hw_one, [Y[i, : o + 1] for o, i in jobs], [H] * len(jobs), chunksize=16))
    out = {o: np.zeros((Y.shape[0], H)) for o in origins}
    for (o, i), fc in zip(jobs, res):
        out[o][i] = fc
    return out


# ----------------------------------------------------------------------------- gradient boosting
FEATURES = [
    "h", "week_of_year", "cat_code", "loc_code",
    "ma4", "ma8", "ma13", "ma26", "last", "yoy_ratio",
    "same_week_ly", "same_week_ly_smooth", "ly_growth_path",
    "promo_disc", "promo_disc_ly", "promo_weeks_recent", "log_scale",
]


def make_features(Y, P, origin, H, week_of_year, cat_code, loc_code) -> pd.DataFrame:
    """Feature frame for all series at one origin and horizons 1..H (one row per series x h)."""
    n = Y.shape[0]
    ma = {k: Y[:, origin - k + 1 : origin + 1].mean(axis=1) for k in (4, 8, 13, 26)}
    recent = Y[:, origin - 7 : origin + 1].sum(axis=1)
    year_ago = Y[:, origin - 7 - SEASON : origin + 1 - SEASON].sum(axis=1)
    yoy = np.clip(np.where(year_ago > 0, recent / np.maximum(year_ago, 1), 1.0), 0.3, 3.0)
    scale = np.maximum(ma[26], 1.0)
    frames = []
    for h in range(1, H + 1):
        t = origin + h
        ly = Y[:, t - SEASON]
        ly_s = Y[:, t - SEASON - 1 : t - SEASON + 2].mean(axis=1)
        # last year's path from origin to target, relative to last year's level at origin
        ly_path = Y[:, t - SEASON - 1 : t - SEASON + 2].mean(axis=1) / np.maximum(
            Y[:, origin - SEASON - 3 : origin - SEASON + 1].mean(axis=1), 1.0)
        frames.append(pd.DataFrame({
            "series": np.arange(n), "origin": origin, "target_week": t, "h": h,
            "week_of_year": week_of_year[t], "cat_code": cat_code, "loc_code": loc_code,
            "ma4": ma[4] / scale, "ma8": ma[8] / scale, "ma13": ma[13] / scale, "ma26": ma[26] / scale,
            "last": Y[:, origin] / scale, "yoy_ratio": yoy,
            "same_week_ly": ly / scale, "same_week_ly_smooth": ly_s / scale,
            "ly_growth_path": np.clip(ly_path, 0, 5),
            "promo_disc": P[:, t], "promo_disc_ly": P[:, t - SEASON],
            "promo_weeks_recent": (P[:, origin - 7 : origin + 1] > 0).sum(axis=1),
            "log_scale": np.log(scale), "scale": scale,
        }))
    return pd.concat(frames, ignore_index=True)


def build_training_set(Y, P, last_origin, H, woy, cat_code, loc_code, first_origin=60, step=1):
    """Stack feature rows for historical origins whose targets are all <= last_origin."""
    frames = []
    for o in range(first_origin, last_origin - H + 1, step):
        f = make_features(Y, P, o, H, woy, cat_code, loc_code)
        f["y"] = Y[f.series.values, f.target_week.values]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


class DirectGBM:
    """Global direct multi-horizon model. Target is demand divided by the series' 26-week
    level so one model can learn shape across SKUs of very different volume."""

    def __init__(self, H: int = 13, **params):
        self.H = H
        defaults = dict(loss="squared_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                        min_samples_leaf=40, l2_regularization=1.0, categorical_features=[2, 3],
                        random_state=0)
        defaults.update(params)
        self.model = HistGradientBoostingRegressor(**defaults)

    def fit(self, Y, P, origin, woy, cat_code, loc_code, first_origin=60, step=2):
        tr = build_training_set(Y, P, origin, self.H, woy, cat_code, loc_code, first_origin, step)
        # single OpenMP thread: avoids oversubscription on small shared machines, same result
        with threadpool_limits(1, user_api="openmp"):
            self.model.fit(tr[FEATURES], tr.y / tr.scale)
        self.n_train_ = len(tr)
        return self

    def predict(self, Y, P, origin, woy, cat_code, loc_code, H=None) -> np.ndarray:
        H = H or self.H
        f = make_features(Y, P, origin, H, woy, cat_code, loc_code)
        with threadpool_limits(1, user_api="openmp"):
            pred = np.clip(self.model.predict(f[FEATURES]) * f.scale.values, 0, None)
        out = np.zeros((Y.shape[0], H))
        out[f.series.values, f.h.values - 1] = pred
        return out
