"""Leakage-safe spatial-lag ("geo autoregressive") rent features.

For every listing we look at the rent per m² of *nearby buildings* and feed that to the model:

  geo_knn_s / geo_knn_l  mean of the `ks[0]` / `ks[1]` nearest buildings (defaults 5 / 20)
  geo_knn_l_std          spread of the `ks[1]` nearest (how mixed the neighbourhood is)
  geo_dist_s_km          mean distance to the `ks[0]` nearest buildings (how dense the area is)
  geo_kern               Gaussian-kernel mean (bandwidth `bandwidth_km`)
  geo_kern_density       total kernel weight (listings-per-area proxy)
  te_town                shrunk mean for the block's town (ward + town name), prior = ward mean
  te_station / te_line   shrunk mean for the nearest station (prior = ward mean) and its line
                         (prior = overall mean); only when the data has `nearest_station` / `nearest_line`

All of them are means of log(rent / m²), taken over **groups** (buildings) rather than rows so a
50-unit tower does not drown out its neighbours. To keep training rows from seeing their own
target, a training row's neighbourhood excludes its own group; held-out rows use every training
group. Fit on the training fold only.

Coordinates come from `services/geocoding.py` (geo_cache/address_coords.json). A fitted instance
serialises to a small JSON file (`<variant>-geo.json`) so the API can compute the same features
for a request that carries an `address`.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from services.geocoding import CACHE_PATH as COORDS_PATH, geocode, load_cache, normalize_address

logger = logging.getLogger(__name__)

EARTH_KM = 6371.0088
ADDRESS_COLS = ["geo_knn_s", "geo_knn_l", "geo_knn_l_std", "geo_dist_s_km", "geo_kern", "geo_kern_density", "te_town"]
STATION_COLS = ["te_station", "te_line"]
GEO_COLS = ADDRESS_COLS + STATION_COLS  # every column this module can emit
# Mostly encode the scrape cap (about 300 buildings per ward), not real density; dropped by `drop_density`
DENSITY_COLS = ["geo_dist_s_km", "geo_kern_density"]


def load_coords(path: Path = COORDS_PATH) -> Dict[str, Optional[list]]:
    return load_cache(path)


def town_of(address: str) -> str:
    """'東京都杉並区西荻南４' -> '杉並区西荻南' (drops the prefecture and the block number)."""
    return re.sub(r"[0-9０-９].*$", "", address.replace("東京都", ""))


def ward_of(address: str) -> str:
    m = re.match(r"(?:東京都)?(.+?区)", address)
    return m.group(1) if m else ""


def log_rent_per_sqm(df: pd.DataFrame) -> pd.Series:
    return np.log(df["rent_price"] / df["sqr_m"])


def structural_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Location-free features (size, age, floor, walk, fee, layout, building type) for the residual model."""
    X = pd.DataFrame({
        "log_sqm": np.log(df["sqr_m"]), "year_built": df["year_built"], "floor": df["floor"],
        "walk": df["nearest_eki_walk"], "maint": df["maintenence_price"],
    }, index=df.index)
    dummies = pd.get_dummies(df[["apartment_type", "house_type"]].astype(str), dtype=float)
    return pd.concat([X, dummies], axis=1)


def oof_residuals(train: pd.DataFrame, groups: pd.Series, n_splits: int = 5) -> pd.Series:
    """Out-of-fold residual of log(rent / m²) from a Ridge on `structural_matrix`. Folds are grouped by
    building, so a row's residual never comes from a model that saw its building. Smoothing these
    instead of raw rent per m² keeps unit mix (small studios vs family flats) out of the lag."""
    X, y = structural_matrix(train), log_rent_per_sqm(train)
    pred = pd.Series(np.nan, index=train.index)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[va] = m.predict(X.iloc[va])
    return y - pred


class GeoRentFeatures:
    """Fit on a training frame, then `transform` any frame (training rows exclude their own group)."""

    def __init__(self, coords: Dict[str, Optional[list]], bandwidth_km: float = 0.4, town_shrink: float = 5.0,
                 ks: Sequence[int] = (5, 20), station_shrink: float = 5.0, line_shrink: float = 20.0,
                 drop_density: bool = False):
        self.coords = coords
        self.drop_density = drop_density
        self.bandwidth_km = bandwidth_km
        self.town_shrink = town_shrink
        self.ks = tuple(ks)
        self.station_shrink = station_shrink
        self.line_shrink = line_shrink

    # ------------------------------------------------------------ locating addresses

    def locate(self, address: str, fallback: bool = False, live: bool = False) -> Optional[List[float]]:
        """[lat, lon] for an address. Training only uses exact (cached) matches; at inference
        `fallback` adds a live lookup (if `live`), then the town centroid, then the ward centroid."""
        hit = self.coords.get(address)
        if hit:
            return hit
        if not fallback:
            return None
        address = normalize_address(address)
        hit = self.coords.get(address)
        if hit:
            return hit
        if live:
            try:
                hit = geocode(address, timeout=3.0)
            except Exception as e:  # network down, rate limited, ...
                logger.warning("Live geocoding failed for %s: %s", address, e)
                hit = None
            self.coords[address] = hit  # memoise misses too: do not hammer the API
            if hit:
                return hit
        return self.town_centroids_.get(town_of(address)) or self.ward_centroids_.get(ward_of(address))

    def _latlon(self, df: pd.DataFrame, fallback: bool = False, live: bool = False) -> np.ndarray:
        pts = [self.locate(a, fallback, live) or [np.nan, np.nan] for a in df["address"]]
        return np.radians(np.asarray(pts, dtype=float))

    # ------------------------------------------------------------ fitting / persistence

    def fit(self, train: pd.DataFrame, groups: pd.Series, target: Optional[pd.Series] = None) -> "GeoRentFeatures":
        """`target` (aligned to `train`) replaces the default log(rent / m²), e.g. with the
        out-of-fold residual of a structural model so the lag is not confounded by unit mix."""
        t = log_rent_per_sqm(train) if target is None else target.loc[train.index]
        g = pd.DataFrame({"group": groups.astype(str).values, "t": np.asarray(t, dtype=float),
                          "address": train["address"].values})
        agg_spec = {"t": ("t", "mean"), "address": ("address", "first")}
        if {"nearest_station", "nearest_line"} <= set(train.columns):
            g["station"], g["line"] = train["nearest_station"].fillna("").values, train["nearest_line"].fillna("").values
            agg_spec.update(station=("station", "first"), line=("line", "first"))
        return self._index(g.groupby("group").agg(**agg_spec).reset_index())

    def _index(self, agg: pd.DataFrame) -> "GeoRentFeatures":
        """Build the lookup structures from a (group, t, address[, station, line]) table."""
        agg = agg.copy()
        agg["town"] = agg["address"].map(town_of)
        agg["ward"] = agg["address"].map(ward_of)
        self.has_station_ = {"station", "line"} <= set(agg.columns)
        self.columns_ = [c for c in ADDRESS_COLS + (STATION_COLS if self.has_station_ else [])
                         if not (self.drop_density and c in DENSITY_COLS)]
        self.agg_ = agg[[c for c in ("group", "t", "address", "station", "line") if c in agg.columns]]
        self.group_t_ = agg.set_index("group")["t"]
        ll = np.radians(np.asarray([self.coords.get(a) or [np.nan, np.nan] for a in agg["address"]], dtype=float))
        has_geo = ~np.isnan(ll).any(axis=1)
        self.tree_ = BallTree(ll[has_geo], metric="haversine")
        self.tree_t_ = agg["t"].to_numpy()[has_geo]
        self.tree_group_ = agg["group"].to_numpy()[has_geo]
        # sums / counts over groups, for leave-own-group-out shrunk target encodings
        keys = ["town", "ward"] + (["station", "line"] if self.has_station_ else [])
        self.sums_ = {k: agg.groupby(k)["t"].sum() for k in keys}
        self.counts_ = {k: agg.groupby(k)["t"].size() for k in keys}
        self.global_sum_, self.global_n_ = float(agg["t"].sum()), len(agg)
        # offline fallbacks for addresses the model has never seen
        deg = pd.DataFrame(np.degrees(ll[has_geo]), columns=["lat", "lon"])
        deg["town"], deg["ward"] = agg["town"].to_numpy()[has_geo], agg["ward"].to_numpy()[has_geo]
        self.town_centroids_ = {k: [float(v.lat), float(v.lon)] for k, v in deg.groupby("town")[["lat", "lon"]].mean().iterrows()}
        self.ward_centroids_ = {k: [float(v.lat), float(v.lon)] for k, v in deg.groupby("ward")[["lat", "lon"]].mean().iterrows()}
        return self

    def to_dict(self) -> dict:
        used = set(self.agg_["address"])
        return {
            "params": {"bandwidth_km": self.bandwidth_km, "town_shrink": self.town_shrink, "ks": list(self.ks),
                       "station_shrink": self.station_shrink, "line_shrink": self.line_shrink,
                       "drop_density": self.drop_density},
            "columns": list(self.agg_.columns),
            "groups": self.agg_.values.tolist(),
            "coords": {a: c for a, c in self.coords.items() if a in used and c},
        }

    @classmethod
    def from_dict(cls, state: dict) -> "GeoRentFeatures":
        obj = cls(state["coords"], **state["params"])
        return obj._index(pd.DataFrame(state["groups"], columns=state.get("columns", ["group", "t", "address"])))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "GeoRentFeatures":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------ features

    def _shrunk(self, key: str, values: pd.Series, prior, shrink: float, own_t, in_train: bool) -> np.ndarray:
        """(sum + shrink*prior) / (n + shrink) per row; NaN where the key is blank. In training the
        row's own group is removed from the sum and count."""
        s = values.map(self.sums_[key]).fillna(0.0).to_numpy()
        n = values.map(self.counts_[key]).fillna(0.0).to_numpy()
        if in_train:
            s, n = s - own_t, n - 1
        out = (s + shrink * prior) / (n + shrink)
        return np.where(values.fillna("").to_numpy() == "", np.nan, out)

    def transform(self, df: pd.DataFrame, groups: Optional[pd.Series] = None, in_train: bool = False,
                  fallback: bool = False, live: bool = False) -> pd.DataFrame:
        """Feature frame aligned to df.index. `in_train` excludes each row's own group;
        `fallback` / `live` are for inference on addresses outside the fitted set (see `locate`)."""
        out = pd.DataFrame(np.nan, index=df.index, columns=ADDRESS_COLS + STATION_COLS)
        own = groups.astype(str).to_numpy() if groups is not None else np.full(len(df), "")
        k_s, k_l = self.ks

        # --- spatial lag
        ll = self._latlon(df, fallback, live)
        ok = ~np.isnan(ll).any(axis=1)
        if ok.any():
            k = min(k_l + 1, len(self.tree_t_))
            dist, idx = self.tree_.query(ll[ok], k=k)
            dist_km = dist * EARTH_KM
            t_nb, g_nb = self.tree_t_[idx], self.tree_group_[idx]
            valid = np.ones_like(dist_km, dtype=bool)
            if in_train:
                valid = g_nb != own[ok][:, None]
            # push excluded neighbours to the end, keep the nearest k_l of what remains
            order = np.argsort(~valid, axis=1, kind="stable")
            dist_km, t_nb, valid = (np.take_along_axis(a, order, 1) for a in (dist_km, t_nb, valid))
            dist_km, t_nb, valid = dist_km[:, :k_l], t_nb[:, :k_l], valid[:, :k_l]
            tm = np.where(valid, t_nb, np.nan)
            with np.errstate(all="ignore"):
                out.loc[ok, "geo_knn_s"] = np.nanmean(tm[:, :k_s], axis=1)
                out.loc[ok, "geo_knn_l"] = np.nanmean(tm, axis=1)
                out.loc[ok, "geo_knn_l_std"] = np.nanstd(tm, axis=1)
                out.loc[ok, "geo_dist_s_km"] = np.nanmean(np.where(valid, dist_km, np.nan)[:, :k_s], axis=1)
                w = np.where(valid, np.exp(-0.5 * (dist_km / self.bandwidth_km) ** 2), 0.0)
                out.loc[ok, "geo_kern"] = (w * np.nan_to_num(t_nb)).sum(1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
                out.loc[ok, "geo_kern_density"] = w.sum(1)

        # --- shrunk target encodings (the row's own group is removed in training, priors included)
        own_t = pd.Series(own).map(self.group_t_).to_numpy() if in_train else 0.0
        ward = df["address"].map(ward_of)
        ws, wn = ward.map(self.sums_["ward"]).fillna(0.0).to_numpy(), ward.map(self.counts_["ward"]).fillna(0.0).to_numpy()
        gs, gn = self.global_sum_, self.global_n_
        if in_train:
            ws, wn, gs, gn = ws - own_t, wn - 1, gs - own_t, gn - 1
        with np.errstate(all="ignore"):
            ward_prior = np.where(wn > 0, ws / np.where(wn > 0, wn, 1), gs / gn)
        out["te_town"] = self._shrunk("town", df["address"].map(town_of), ward_prior, self.town_shrink, own_t, in_train)
        if self.has_station_ and "nearest_station" in df.columns:
            out["te_station"] = self._shrunk("station", df["nearest_station"], ward_prior,
                                             self.station_shrink, own_t, in_train)
            out["te_line"] = self._shrunk("line", df["nearest_line"], gs / gn, self.line_shrink, own_t, in_train)
        return out[self.columns_]

    def transform_address(self, address: Optional[str], station: Optional[str] = None,
                          line: Optional[str] = None) -> Dict[str, float]:
        """Features for one request. Address-based columns are NaN without a usable address,
        station-based ones NaN without a station / line."""
        has_address = bool(address and str(address).strip())
        live = os.getenv("GEO_LIVE_GEOCODE", "1") != "0"
        frame = pd.DataFrame({"address": [normalize_address(address) if has_address else ""],
                              "nearest_station": [station or ""], "nearest_line": [line or ""]})
        res = {k: float(v) for k, v in self.transform(frame, fallback=has_address, live=live and has_address).iloc[0].items()}
        if not has_address:
            res.update({c: float("nan") for c in ADDRESS_COLS if c in res})
        return res


def mask_geo_features(X: pd.DataFrame, frac: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Copy of X with the address-based and the station-based geo columns each set to NaN on a
    random share of rows, so a tree model also learns the path taken by requests without them."""
    out = X.copy()
    for i, group in enumerate((ADDRESS_COLS, STATION_COLS)):
        cols = [c for c in group if c in out.columns]
        if cols and frac > 0:
            mask = np.random.default_rng(random_state + i).random(len(out)) < frac
            out.loc[mask, cols] = np.nan
    return out


def fit_geo_features(df: pd.DataFrame, groups: pd.Series, train_index, test_index, geocode_new: bool = True,
                     residual: bool = False, **params):
    """Fit on the training rows and return (fitted, train_features, test_features).

    Rows in `df` need `address`, `rent_price`, `sqr_m` (and `nearest_station` / `nearest_line` for the
    station encodings). Training rows' features leave out their own group (building); test rows' use
    every training group. Addresses missing from the geocode cache are looked up first (network);
    rows that still cannot be located get NaN features. `residual` smooths the out-of-fold residual of a
    structural model (`oof_residuals`) instead of raw log rent per m². `params` go to `GeoRentFeatures`.
    """
    from services.geocoding import geocode_missing
    coords = load_coords()
    if geocode_new:
        coords = geocode_missing(df["address"].unique(), coords)
    train, g_train = df.loc[train_index], groups.loc[train_index]
    target = oof_residuals(train, g_train) if residual else None
    geo = GeoRentFeatures(coords, **params).fit(train, g_train, target=target)
    f_tr = geo.transform(df.loc[train_index], groups.loc[train_index], in_train=True)
    f_te = geo.transform(df.loc[test_index], groups.loc[test_index], in_train=False)
    return geo, f_tr, f_te
