"""Build the panel dataset used by all downstream IPO experiments.

Reads the cached daily prices in ``price_cache/`` (one CSV per ticker, produced
by ``Project/get_data.ipynb``) and the factor files in ``factor data/``, and
emits two artefacts in ``data_cache/``:

* ``sp500_filtered.csv`` — the long-format daily panel of price, return, and
  factor data for the filtered ticker universe.
* ``panel_linear.npz`` or ``panel_nonlinear.npz`` — a per-month dataset of 
  cross-sectional design matrices ``X_t``, realized next-month returns ``y_t``, 
  and shrinkage covariance estimates ``V_t``. This is what the IPO / OLS models consume.

Run from the ``Project_extension/`` directory:

    python build_dataset.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PRICE_CACHE = HERE / "price_cache"
FACTOR_DIR = HERE / "factor data"
OUT_DIR = HERE / "data_cache"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ===========================================================================
# MASTER TOGGLE: Choose your covariance estimation method
# Options: "linear" (2004 Ledoit-Wolf) or "nonlinear" (2020 Analytical)
# ===========================================================================
COV_METHOD = "linear"  


START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2024-12-31")
MIN_HISTORY_END = pd.Timestamp("2024-01-01")

# We rebalance monthly: features are computed on the last trading day of each
# month, returns are realized over the *next* month.
REBAL_FREQ = "M"

# Rolling window used to compute the sample covariance (in trading days).
COV_WINDOW = 60

# Restrict to the most-liquid ``UNIVERSE_SIZE`` names (by median dollar volume
# over the full sample). Smaller cross-sections ⇒ much faster training.
UNIVERSE_SIZE = 100


# ---------------------------------------------------------------------------
# Step 1. Build the long-format S&P 500 panel
# ---------------------------------------------------------------------------

def _load_one_ticker(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns=str.lower)
    df["ticker"] = path.stem
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def build_sp500_filtered() -> pd.DataFrame:
    """Reproduce the ``sp500_filtered.csv`` build that lives in ``get_data.ipynb``."""
    files = sorted(PRICE_CACHE.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {PRICE_CACHE}. Symlink price_cache/ first.")

    rows = []
    for f in tqdm(files, desc="reading price cache"):
        try:
            df = _load_one_ticker(f)
        except Exception as exc:
            print(f"skipping {f.stem}: {exc}")
            continue
        if df.empty:
            continue
        first, last = df["date"].min(), df["date"].max()
        if first > START_DATE:
            continue
        if last < MIN_HISTORY_END:
            continue
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel[(panel["date"] >= START_DATE) & (panel["date"] <= END_DATE)]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily simple return.
    panel["ret"] = panel.groupby("ticker")["close"].pct_change()

    # Liquidity filter
    if UNIVERSE_SIZE is not None and UNIVERSE_SIZE > 0:
        dvol = (panel["close"] * panel["volume"]).groupby(panel["ticker"]).median()
        keep = dvol.sort_values(ascending=False).head(UNIVERSE_SIZE).index
        panel = panel[panel["ticker"].isin(keep)].reset_index(drop=True)
        print(f"liquidity filter: kept {len(keep)} of {len(dvol)} tickers")
    return panel


# ---------------------------------------------------------------------------
# Step 2. Read FF5 + Mom factor files and merge onto the panel
# ---------------------------------------------------------------------------

def _read_factor_file(path: Path, skiprows: int) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=skiprows)
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\d{8}$")].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # pct -> decimal
    return df


def merge_factors(panel: pd.DataFrame) -> pd.DataFrame:
    ff5 = _read_factor_file(FACTOR_DIR / "F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=3)
    mom = _read_factor_file(FACTOR_DIR / "F-F_Momentum_Factor_daily.csv", skiprows=13)
    mom = mom.rename(columns={c: "Mom" for c in mom.columns if c != "date"})

    panel = panel.merge(ff5, on="date", how="left")
    panel = panel.merge(mom, on="date", how="left")
    return panel


# ---------------------------------------------------------------------------
# Step 3. Build cross-sectional features used as predictors
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return s - mu
    return (s - mu) / sd


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("ticker")

    panel["mom_12_2"] = (
        g["close"].apply(lambda c: c.pct_change(252).shift(21)).reset_index(level=0, drop=True)
    )
    panel["mom_1m"] = g["ret"].apply(lambda r: r.rolling(21).sum()).reset_index(level=0, drop=True)
    panel["rev_1d"] = g["ret"].shift(1)
    panel["vol_60"] = g["ret"].apply(lambda r: r.rolling(60).std()).reset_index(level=0, drop=True)
    panel["log_dollar_vol_60"] = (
        g.apply(lambda d: np.log((d["close"] * d["volume"]).rolling(60).mean()))
         .reset_index(level=0, drop=True)
    )

    return panel


def cross_sectional_standardise(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for c in feat_cols:
        out[c] = out.groupby("date")[c].transform(_zscore)
    for c in feat_cols:
        out[c] = out.groupby("date")[c].transform(lambda s: s.fillna(s.median()))
        out[c] = out[c].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Step 4. Build the per-rebalance arrays (X_t, y_t, V_t)
# ---------------------------------------------------------------------------

ASSET_FEATURES = ["mom_12_2", "mom_1m", "rev_1d", "vol_60", "log_dollar_vol_60"]
COMMON_FEATURES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# --- METHOD 1: 2004 Linear Shrinkage ---
def ledoit_wolf_shrinkage(R: np.ndarray) -> np.ndarray:
    """Linear shrinkage of the sample covariance towards a scaled identity."""
    T, n = R.shape
    R = R - R.mean(axis=0, keepdims=True)
    S = (R.T @ R) / T  # sample covariance (MLE form)
    mu = np.trace(S) / n
    F = mu * np.eye(n)

    d2 = np.linalg.norm(S - F, "fro") ** 2
    R2 = R ** 2
    pi_mat = (R2.T @ R2) / T - S ** 2
    pi_hat = pi_mat.sum()
    kappa = pi_hat / d2 if d2 > 0 else 0.0
    shrink = max(0.0, min(1.0, kappa / T))
    return shrink * F + (1 - shrink) * S

# --- METHOD 2: 2020 Analytical Nonlinear Shrinkage ---
def analytical_nonlinear_shrinkage(R: np.ndarray) -> np.ndarray:
    """Analytical nonlinear shrinkage of the sample covariance matrix."""
    n, p = R.shape
    
    R_centered = R - R.mean(axis=0, keepdims=True)
    S = (R_centered.T @ R_centered) / n
    
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    
    h_n = n ** (-1/3)
    h_n_j = eigenvalues * h_n
    
    f_tilde = np.zeros(p)
    H_f_tilde = np.zeros(p)
    
    sqrt5 = np.sqrt(5.0)
    pi = np.pi
    
    for i in range(p):
        l_i = eigenvalues[i]
        diff = l_i - eigenvalues
        
        ratio = diff / h_n_j
        sq_ratio = ratio ** 2
        
        kernel_part = np.maximum(1.0 - 0.2 * sq_ratio, 0.0)
        f_tilde[i] = np.mean((3.0 / (4.0 * sqrt5 * h_n_j)) * kernel_part)
        
        term1 = - (3.0 * diff) / (10.0 * pi * (h_n_j ** 2))
        term2_coef = (3.0 / (4.0 * sqrt5 * pi * h_n_j)) * kernel_part
        
        num = sqrt5 * h_n_j - diff
        den = sqrt5 * h_n_j + diff
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(np.abs(num / den))
        
        log_term[np.isnan(log_term)] = 0.0 
        log_term[np.isinf(log_term)] = 0.0
        
        H_f_tilde[i] = np.mean(term1 + term2_coef * log_term)
        
    c = p / n
    denominator = (pi * c * eigenvalues * f_tilde)**2 + (1.0 - c - pi * c * eigenvalues * H_f_tilde)**2
    d_tilde = eigenvalues / denominator
    d_tilde = np.maximum(d_tilde, 1e-6)
    
    S_tilde = eigenvectors @ np.diag(d_tilde) @ eigenvectors.T
    
    return S_tilde


def build_panel(panel: pd.DataFrame) -> dict:
    """Build per-month numpy arrays for the IPO/OLS pipeline."""
    panel = panel.dropna(subset=["ret"]).sort_values(["date", "ticker"])

    ret_wide = panel.pivot(index="date", columns="ticker", values="ret").sort_index()

    month_ends = (
        ret_wide.index.to_series()
        .groupby(ret_wide.index.to_period("M"))
        .last()
        .values
    )
    month_ends = pd.DatetimeIndex(month_ends)

    feat_cols = ASSET_FEATURES + COMMON_FEATURES
    panel_idx = panel.set_index(["date", "ticker"]).sort_index()

    out_dates: list[pd.Timestamp] = []
    out_tickers: list[np.ndarray] = []
    out_X: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_V: list[np.ndarray] = []

    all_dates = ret_wide.index
    me_set = set(month_ends)

    for i, t in enumerate(tqdm(month_ends, desc="rebalances")):
        if t not in me_set:
            continue
        loc = all_dates.get_loc(t)
        if loc < COV_WINDOW:
            continue
        try:
            cs = panel_idx.loc[t]
        except KeyError:
            continue
        cs = cs.dropna(subset=feat_cols)
        if len(cs) < 30: 
            continue

        next_month_ends = month_ends[i + 1] if i + 1 < len(month_ends) else None
        if next_month_ends is None:
            continue
        win = ret_wide.loc[(all_dates > t) & (all_dates <= next_month_ends)]
        next_ret = (1.0 + win).prod(axis=0) - 1.0
        next_ret = next_ret.reindex(cs.index)
        valid = next_ret.notna()
        cs = cs.loc[valid]
        next_ret = next_ret.loc[valid]
        if len(cs) < 30:
            continue

        R_window = ret_wide.iloc[loc - COV_WINDOW + 1 : loc + 1][cs.index].fillna(0.0).to_numpy()
        
        # ---------------------------------------------------
        # Apply the toggle selection for covariance shrinkage
        # ---------------------------------------------------
        if COV_METHOD == "linear":
            V = ledoit_wolf_shrinkage(R_window) * 21.0
        elif COV_METHOD == "nonlinear":
            V = analytical_nonlinear_shrinkage(R_window) * 21.0
        else:
            raise ValueError(f"Unknown COV_METHOD selected: {COV_METHOD}")

        out_dates.append(t)
        out_tickers.append(cs.index.to_numpy())
        out_X.append(cs[feat_cols].to_numpy(dtype=np.float64))
        out_y.append(next_ret.to_numpy(dtype=np.float64))
        out_V.append(V.astype(np.float64))

    return {
        "dates": np.array(out_dates),
        "tickers_per_t": out_tickers,
        "X": out_X,
        "y": out_y,
        "V": out_V,
        "feature_names": feat_cols,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_panel(panel: dict, path: Path) -> None:
    flat = {}
    flat["dates"] = panel["dates"].astype("datetime64[ns]")
    flat["feature_names"] = np.array(panel["feature_names"])
    for i, (Xi, yi, Vi, ti) in enumerate(zip(panel["X"], panel["y"], panel["V"], panel["tickers_per_t"])):
        flat[f"X_{i}"] = Xi
        flat[f"y_{i}"] = yi
        flat[f"V_{i}"] = Vi
        flat[f"t_{i}"] = ti
    np.savez_compressed(path, **flat, n=len(panel["X"]))


def load_panel(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    n = int(z["n"])
    out = {
        "dates": pd.DatetimeIndex(z["dates"]),
        "feature_names": list(z["feature_names"]),
        "X": [z[f"X_{i}"] for i in range(n)],
        "y": [z[f"y_{i}"] for i in range(n)],
        "V": [z[f"V_{i}"] for i in range(n)],
        "tickers_per_t": [z[f"t_{i}"] for i in range(n)],
    }
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"PRICE_CACHE  = {PRICE_CACHE}")
    print(f"FACTOR_DIR   = {FACTOR_DIR}")
    print(f"OUT_DIR      = {OUT_DIR}")
    print(f"COV_METHOD   = {COV_METHOD}")

    panel = build_sp500_filtered()
    panel = merge_factors(panel)
    panel = add_features(panel)
    panel = cross_sectional_standardise(panel, ASSET_FEATURES)

    out_csv = OUT_DIR / "sp500_filtered.csv"
    panel.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  shape={panel.shape}")

    print("\nbuilding per-month panel arrays...")
    arrays = build_panel(panel)
    n_months = len(arrays["X"])
    n_per_t = np.array([X.shape[0] for X in arrays["X"]])
    print(
        f"  T = {n_months} rebalances, "
        f"cross-section size: min={n_per_t.min()} median={int(np.median(n_per_t))} max={n_per_t.max()}"
    )

    # Automatically name the output file based on the selected method
    out_npz = OUT_DIR / f"panel_{COV_METHOD}.npz"
    save_panel(arrays, out_npz)
    print(f"wrote {out_npz}")


if __name__ == "__main__":
    main()