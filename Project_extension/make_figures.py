"""Generate the figures and tables that go into the project write-up.

Reads the artefacts produced by ``evaluate.py`` from ``results/`` and writes
PNGs / PDFs into ``figures/``. Each figure corresponds to a paragraph in the
report:

1. ``figures/cum_returns.png``      — cumulative-return curves per model
2. ``figures/cost_bar.png``         — mean realized MVO cost (bar chart)
3. ``figures/sharpe_bar.png``       — annualized Sharpe ratio (bar chart)
4. ``figures/cost_vs_r2.png``       — scatter of mean cost against predictive R²
5. ``figures/weight_l1_box.png``    — box plot of per-month gross exposure
6. ``figures/dominance_heatmap.png``— bootstrap dominance probability vs OLS-linear

All plots use a single matplotlib import to keep the dependency footprint
small. Colour assignment is deterministic so the report's narrative can refer
to specific colours without ambiguity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

# Order columns so OLS group plots first, then IPO group, in the same family
# order. This keeps the legend reading "OLS-linear, OLS-ridge, ... IPO-linear,
# IPO-ridge, ..." which is what the report references.
FAMILY_ORDER = ["linear", "ridge", "polynomial", "kernel", "nn"]
PARADIGM_ORDER = ["OLS", "IPO"]


def _ordered(cols: list[str]) -> list[str]:
    rank = {f"{p}_{f}": (i, j) for i, p in enumerate(PARADIGM_ORDER)
            for j, f in enumerate(FAMILY_ORDER)}
    return sorted(cols, key=lambda c: rank.get(c, (99, 99)))


def _colour(name: str) -> str:
    """Stable colour per family, lighter shade for OLS, darker for IPO."""
    base = {
        "linear":     ("#a6cee3", "#1f78b4"),
        "ridge":      ("#fdbf6f", "#ff7f00"),
        "polynomial": ("#b2df8a", "#33a02c"),
        "kernel":     ("#cab2d6", "#6a3d9a"),
        "nn":         ("#fb9a99", "#e31a1c"),
    }
    paradigm, family = name.split("_", 1)
    light, dark = base.get(family, ("#cccccc", "#666666"))
    return light if paradigm == "OLS" else dark


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def cumulative_returns(per_ret: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = _ordered(per_ret.columns.tolist())
    for c in cols:
        eq = (1.0 + per_ret[c]).cumprod()
        ax.plot(per_ret.index, eq, label=c, color=_colour(c), lw=1.4)
    ax.set_title("Out-of-sample cumulative growth (test period)")
    ax.set_ylabel("growth of \\$1")
    ax.set_xlabel("date")
    ax.axhline(1.0, color="black", lw=0.5, ls="--")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = FIGURES / "cum_returns.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def bar_metric(perf: pd.DataFrame, col: str, title: str, fname: str,
               better: str = "lower") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = _ordered(perf.index.tolist())
    vals = perf.loc[cols, col].to_numpy()
    colours = [_colour(c) for c in cols]
    ax.bar(range(len(cols)), vals, color=colours)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylabel(col + (" (lower is better)" if better == "lower" else " (higher is better)"))
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.4)
    fig.tight_layout()
    out = FIGURES / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def cost_vs_r2(perf: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    cols = _ordered(perf.index.tolist())
    for c in cols:
        ax.scatter(perf.loc[c, "r2_mean"], perf.loc[c, "mean_cost"],
                   color=_colour(c), s=70,
                   marker="o" if c.startswith("OLS") else "^",
                   label=c)
        ax.annotate(c, (perf.loc[c, "r2_mean"], perf.loc[c, "mean_cost"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("mean cross-sectional R² (predictive)")
    ax.set_ylabel("realized MVO cost (lower is better)")
    ax.set_title("Prediction quality is not decision quality")
    fig.tight_layout()
    out = FIGURES / "cost_vs_r2.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def weight_l1_box(per_l1: pd.DataFrame | None) -> Path | None:
    """Per-month |z|_1 distribution. ``per_l1`` not currently persisted by
    ``evaluate.py``; we fall back to displaying the means from ``performance.csv``
    if the per-month table is unavailable.
    """
    if per_l1 is None or per_l1.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = _ordered(per_l1.columns.tolist())
    data = [per_l1[c].dropna().to_numpy() for c in cols]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(_colour(c))
        patch.set_alpha(0.85)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylabel("Σ |z_i|  per rebalance (gross exposure)")
    ax.set_title("Portfolio gross-exposure stability")
    fig.tight_layout()
    out = FIGURES / "weight_l1_box.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def dominance_heatmap(perf: pd.DataFrame) -> Path | None:
    """Bootstrap dominance probability against OLS-linear, as a horizontal bar.

    A value > 0.5 means the candidate beats the OLS-linear baseline more often
    than chance in the bootstrap.
    """
    if "prob_dominates" not in perf.columns:
        return None
    cols = _ordered(perf.index.tolist())
    vals = perf.loc[cols, "prob_dominates"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = [_colour(c) for c in cols]
    ax.barh(range(len(cols)), vals, color=colours)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.axvline(0.5, color="black", lw=0.6, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(model has lower MVO cost than OLS-linear, bootstrap)")
    ax.set_title("Bootstrap dominance vs OLS-linear baseline")
    fig.tight_layout()
    out = FIGURES / "dominance_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    perf = pd.read_csv(RESULTS / "performance.csv", index_col="model")
    per_ret = pd.read_csv(RESULTS / "per_month.csv", index_col="date", parse_dates=True)

    paths = []
    paths.append(cumulative_returns(per_ret))
    paths.append(bar_metric(perf, "mean_cost",
                            "Realized MVO cost (test period)",
                            "cost_bar.png", better="lower"))
    paths.append(bar_metric(perf, "sharpe",
                            "Annualized Sharpe ratio (test period)",
                            "sharpe_bar.png", better="higher"))
    paths.append(cost_vs_r2(perf))

    dom = dominance_heatmap(perf)
    if dom is not None:
        paths.append(dom)

    for p in paths:
        if p is not None:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
