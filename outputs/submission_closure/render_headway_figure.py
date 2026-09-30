"""Post-process frozen S6 data into the transportation-oriented Fig. 2.

This script does not call the vehicle simulator.  Panel (c) applies the
manuscript's q_platoon expression to frozen straight-road headway values,
the declared vehicle lengths, the 3 m standstill gap and v=25 m/s.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "S6" / "data"
FIGURE_DIR = ROOT / "figures" / "pdf"
OUT = ROOT / "outputs" / "submission_closure"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "legend.fontsize": 6.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    p1 = pd.read_csv(DATA / "S6_P1_physical_headway_boundary.csv")
    speed = pd.read_csv(DATA / "S6_speed_sensitivity.csv")
    params = json.loads((ROOT / "simulation" / "parameters_s6.json").read_text(encoding="utf-8"))

    lengths = np.asarray(params["fleet"]["body_length_m"], dtype=float)
    pair_lengths = 0.5 * (lengths[:-1] + lengths[1:])
    mean_pair_length = float(pair_lengths.mean())
    standstill_gap = float(params["road_and_operation"]["standstill_gap_m"])
    reference_speed = float(params["road_and_operation"]["reference_speed_mps"])

    straight = p1[np.isclose(p1["curvature_m_inv"], 0.0)].copy().sort_values("friction")
    straight["passage_rate_proxy_veh_s"] = reference_speed / (
        mean_pair_length + standstill_gap + straight["h_min_s"] * reference_speed
    )
    straight["normalized_passage_rate_proxy"] = (
        straight["passage_rate_proxy_veh_s"] / straight["passage_rate_proxy_veh_s"].max()
    )
    straight[
        [
            "friction",
            "curvature_m_inv",
            "h_min_s",
            "passage_rate_proxy_veh_s",
            "normalized_passage_rate_proxy",
        ]
    ].to_csv(OUT / "derived_passage_rate_proxy.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.35, 2.72),
        gridspec_kw={"width_ratios": [2.15, 0.92, 1.05]},
    )
    colors = ["#9E2A2B", "#D17C0B", "#2A9D8F", "#0072B2"]
    mus = [0.45, 0.60, 0.80, 1.00]
    max_grid = 4.0
    operational_h = 2.8

    ax = axes[0]
    for color, mu, ls in zip(colors, mus, ["-", "--", "-.", ":"]):
        rows = p1[np.isclose(p1["friction"], mu)]
        finite = rows["status"].isin(["PASS_OPERATIONAL", "PASS_EXTENDED"])
        ax.plot(rows["curvature_m_inv"], rows["h_min_s"].where(finite), color=color, linestyle=ls, label=rf"$\mu={mu:.2f}$")
        no_brake = rows[rows["status"] == "NO_POSITIVE_BRAKE_BOUND"].iloc[:1]
        outside = rows[rows["status"] == "OUTSIDE_EXTENDED_RANGE"].iloc[:1]
        if not no_brake.empty:
            ax.scatter(no_brake["curvature_m_inv"], np.full(len(no_brake), max_grid + 0.06), marker="x", s=12, color=color, alpha=0.82)
        if not outside.empty:
            ax.scatter(outside["curvature_m_inv"], np.full(len(outside), max_grid - 0.08), marker="^", s=11, color=color, alpha=0.82)
    ax.axhline(operational_h, color="#555555", linestyle="--", linewidth=0.8)
    ax.text(0.0125, operational_h + 0.05, "2.8 s reporting limit", ha="right", va="bottom", fontsize=6.1)
    ax.set(
        xlabel=r"Path curvature $|\kappa|$ (m$^{-1}$)",
        ylabel=r"Screened headway $\widehat h_{\min}^{\rm num}$ (s)",
        ylim=(0.35, max_grid + 0.16),
    )
    friction_legend = ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0.03, 1.01), handlelength=2.1, columnspacing=1.0)
    ax.add_artist(friction_legend)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="^", color="#333333", linestyle="None", markersize=3.5, label=">4.0 s"),
        Line2D([], [], marker="x", color="#333333", linestyle="None", markersize=3.5, label="Insufficient braking")
    ], frameon=False, loc="lower right", fontsize=5.8, handletextpad=0.3)
    ax.text(-0.17, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom")

    ax = axes[1]
    ax.plot(speed["speed_mps"], speed["h_min_s"], color="#0072B2", marker="o", markersize=3.4)
    for _, row in speed.iterrows():
        ax.annotate(f"{row.h_min_s:.2f}", (row.speed_mps, row.h_min_s), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2)
    ax.set(
        xlabel="Reference speed (m/s)",
        ylabel=r"$\widehat h_{\min}^{\rm num}$ (s)",
        title=r"$\mu=0.70$, $|\kappa|=0.004$ m$^{-1}$",
        xticks=[20, 25, 30],
    )
    ax.text(-0.31, 1.02, "(b)", transform=ax.transAxes, fontweight="bold", va="bottom")

    ax = axes[2]
    x = straight["friction"].to_numpy()
    y = straight["normalized_passage_rate_proxy"].to_numpy()
    ax.plot(x, y, color="#6A3D9A", marker="o", markersize=3.6, linewidth=1.25)
    ax.fill_between(x, y, 0.65, color="#6A3D9A", alpha=0.10)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2)
    ax.set(
        xlabel=r"Friction coefficient $\mu$",
        ylabel="Normalized within-platoon\npassage-rate proxy",
        title=r"Straight road, $v=25$ m/s",
        xlim=(0.41, 1.04),
        ylim=(0.65, 1.055),
        xticks=x,
    )
    ax.text(-0.27, 1.02, "(c)", transform=ax.transAxes, fontweight="bold", va="bottom")

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.79, wspace=0.52)
    fig.savefig(FIGURE_DIR / "S6_P1_transportation_headway_boundary.pdf", bbox_inches="tight", bbox_extra_artists=(friction_legend,))
    fig.savefig(OUT / "S6_P1_transportation_headway_boundary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
