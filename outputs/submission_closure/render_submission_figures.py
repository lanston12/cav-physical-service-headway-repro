"""Render the frozen submission figures from final CSV data only.

This script does not run simulations or alter numerical results. Run it from the
reproduction-package root after ``python simulation/run_s6.py`` or against the
included final data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[2]
PARAMS = json.loads((ROOT / "simulation" / "parameters_s6.json").read_text(encoding="utf-8"))
DATA = ROOT / "outputs" / "S6" / "data"
PDF = ROOT / "figures" / "pdf"
PNG = ROOT / "figures" / "png"
PDF.mkdir(parents=True, exist_ok=True)
PNG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "CACC": "#666666",
    "Switching DMPC": "#D17C0B",
    "Robust DMPC": "#6A4C93",
    "Proposed": "#0072B2",
    "Zheng et al. (2024)": "#009E73",
}


DISPLAY_NAMES = {
    "CACC": "CACC",
    "Switching DMPC": "Switching predictive (rep.)",
    "Robust DMPC": "Robust predictive (rep.)",
    "Zheng et al. (2024)": "Zheng et al. (2024) adapt.",
    "Proposed": "Proposed",
}

def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.labelsize": 9,
        "axes.titlesize": 9.5, "legend.fontsize": 7.2, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.6,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem):
    fig.savefig(PDF / f"{stem}.pdf")
    fig.savefig(PNG / f"{stem}.png", dpi=320)
    plt.close(fig)


def render_p1():
    df = pd.read_csv(DATA / "S6_P1_physical_headway_boundary.csv")
    speed = pd.read_csv(DATA / "S6_speed_sensitivity.csv")
    mus = [0.45, 0.60, 0.80, 1.00]
    colors = ["#9E2A2B", "#D17C0B", "#2A9D8F", "#0072B2"]
    operational = PARAMS["physical_boundary"]["operational_headway_limit_s"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [2.25, 1.0]})
    ax = axes[0]
    for color, mu in zip(colors, mus):
        q = df[df.friction == mu].sort_values("curvature_m_inv")
        finite = q[q.status.isin(["PASS_OPERATIONAL", "PASS_EXTENDED"])]
        ax.plot(finite.curvature_m_inv, finite.h_min_s, color=color, label=fr"$\mu={mu:.2f}$")
        outside = q[q.status == "OUTSIDE_EXTENDED_RANGE"].iloc[::4]
        no_brake = q[q.status == "NO_POSITIVE_BRAKE_BOUND"].iloc[::4]
        if len(outside):
            ax.scatter(outside.curvature_m_inv, np.full(len(outside), 3.91), marker="^",
                       s=18, facecolors="none", edgecolors=color, linewidths=0.8, zorder=4)
        if len(no_brake):
            ax.scatter(no_brake.curvature_m_inv, np.full(len(no_brake), 4.06), marker="x",
                       s=17, color=color, linewidths=0.9, zorder=4)
    ax.axhline(operational, color="#3f3f3f", linestyle="--", linewidth=0.9)
    ax.text(df.curvature_m_inv.max(), operational + 0.04, "2.8 s operating limit",
            ha="right", va="bottom", fontsize=6.8)
    ax.set(xlabel=r"Path curvature $|\kappa|$ (m$^{-1}$)",
           ylabel=r"Screened headway $\widehat h_{\min}^{\rm num}$ (s)", ylim=(0.35, 4.16))
    color_legend = ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.add_artist(color_legend)
    status_handles = [
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="none",
               markeredgecolor="#333333", label=r"$\widehat h_{\min}^{\rm num}>4.0$ s"),
        Line2D([], [], marker="x", linestyle="None", color="#333333",
               label="Insufficient braking capability"),
    ]
    ax.legend(handles=status_handles, frameon=False, loc="lower right", fontsize=6.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(speed.speed_mps, speed.h_min_s, color="#0072B2", marker="o")
    for _, row in speed.iterrows():
        ax.annotate(f"{row.h_min_s:.2f}", (row.speed_mps, row.h_min_s),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set(xlabel="Reference speed (m/s)", ylabel=r"$\widehat h_{\min}^{\rm num}$ (s)",
           title=r"$\mu=0.70$, $|\kappa|=0.004$ m$^{-1}$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "S6_P1_physical_headway_boundary")


def render_p2():
    df = pd.read_csv(DATA / "S6_P2_curvature_friction_map.csv")
    k = np.sort(df.curvature_m_inv.unique())
    mu = np.sort(df.friction.unique())
    H = np.full((len(mu), len(k)), np.nan)
    status = np.zeros_like(H, dtype=int)
    ki, mi = {v: i for i, v in enumerate(k)}, {v: i for i, v in enumerate(mu)}
    for _, r in df.iterrows():
        a, b = mi[r.friction], ki[r.curvature_m_inv]
        if r.status in ("PASS_OPERATIONAL", "PASS_EXTENDED"):
            H[a, b] = r.h_min_s
        elif r.status == "OUTSIDE_EXTENDED_RANGE":
            status[a, b] = 1
        elif r.status == "NO_POSITIVE_BRAKE_BOUND":
            status[a, b] = 2
    max_grid = PARAMS["physical_boundary"]["headway_search_grid_s"][1]
    operational = PARAMS["physical_boundary"]["operational_headway_limit_s"]
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad((0, 0, 0, 0))
    mesh = ax.pcolormesh(k, mu, H, cmap=cmap, shading="auto", vmin=0.5, vmax=max_grid)
    ax.contourf(k, mu, np.where(status == 1, 1, np.nan), levels=[0.5, 1.5],
                colors=["#d7d7d7"], hatches=["///"], alpha=1.0)
    ax.contourf(k, mu, np.where(status == 2, 1, np.nan), levels=[0.5, 1.5],
                colors=["#777777"], hatches=["xx"], alpha=1.0)
    finite = np.where(np.isfinite(H), H, max_grid + 0.2)
    ax.contour(k, mu, finite, levels=[operational], colors="black", linewidths=2.3)
    ax.contour(k, mu, finite, levels=[operational], colors="white", linewidths=1.1)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"$\widehat h_{\min}^{\rm num}$ (s)")
    handles = [
        Patch(facecolor="#d7d7d7", edgecolor="#555555", hatch="///",
              label=r"$\widehat h_{\min}^{\rm num}>4.0$ s"),
        Patch(facecolor="#777777", edgecolor="#333333", hatch="xx",
              label="Insufficient braking capability"),
        Line2D([], [], color="black", linewidth=2.0, label="2.8 s operating limit"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", framealpha=1.0, edgecolor="#cccccc", loc="lower left", fontsize=7)
    ax.set(xlabel=r"Path curvature $|\kappa|$ (m$^{-1}$)",
           ylabel=r"Friction coefficient $\mu$",
           title="Numerically screened physical operating boundary")
    save(fig, "S6_P2_curvature_friction_headway_map")


def _edges(centres):
    centres = np.asarray(centres, dtype=float)
    mid = (centres[:-1] + centres[1:]) / 2
    return np.r_[centres[0] - (mid[0] - centres[0]), mid, centres[-1] + (centres[-1] - mid[-1])]


def render_p3():
    df = pd.read_csv(DATA / "S6_P3_service_probability.csv")
    names = ["nominal", "moderate", "severe"]
    headways = np.sort(df.headway_s.unique())
    budgets = np.sort(df.budget.unique()).astype(int)
    physical = 1.728625967609854
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True, sharey=True)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    xe, ye = _edges(headways), np.arange(-0.5, len(budgets) + 0.5, 1)
    mesh = None
    for ax, name in zip(axes, names):
        q = df[df.regime == name]
        Z = np.full((len(budgets), len(headways)), np.nan)
        for a, b in enumerate(budgets):
            for c, h in enumerate(headways):
                row = q[(q.budget == b) & (q.headway_s == h)].iloc[0]
                if bool(row.physical_feasible):
                    Z[a, c] = row.service_success_fraction
        mesh = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=0, vmax=1, shading="flat")
        passmask = np.where(np.isfinite(Z), (Z >= 0.80).astype(float), np.nan)
        if np.nanmin(passmask) <= 0.5 <= np.nanmax(passmask):
            ax.contour(headways, budgets, passmask, levels=[0.5], colors="white",
                       linewidths=1.25, corner_mask=False)
        ax.axvline(physical, color="black", linestyle="--", linewidth=1.1)
        for y in np.arange(0.5, 3.6, 1):
            ax.axhline(y, color="white", linewidth=0.45, alpha=0.8)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-0.5, 3.5)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Headway (s)")
    axes[0].set_ylabel("Attempt budget per sample")
    cb = fig.colorbar(mesh, ax=axes, pad=0.02, fraction=0.025)
    cb.set_label(r"Estimated service probability $\widehat p_{\rm svc}$")
    fig.text(0.5, 0.01,
             "Budgets are discrete rows; dashed: numerical physical boundary; white: 0.80 cell boundary; gray: physically infeasible",
             ha="center", fontsize=6.7)
    fig.subplots_adjust(bottom=0.20, wspace=0.12, right=0.88)
    save(fig, "S6_P3_physical_service_probability")


def render_p4():
    keys = ["cacc", "dmpc", "rdmpc", "zheng2024", "proposed"]
    traces = [pd.read_csv(DATA / f"S6_P4_time_history_{k}.csv") for k in keys]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for tr in traces:
        name, color = tr.method.iloc[0], COLORS[tr.method.iloc[0]]
        axes[0, 0].plot(tr.time_s, tr.gap_f1_m, label=DISPLAY_NAMES[name], color=color)
        axes[0, 1].plot(tr.time_s, tr.minimum_safe_margin_m, label=DISPLAY_NAMES[name], color=color)
        axes[1, 0].plot(tr.time_s, tr.f1_speed_mps, label=DISPLAY_NAMES[name], color=color)
    prop = traces[-1]
    axes[0, 0].plot(prop.time_s, prop.safe_gap_f1_m, color="black", linestyle="--", linewidth=1.2, label="Required gap")
    axes[1, 0].plot(prop.time_s, prop.leader_speed_mps, color="black", linestyle="--", linewidth=1.2, label="Leader")
    age_ax = axes[1, 1]
    frac_ax = age_ax.twinx()
    age_line = age_ax.plot(prop.time_s, prop.mean_packet_age_s, color="#0072B2", label="Mean packet age")[0]
    frac_line = frac_ax.plot(prop.time_s, prop.available_packet_fraction, color="#D17C0B", label="Available fraction")[0]
    for ax in axes.flat:
        ax.axvspan(10, 18, color="#bdbdbd", alpha=0.25, linewidth=0)
        ax.spines[["top", "right"]].set_visible(False)
    frac_ax.spines["top"].set_visible(False)
    axes[0, 0].set_ylabel("Follower 1 bumper gap (m)")
    axes[0, 1].set_ylabel("Minimum safety margin (m)")
    axes[1, 0].set_ylabel("Speed (m/s)")
    age_ax.set_ylabel("Mean packet age (s)", color="#0072B2")
    frac_ax.set_ylabel("Available fraction (-)", color="#D17C0B")
    age_ax.tick_params(axis="y", colors="#0072B2")
    frac_ax.tick_params(axis="y", colors="#D17C0B")
    frac_ax.set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("Time (s)")
    age_ax.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    age_ax.legend([age_line, frac_line], ["Mean packet age", "Available fraction"],
                  frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  borderaxespad=0.0, fontsize=6.7)
    axes[0, 1].text(0.50, 0.08, "shaded: complete V2V unavailability",
                    transform=axes[0, 1].transAxes, ha="center", fontsize=7.2)
    fig.subplots_adjust(top=0.84, wspace=0.34, hspace=0.24)
    save(fig, "S6_P4_emergency_braking_v2v_unavailability")


def render_p6():
    summary = pd.read_csv(DATA / "S6_P6_resource_summary.csv")
    order = ["Periodic", "Reduced rate", "Fixed event trigger", "Aware B=1", "Aware B=2", "Aware B=3"]
    markers = ["s", "D", "^", "o", "o", "o"]
    colors = ["#666666", "#D17C0B", "#009E73", "#56B4E9", "#0072B2", "#003F5C"]
    fig, ax = plt.subplots(figsize=(5.8, 3.65))

    def plot_points(target, labels=True):
        for label, marker, color in zip(order, markers, colors):
            r = summary[summary.policy == label].iloc[0]
            target.errorbar(r.packet_mean, r.score_mean, xerr=1.96 * r.packet_sem,
                            yerr=1.96 * r.score_sem, fmt=marker, color=color, capsize=2.2,
                            label=("Fixed threshold (generic)" if label == "Fixed event trigger" else label) if labels else None)
        aware = summary[summary.policy.str.startswith("Aware")].sort_values("packet_mean")
        target.plot(aware.packet_mean, aware.score_mean, color="#0072B2", alpha=0.55, linewidth=1.0)

    plot_points(ax, True)
    ax.set(xlabel="Attempted packets (vehicle-km$^{-1}$)",
           ylabel="Combined tracking score (lower is better)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), columnspacing=1.1, handletextpad=0.4)
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.14, right=0.98)
    ax.set_ylim(2.45, 3.11)
    ax.spines[["top", "right"]].set_visible(False)
    ins = inset_axes(ax, width="45%", height="36%", loc="upper right", borderpad=1.1)
    plot_points(ins, False)
    ins.set_xlim(60, 390)
    ins.set_ylim(2.472, 2.502)
    ins.set_title("High-performing policies", fontsize=6.8, pad=2)
    ins.tick_params(labelsize=6.2)
    ins.grid(color="#dddddd", linewidth=0.45)
    save(fig, "S6_P6_communication_performance_pareto")


def main():
    style()
    render_p2()
    render_p3()
    render_p4()
    render_p6()
    print("Rendered four presentation-corrected figures from frozen CSV data.")


if __name__ == "__main__":
    main()
