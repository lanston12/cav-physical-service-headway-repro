"""Re-layout the frozen delay results for a readable one-column appendix figure."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "S6" / "data" / "S6_P5_delay_sensitivity_runs.csv"
OUT = ROOT / "figures" / "pdf" / "S6_P5_delay_sensitivity_column.pdf"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)

COLORS = {
    "CACC": "#7F7F7F",
    "Switching DMPC": "#D55E00",
    "Robust DMPC": "#6A3D9A",
    "Zheng et al. (2024)": "#009E73",
    "Proposed": "#0072B2",
}


DISPLAY_NAMES = {
    "CACC": "CACC",
    "Switching DMPC": "Switching predictive (rep.)",
    "Robust DMPC": "Robust predictive (rep.)",
    "Zheng et al. (2024)": "Zheng et al. (2024) adapt.",
    "Proposed": "Proposed",
}

def main() -> None:
    df = pd.read_csv(DATA)
    specs = [
        ("gap_rmse_m", "Gap RMSE (m)"),
        ("speed_rmse_mps", "Speed RMSE (m/s)"),
        ("minimum_safe_margin_m", "Minimum safety margin (m)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.75), sharex=True)
    for ax, (metric, ylabel) in zip(axes, specs):
        for name, color in COLORS.items():
            rows = df[df["method"] == name].groupby("maximum_delay_s")[metric].agg(["mean", "std", "count"]).reset_index()
            ci = 1.96 * rows["std"] / np.sqrt(rows["count"])
            ax.plot(rows["maximum_delay_s"], rows["mean"], label=DISPLAY_NAMES[name], color=color, linewidth=1.1)
            ax.fill_between(rows["maximum_delay_s"], rows["mean"] - ci, rows["mean"] + ci, color=color, alpha=0.10, linewidth=0)
        if metric == "minimum_safe_margin_m":
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Maximum delay (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.005), columnspacing=1.0)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.99, bottom=0.21, hspace=0.20)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
