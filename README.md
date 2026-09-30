# Reproduction package

Separating physical safety from communication service in connected automated vehicle platoons: Curvature–friction headway boundaries

Authors: Jianbo Feng; Bingying Guo; Zepeng Gao; Hongbin Ren; Yanxue Wang; Zhenkun Guo. Corresponding author: Jianbo Feng.

The simulation, parameters, paired seeds and final CSV data are frozen. The closure edits affect figure presentation and submission metadata only.

Public repository: https://github.com/lanston12/cav-physical-service-headway-repro.

## Regenerate final layouts from included data

Use Python 3.9 or newer with NumPy, pandas, SciPy and Matplotlib. From this root:

```text
python outputs/submission_closure/render_headway_figure.py
python outputs/submission_closure/render_submission_figures.py
python outputs/submission_closure/render_delay_figure.py
```

These commands read frozen CSV data; they do not call the simulation. The headway renderer also writes a deterministic passage-rate proxy derived from the existing straight-road data.

## Frozen full-study implementation

`simulation/s6_simulation.py`, `simulation/run_s6.py`, `simulation/parameters_s6.json` and `random_seeds.json` preserve the study implementation. The original `simulation/render_final_figures.py` remains for archival reproducibility; use the three commands above for the corrected final layouts. To rerun the full simulation, work in a separate copy and run `python simulation/run_s6.py`; it writes result files under `outputs/S6`. No local absolute path is required.

## Evidence scope

Physical maps are numerical screens under declared assumptions, not constructed invariant certificates. Service contours use 50-run point estimates and synthetic channels with study-defined targets. Zheng et al. (2024) is a common-plant adaptation. Switching/robust predictive controls are representative implementations. The generic fixed threshold is not Xue et al. (2024).

No reuse license is included; contact the corresponding author regarding reuse rights.
