"""Reproducible S6 evidence-closure simulations for the TR-C manuscript.

Evidence levels are kept separate:
1. analytical transient stopping-screen calculations;
2. Sobol/corner numerical screening over the declared parameter set;
3. stochastic closed-loop simulations under synthetic communication.

The B1--B3 controllers and Zheng et al. adaptation are transparent common-plant
implementations, not claims of bit-for-bit reproduction of author code.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT / "simulation" / "parameters_s6.json"
OUT = ROOT / "outputs" / "S6"
DATA = OUT / "data"
FIG_PDF = ROOT / "figures" / "pdf"
FIG_PNG = ROOT / "figures" / "png"
G = 9.81


def load_parameters():
    with PARAM_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


P = load_parameters()


def setup_dirs():
    for p in (OUT, DATA, FIG_PDF, FIG_PNG, OUT / "scripts"):
        p.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    shutil.copy2(PARAM_PATH, OUT / "scripts" / PARAM_PATH.name)
    runner = ROOT / "simulation" / "run_s6.py"
    if runner.exists():
        shutil.copy2(runner, OUT / "scripts" / runner.name)


def set_plot_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


COLORS = {
    "CACC": "#777777",
    "Switching DMPC": "#D17C0B",
    "Robust DMPC": "#6A4C93",
    "Proposed": "#0072B2",
    "Zheng et al. (2024)": "#009E73",
}


def save_figure(fig, stem):
    fig.savefig(FIG_PDF / f"{stem}.pdf")
    fig.savefig(FIG_PNG / f"{stem}.png", dpi=320)
    plt.close(fig)


def follower_travel(t, v, reaction, brake):
    if t <= reaction:
        return v * t
    tb = t - reaction
    stop = v / brake
    if tb <= stop:
        return v * reaction + v * tb - 0.5 * brake * tb * tb
    return v * reaction + v * v / (2.0 * brake)


def predecessor_travel(t, v, brake):
    stop = v / brake
    if t <= stop:
        return v * t - 0.5 * brake * t * t
    return v * v / (2.0 * brake)


def transient_closure(vf, vp, reaction, bf, bp):
    """Exact maximum of the manuscript's piecewise-quadratic travel difference."""
    if bf <= 0 or bp <= 0:
        return math.inf
    breaks = sorted(set([0.0, reaction, vp / bp, reaction + vf / bf]))
    candidates = list(breaks)
    for left, right in zip(breaks[:-1], breaks[1:]):
        mid = 0.5 * (left + right)
        af = 0.0 if mid < reaction or mid > reaction + vf / bf else -bf
        ap = -bp if mid < vp / bp else 0.0
        vf_left = vf if left <= reaction else max(0.0, vf - bf * (left - reaction))
        vp_left = max(0.0, vp - bp * left)
        slope = af - ap
        if abs(slope) > 1e-12:
            root = left - (vf_left - vp_left) / slope
            if left <= root <= right:
                candidates.append(root)
    values = [follower_travel(t, vf, reaction, bf) - predecessor_travel(t, vp, bp) for t in candidates]
    return max(0.0, max(values))


def remaining_longitudinal_accel(mu, curvature, speed, lateral_extra=0.0):
    demand = abs(speed * speed * curvature) + lateral_extra
    total = mu * G
    if demand >= total:
        return 0.0
    return math.sqrt(max(0.0, total * total - demand * demand))


def body_projection_extra(length_f, width_f, length_p, width_p, curvature, heading_bound):
    def projection(length, width):
        straight = 0.5 * length
        rotated = 0.5 * length * math.cos(heading_bound) + 0.5 * width * math.sin(heading_bound)
        sagitta = 0.5 * abs(curvature) * (0.5 * math.hypot(length, width)) ** 2
        return max(straight, rotated + sagitta) - straight

    return projection(length_f, width_f) + projection(length_p, width_p)


def headway_sample(mu, curvature, vf, vp, tau, eff_f, eff_p, pair, sensor_error):
    phys = P["physical_boundary"]
    local = P["local_sensor"]
    road = P["road_and_operation"]
    remaining_f = remaining_longitudinal_accel(mu, curvature, vf, phys["lateral_disturbance_bound_mps2"])
    remaining_p = remaining_longitudinal_accel(mu, curvature, vp, phys["lateral_disturbance_bound_mps2"])
    bf = eff_f * remaining_f - phys["longitudinal_disturbance_bound_mps2"]
    bp = eff_p * remaining_p + 0.10
    if bf <= 0.05 or bp <= 0.05:
        return math.inf, bf, bp
    reaction = (
        local["update_period_s"]
        + local["latency_s"]
        + phys["controller_deadline_allowance_s"]
        + phys["actuator_settling_multiplier"] * tau
    )
    closure = transient_closure(vf, vp, reaction, bf, bp)
    fleet = P["fleet"]
    i, j = pair
    projection = body_projection_extra(
        fleet["body_length_m"][i],
        fleet["body_width_m"][i],
        fleet["body_length_m"][j],
        fleet["body_width_m"][j],
        curvature,
        phys["heading_error_bound_rad"],
    )
    required_gap = road["standstill_gap_m"] + closure + sensor_error + projection
    return max(0.0, (required_gap - road["standstill_gap_m"]) / max(vf, 0.1)), bf, bp


def robust_headway(mu, curvature, speed=25.0, samples=None, seed=17):
    """Worst corner/Sobol headway for the declared numerical screening set."""
    if samples is None:
        samples = P["physical_boundary"]["space_filling_samples_per_cell"]
    fleet = P["fleet"]
    sensor_bound = P["local_sensor"]["range_error_bound_m"]
    tests = []
    # Explicit adverse corners for every adjacent vehicle pair.
    for pair in [(i, i - 1) for i in range(1, P["fleet"]["vehicle_count"])]:
        tests.append((speed + 1.0, speed - 1.0, max(fleet["actuator_time_constant_s"]), 0.84, 0.99, pair, sensor_bound))
    # Reproducible Sobol space-filling samples for numerical falsification.
    power = int(math.ceil(math.log2(max(1, samples))))
    unit = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(power)[:samples]
    for u in unit:
        i = min(P["fleet"]["vehicle_count"] - 1, 1 + int(u[5] * (P["fleet"]["vehicle_count"] - 1)))
        tests.append(
            (
                speed - 1.0 + 2.0 * u[0],
                speed - 1.0 + 2.0 * u[1],
                0.18 + 0.10 * u[2],
                0.84 + 0.08 * u[3],
                0.94 + 0.05 * u[4],
                (i, i - 1),
                sensor_bound * u[6],
            )
        )
    worst = (-math.inf, math.nan, math.nan)
    for vf, vp, tau, ef, ep, pair, se in tests:
        h, bf, bp = headway_sample(mu, curvature, vf, vp, tau, ef, ep, pair, se)
        if h > worst[0]:
            worst = (h, bf, bp)
    h, bf, bp = worst
    max_h = P["physical_boundary"]["headway_search_grid_s"][1]
    operational_h = P["physical_boundary"]["operational_headway_limit_s"]
    if not math.isfinite(h):
        status = "NO_POSITIVE_BRAKE_BOUND"
    elif h <= operational_h:
        status = "PASS_OPERATIONAL"
    elif h <= max_h:
        status = "PASS_EXTENDED"
    else:
        status = "OUTSIDE_EXTENDED_RANGE"
    return {"h_min_s": h, "follower_brake_mps2": bf, "predecessor_brake_mps2": bp, "status": status}


@lru_cache(maxsize=256)
def lqr_gain(tau_rounded, headway_rounded):
    tau = float(tau_rounded)
    h = float(headway_rounded)
    dt = P["road_and_operation"]["simulation_step_s"]
    A = np.array([[1.0, dt, -h * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0 - dt / tau]])
    B = np.array([[0.0], [0.0], [dt / tau]])
    Q = np.diag(P["controller"]["lqr_state_weights"])
    R = np.array([[P["controller"]["lqr_input_weight"]]])
    X = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ X @ B + R, B.T @ X @ A)
    return K.ravel()


@lru_cache(maxsize=32)
def lateral_lqr_gain(vehicle_index):
    """Discrete LQR gain for the linear bicycle error model at 25 m/s."""
    i = int(vehicle_index)
    fleet = P["fleet"]
    v = P["road_and_operation"]["reference_speed_mps"]
    dt = P["road_and_operation"]["simulation_step_s"] / 5.0
    m = fleet["mass_kg"][i]
    iz = fleet["yaw_inertia_kg_m2"][i]
    wheelbase = fleet["wheelbase_m"][i]
    lf = fleet["front_axle_fraction"][i] * wheelbase
    lr = wheelbase - lf
    cf = fleet["front_cornering_stiffness_N_rad"][i]
    cr = fleet["rear_cornering_stiffness_N_rad"][i]
    A = np.array(
        [
            [0.0, v, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(cf + cr) / (m * v), (-lf * cf + lr * cr) / (m * v) - v],
            [0.0, 0.0, (-lf * cf + lr * cr) / (iz * v), -(lf * lf * cf + lr * lr * cr) / (iz * v)],
        ]
    )
    B = np.array([[0.0], [0.0], [cf / m], [lf * cf / iz]])
    Ad = np.eye(4) + dt * A
    Bd = dt * B
    Q = np.diag([1.0, 6.0, 0.2, 0.5])
    R = np.array([[5.0]])
    X = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(Bd.T @ X @ Bd + R, Bd.T @ X @ Ad).ravel()


def channel_events(nsteps, nlinks, loss, persistence, max_delay, seed):
    rng = np.random.default_rng(seed)
    bad = np.zeros((nsteps, nlinks), dtype=bool)
    delays = rng.integers(0, max_delay + 1, size=(nsteps, nlinks)) if max_delay > 0 else np.zeros((nsteps, nlinks), dtype=int)
    p_bb = persistence + (1.0 - persistence) * loss
    p_gb = (1.0 - persistence) * loss
    state = rng.random(nlinks) < loss
    for k in range(nsteps):
        bad[k] = state
        draw = rng.random(nlinks)
        state = np.where(state, draw < p_bb, draw < p_gb)
    return bad, delays


def road_state(scenario, t):
    if scenario in ("emergency", "ablation"):
        curvature = 0.0045 * (0.5 + 0.5 * math.tanh((t - 5.0) / 1.5))
        mu = 0.62 if 9.0 <= t <= 22.0 else 0.78
    elif scenario == "near_boundary":
        curvature = 0.006
        mu = 0.60
    elif scenario == "service":
        curvature = 0.004
        mu = 0.70
    else:
        curvature = 0.004 * math.sin(0.06 * t) ** 2
        mu = 0.78
    return curvature, mu


def leader_command(scenario, t):
    if scenario in ("emergency", "ablation", "near_boundary"):
        if 12.0 <= t < 14.2:
            return -5.2
        if 20.0 <= t < 23.0:
            return 1.4
        return 0.0
    return 0.65 * math.sin(0.34 * t) + 0.25 * math.sin(0.83 * t)


def controller_name(method):
    return {
        "cacc": "CACC",
        "dmpc": "Switching DMPC",
        "rdmpc": "Robust DMPC",
        "proposed": "Proposed",
        "zheng2024": "Zheng et al. (2024)",
    }[method]


def online_safe_gap(i, vf, vp, mu, curvature, coupling=True):
    fleet = P["fleet"]
    phys = P["physical_boundary"]
    local = P["local_sensor"]
    kappa = curvature if coupling else 0.0
    assumed_mu = mu if coupling else P["controller"]["robust_baseline_assumed_mu"]
    remaining_f = remaining_longitudinal_accel(assumed_mu, kappa, vf, phys["lateral_disturbance_bound_mps2"])
    remaining_p = remaining_longitudinal_accel(assumed_mu, kappa, vp, phys["lateral_disturbance_bound_mps2"])
    bf = max(0.05, fleet["brake_effectiveness"][i] * remaining_f - phys["longitudinal_disturbance_bound_mps2"])
    bp = max(0.05, min(0.99, fleet["brake_effectiveness"][i - 1] + 0.05) * remaining_p + 0.10)
    reaction = local["update_period_s"] + local["latency_s"] + phys["controller_deadline_allowance_s"] + phys["actuator_settling_multiplier"] * fleet["actuator_time_constant_s"][i]
    closure = transient_closure(vf, vp, reaction, bf, bp)
    projection = body_projection_extra(
        fleet["body_length_m"][i], fleet["body_width_m"][i], fleet["body_length_m"][i - 1], fleet["body_width_m"][i - 1], kappa, phys["heading_error_bound_rad"]
    )
    return P["road_and_operation"]["standstill_gap_m"] + closure + local["range_error_bound_m"] + projection, bf


def simulate(method="proposed", scenario="service", comm=None, headway=1.4, budget=2, seed=2101, strategy=None, ablation=None, gain_headway_override=None):
    if comm is None:
        comm = P["communication"]["moderate"]
    dt = P["road_and_operation"]["simulation_step_s"]
    duration = P["road_and_operation"]["episode_duration_s"]
    nsteps = int(round(duration / dt)) + 1
    nveh = P["fleet"]["vehicle_count"]
    nlinks = nveh - 1
    fleet = P["fleet"]
    ctrl = P["controller"]
    local = P["local_sensor"]
    d0 = P["road_and_operation"]["standstill_gap_m"]
    v0 = P["road_and_operation"]["reference_speed_mps"]
    if strategy is None:
        strategy = "aware" if method == "proposed" else "periodic"

    s = np.zeros(nveh)
    for i in range(1, nveh):
        pair_length = 0.5 * (fleet["body_length_m"][i - 1] + fleet["body_length_m"][i])
        s[i] = s[i - 1] - pair_length - d0 - headway * v0
    s_initial = s.copy()
    v = np.full(nveh, v0)
    if scenario == "near_boundary":
        # Exercise the declared adverse relative-speed corner on the pair with
        # the slowest actuator, while retaining the same nominal spacing.
        v[2] = v0 - 1.0
        v[3] = v0 + 1.0
    acc = np.zeros(nveh)
    cmd_prev = np.zeros(nveh)
    ey = np.zeros(nveh)
    epsi = np.zeros(nveh)
    vy = np.zeros(nveh)
    yaw = np.zeros(nveh)

    bad, delays = channel_events(nsteps, nlinks, comm["loss_probability"], comm["burst_persistence"], comm["maximum_delay_steps"], seed + 30000)
    sensor_rng = np.random.default_rng(seed + 10000)
    dist_rng = np.random.default_rng(seed + 20000)
    range_noise = sensor_rng.uniform(-local["range_error_bound_m"], local["range_error_bound_m"], size=(nsteps, nlinks))
    relv_noise = sensor_rng.uniform(-local["relative_speed_error_bound_mps"], local["relative_speed_error_bound_mps"], size=(nsteps, nlinks))
    disturbances = np.clip(dist_rng.normal(0.0, 0.035, size=(nsteps, nveh)), -0.10, 0.10)

    last_msg_a = np.zeros(nlinks)
    last_msg_u = np.zeros(nlinks)
    last_msg_k = np.zeros(nlinks, dtype=int)
    queue = []
    attempted = 0
    delivered = 0
    occupied = 0
    computation_ms = []
    records = []
    pair_errors = []
    speed_errors = []
    lateral_errors = []
    jerk_values = []
    friction_reserves = []

    outage_start, outage_end = P["communication"]["representative_outage_s"]

    for k in range(nsteps):
        t = k * dt
        curvature, mu = road_state(scenario, t)
        gaps = np.array([s[i - 1] - s[i] - 0.5 * (fleet["body_length_m"][i - 1] + fleet["body_length_m"][i]) for i in range(1, nveh)])
        relv = v[:-1] - v[1:]
        measured_gaps = gaps + range_noise[k]
        measured_relv = relv + relv_noise[k]

        # Communication attempts are chosen before the control update.
        ages = (k - last_msg_k) * dt
        if strategy == "periodic":
            attempt_mask = np.ones(nlinks, dtype=bool)
        elif strategy == "reduced":
            attempt_mask = np.full(nlinks, k % 4 == 0, dtype=bool)
        elif strategy == "event":
            attempt_mask = (
                (np.abs(acc[:-1] - last_msg_a) >= ctrl["event_acceleration_threshold_mps2"])
                | (ages >= ctrl["event_max_silence_s"])
            )
        else:
            scores = 1.2 * ages + 1.8 * np.abs(acc[:-1] - last_msg_a) + 0.35 * np.maximum(0.0, 2.0 - gaps)
            attempt_mask = np.zeros(nlinks, dtype=bool)
            candidates = np.lexsort((np.arange(nlinks), -scores))
            used = 0
            for j in candidates:
                if used >= max(0, budget):
                    break
                if scores[j] > 0.16 or ages[j] >= 0.4:
                    attempt_mask[j] = True
                    used += 1
        attempted += int(attempt_mask.sum())
        occupied += int(attempt_mask.sum())
        explicit_outage = scenario in ("emergency", "ablation", "near_boundary") and outage_start <= t <= outage_end
        for j in range(nlinks):
            if not attempt_mask[j] or explicit_outage or bad[k, j]:
                continue
            arrival = min(nsteps - 1, k + int(delays[k, j]))
            queue.append((arrival, j, float(acc[j]), float(cmd_prev[j]), k))
        remaining_queue = []
        for arrival, j, a_msg, u_msg, gen_k in queue:
            if arrival <= k:
                if gen_k >= last_msg_k[j]:
                    last_msg_a[j] = a_msg
                    last_msg_u[j] = u_msg
                    last_msg_k[j] = gen_k
                    delivered += 1
            else:
                remaining_queue.append((arrival, j, a_msg, u_msg, gen_k))
        queue = remaining_queue

        tic = time.perf_counter()
        commands = np.zeros(nveh)
        commands[0] = leader_command(scenario, t)
        safe_gaps = np.zeros(nlinks)
        margins = np.zeros(nlinks)
        packet_age = (k - last_msg_k) * dt
        for j in range(nlinks):
            i = j + 1
            age = packet_age[j]
            if age <= ctrl["maximum_packet_age_s"]:
                if ablation == "no_timestamp":
                    a_est = last_msg_a[j]
                else:
                    tau_p = fleet["actuator_time_constant_s"][j]
                    a_est = last_msg_u[j] + (last_msg_a[j] - last_msg_u[j]) * math.exp(-age / tau_p)
            else:
                a_est = 0.0
            desired = d0 + headway * v[i]
            e_gap = measured_gaps[j] - desired
            xerr = np.array([e_gap, measured_relv[j], acc[i]])
            if method == "cacc":
                u = ctrl["cacc_kp"] * e_gap + ctrl["cacc_kd"] * measured_relv[j] + ctrl["cacc_feedforward"] * a_est
            elif method == "zheng2024":
                k1 = ctrl["zheng_k1"]
                k2 = ctrl["zheng_k2"]
                local_term = k1 * e_gap + k2 * (measured_relv[j] - headway * acc[i]) + acc[i]
                q_term = (a_est - acc[i]) if age <= ctrl["maximum_packet_age_s"] else 0.0
                u = local_term + q_term
            else:
                gain_headway = headway if gain_headway_override is None else gain_headway_override
                K = lqr_gain(round(fleet["actuator_time_constant_s"][i], 3), round(gain_headway, 3))
                u = -float(K @ xerr) + a_est
                if method == "dmpc" and age > ctrl["maximum_packet_age_s"]:
                    u -= 0.10 * acc[i]

            actual_safe, actual_bf = online_safe_gap(i, v[i], v[i - 1], mu, curvature, coupling=True)
            safe_gaps[j] = actual_safe
            margins[j] = gaps[j] - actual_safe
            if method in ("rdmpc", "proposed") and ablation != "no_tightening":
                coupling = method == "proposed" and ablation != "no_coupling"
                filter_gap, filter_bf = online_safe_gap(i, v[i], v[i - 1], mu, curvature, coupling=coupling)
                filter_margin = measured_gaps[j] - filter_gap
                sensitivity = max(0.5, local["update_period_s"] + local["latency_s"] + v[i] / max(filter_bf, 0.2))
                u_cap = (measured_relv[j] + ctrl["safety_filter_alpha"] * filter_margin) / sensitivity
                u = min(u, u_cap)
            if measured_gaps[j] < 1.0:  # common independent emergency fallback
                u = ctrl["acceleration_min_mps2"]
            commands[i] = float(np.clip(u, ctrl["acceleration_min_mps2"], ctrl["acceleration_max_mps2"]))
        computation_ms.append((time.perf_counter() - tic) * 1000.0)

        acc_before_step = acc.copy()
        # Common nonlinear longitudinal/dynamic-bicycle plant. Five substeps
        # resolve the fast tire dynamics; path progress uses path speed, while
        # lateral force still reduces the friction available for braking.
        nsub = 5
        dts = dt / nsub
        step_reserve = 1.0
        for _ in range(nsub):
            for i in range(nveh):
                wheelbase = fleet["wheelbase_m"][i]
                lf = fleet["front_axle_fraction"][i] * wheelbase
                lr = wheelbase - lf
                delta_ff = math.atan(wheelbase * curvature)
                lateral_state = np.array([ey[i], epsi[i], vy[i], yaw[i] - v[i] * curvature])
                delta = delta_ff - float(lateral_lqr_gain(i) @ lateral_state)
                delta = float(np.clip(delta, -ctrl["steering_limit_rad"], ctrl["steering_limit_rad"]))
                vx = max(v[i], 5.0)
                alpha_f = delta - (vy[i] + lf * yaw[i]) / vx
                alpha_r = -(vy[i] - lr * yaw[i]) / vx
                fyf = fleet["front_cornering_stiffness_N_rad"][i] * alpha_f
                fyr = fleet["rear_cornering_stiffness_N_rad"][i] * alpha_r
                lateral_limit = mu * fleet["mass_kg"][i] * G * 0.98
                total_fy = fyf + fyr
                if abs(total_fy) > lateral_limit:
                    scale = lateral_limit / abs(total_fy)
                    fyf *= scale
                    fyr *= scale
                    total_fy = fyf + fyr
                long_force_limit = math.sqrt(max(0.0, (mu * fleet["mass_kg"][i] * G) ** 2 - total_fy ** 2))
                long_accel_limit = fleet["brake_effectiveness"][i] * long_force_limit / fleet["mass_kg"][i]
                target = float(np.clip(commands[i], -long_accel_limit, min(ctrl["acceleration_max_mps2"], long_accel_limit)))
                longitudinal_tire_force = abs(target) * fleet["mass_kg"][i] / max(fleet["brake_effectiveness"][i], 1e-9)
                capacity = mu * fleet["mass_kg"][i] * G
                reserve = max(0.0, 1.0 - math.hypot(total_fy, longitudinal_tire_force) / max(capacity, 1e-9))
                step_reserve = min(step_reserve, reserve)
                acc_dot = (target - acc[i]) / fleet["actuator_time_constant_s"][i]
                vy_dot = total_fy / fleet["mass_kg"][i] - v[i] * yaw[i]
                yaw_dot = (lf * fyf - lr * fyr) / fleet["yaw_inertia_kg_m2"][i]
                s_dot = v[i]
                ey_dot = vy[i] + v[i] * epsi[i]
                epsi_dot = yaw[i] - curvature * v[i]
                acc[i] += dts * acc_dot
                v[i] = max(0.0, v[i] + dts * (acc[i] + disturbances[k, i]))
                s[i] += dts * s_dot
                ey[i] += dts * ey_dot
                epsi[i] += dts * epsi_dot
                vy[i] += dts * vy_dot
                yaw[i] += dts * yaw_dot
        friction_reserves.append(step_reserve)
        cmd_prev = commands.copy()

        if k * dt >= 5.0:
            pair_errors.extend((gaps - (d0 + headway * v[1:])).tolist())
            speed_errors.extend((v[1:] - v[0]).tolist())
            lateral_errors.extend(ey[1:].tolist())
            jerk_values.extend(((acc[1:] - acc_before_step[1:]) / dt).tolist())
        records.append(
            {
                "time_s": t,
                "method": controller_name(method),
                "gap_f1_m": gaps[0],
                "minimum_gap_m": float(np.min(gaps)),
                "safe_gap_f1_m": safe_gaps[0],
                "minimum_safe_margin_m": float(np.min(margins)),
                "leader_speed_mps": v[0],
                "f1_speed_mps": v[1],
                "mean_packet_age_s": float(np.mean(np.minimum(packet_age, 2.0))),
                "available_packet_fraction": float(np.mean(packet_age <= ctrl["maximum_packet_age_s"])),
                "curvature_m_inv": curvature,
                "friction": mu,
                "minimum_friction_reserve_fraction": step_reserve,
            }
        )

    pair_errors = np.asarray(pair_errors)
    speed_errors = np.asarray(speed_errors)
    lateral_errors = np.asarray(lateral_errors)
    jerk_values = np.asarray(jerk_values)
    travelled_km = max(1e-9, float(np.sum(s[1:] - s_initial[1:])) / 1000.0)
    gap_rmse = float(np.sqrt(np.mean(pair_errors ** 2)))
    speed_rmse = float(np.sqrt(np.mean(speed_errors ** 2)))
    lateral_rmse = float(np.sqrt(np.mean(lateral_errors ** 2)))
    jerk_rms = float(np.sqrt(np.mean(jerk_values ** 2)))
    minimum_margin = float(min(r["minimum_safe_margin_m"] for r in records))
    thresholds = P["service_thresholds"]
    j_track = float(
        (gap_rmse / thresholds["gap_rmse_m"]) ** 2
        + (speed_rmse / thresholds["speed_rmse_mps"]) ** 2
        + (lateral_rmse / thresholds["lateral_rmse_m"]) ** 2
    )
    metrics = {
        "method": controller_name(method),
        "scenario": scenario,
        "seed": seed,
        "headway_s": headway,
        "budget_attempts_per_sample": budget,
        "strategy": strategy,
        "ablation": ablation or "none",
        "loss_probability": comm["loss_probability"],
        "burst_persistence": comm["burst_persistence"],
        "maximum_delay_steps": comm["maximum_delay_steps"],
        "gap_rmse_m": gap_rmse,
        "speed_rmse_mps": speed_rmse,
        "lateral_rmse_m": lateral_rmse,
        "jerk_rms_mps3": jerk_rms,
        "minimum_safe_margin_m": minimum_margin,
        "minimum_friction_reserve_fraction": float(np.min(friction_reserves)),
        "friction_reserve_p05_fraction": float(np.percentile(friction_reserves, 5)),
        "low_friction_reserve_time_fraction": float(np.mean(np.asarray(friction_reserves) <= 0.05)),
        "attempted_packets": attempted,
        "delivered_packets": delivered,
        "occupied_slots": occupied,
        "attempts_per_vehicle_km": attempted / travelled_km,
        "delivery_ratio": delivered / attempted if attempted else math.nan,
        "computation_p95_ms": float(np.percentile(computation_ms, 95)),
        "computation_max_ms": float(np.max(computation_ms)),
        "combined_tracking_score": j_track,
        "combined_tracking_score_definition": "(gap/tau_g)^2+(speed/tau_v)^2+(lateral/tau_y)^2",
        "collision": bool(min(r["minimum_gap_m"] for r in records) < 0),
    }
    return pd.DataFrame(records), metrics


def simulate_job(args):
    """Top-level worker wrapper so the Monte Carlo studies work on Windows."""
    method, scenario, comm, headway, budget, seed, strategy, ablation, gain_headway_override = args
    return simulate(method, scenario, comm, headway, budget, seed, strategy, ablation, gain_headway_override)[1]


def run_jobs(jobs):
    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(simulate_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


def service_success(metrics, scale=1.0):
    th = P["service_thresholds"]
    return bool(
        metrics["gap_rmse_m"] <= scale * th["gap_rmse_m"]
        and metrics["speed_rmse_mps"] <= scale * th["speed_rmse_mps"]
        and metrics["lateral_rmse_m"] <= scale * th["lateral_rmse_m"]
        and metrics["jerk_rms_mps3"] <= scale * th["jerk_rms_mps3"]
        and metrics["computation_p95_ms"] <= 1000.0 * th["computation_deadline_s"]
        and metrics["minimum_safe_margin_m"] >= th["minimum_safe_margin_m"]
    )


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def physical_boundary_experiments():
    kappas = np.linspace(0.0, 0.0125, 51)
    mus = [0.45, 0.60, 0.80, 1.00]
    rows = []
    for mi, mu in enumerate(mus):
        for ki, kappa in enumerate(kappas):
            result = robust_headway(mu, kappa, seed=17000 + 100 * mi + ki)
            rows.append({"friction": mu, "curvature_m_inv": kappa, **result})
    p1 = pd.DataFrame(rows)
    p1.to_csv(DATA / "S6_P1_physical_headway_boundary.csv", index=False)

    speeds = [20.0, 25.0, 30.0]
    speed_rows = []
    for vi, speed in enumerate(speeds):
        result = robust_headway(0.70, 0.004, speed=speed, seed=19000 + vi)
        speed_rows.append({"speed_mps": speed, "friction": 0.70, "curvature_m_inv": 0.004, **result})
    speed_df = pd.DataFrame(speed_rows)
    speed_df.to_csv(DATA / "S6_speed_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [2.25, 1.0]})
    ax = axes[0]
    colors = ["#9E2A2B", "#D17C0B", "#2A9D8F", "#0072B2"]
    max_grid = P["physical_boundary"]["headway_search_grid_s"][1]
    operational_h = P["physical_boundary"]["operational_headway_limit_s"]
    for color, mu in zip(colors, mus):
        q = p1[p1.friction == mu]
        y = q.h_min_s.where(q.status.isin(["PASS_OPERATIONAL", "PASS_EXTENDED"]))
        ax.plot(q.curvature_m_inv, y, color=color, label=fr"$\mu={mu:.2f}$")
        no_brake = q[q.status == "NO_POSITIVE_BRAKE_BOUND"]
        outside = q[q.status == "OUTSIDE_EXTENDED_RANGE"]
        if len(no_brake):
            ax.scatter(no_brake.curvature_m_inv, np.full(len(no_brake), max_grid), marker="x", s=14, color=color, alpha=0.8)
        if len(outside):
            ax.scatter(outside.curvature_m_inv, np.full(len(outside), max_grid - 0.08), marker="^", s=13, color=color, alpha=0.8)
    ax.axhline(operational_h, color="#555555", linestyle="--", linewidth=0.9)
    ax.text(kappas[-1], operational_h + 0.04, "operational reporting limit", ha="right", va="bottom", fontsize=6.8)
    ax.set(xlabel=r"Path curvature $|\kappa|$ (m$^{-1}$)", ylabel=r"Screened headway $\widehat h_{\min}^{\rm num}$ (s)", ylim=(0.35, max_grid + 0.10))
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(speed_df.speed_mps, speed_df.h_min_s, color="#0072B2", marker="o")
    for _, row in speed_df.iterrows():
        ax.annotate(f"{row.h_min_s:.2f}", (row.speed_mps, row.h_min_s), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set(xlabel="Reference speed (m/s)", ylabel=r"$\widehat h_{\min}^{\rm num}$ (s)", title=r"$\mu=0.70$, $|\kappa|=0.004$ m$^{-1}$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.38)
    save_figure(fig, "S6_P1_physical_headway_boundary")

    k2 = np.linspace(0.0, 0.014, 57)
    m2 = np.linspace(0.40, 1.00, 49)
    rows = []
    H = np.full((len(m2), len(k2)), np.nan)
    status_grid = np.zeros((len(m2), len(k2)), dtype=int)
    for a, mu in enumerate(m2):
        for b, kappa in enumerate(k2):
            result = robust_headway(mu, kappa, seed=26000 + a * 100 + b)
            rows.append({"friction": mu, "curvature_m_inv": kappa, **result})
            if result["status"] in ("PASS_OPERATIONAL", "PASS_EXTENDED"):
                H[a, b] = result["h_min_s"]
            elif result["status"] == "NO_POSITIVE_BRAKE_BOUND":
                status_grid[a, b] = 2
            else:
                status_grid[a, b] = 1
    pd.DataFrame(rows).to_csv(DATA / "S6_P2_curvature_friction_map.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad((0, 0, 0, 0))
    mesh = ax.pcolormesh(k2, m2, H, cmap=cmap, shading="auto", vmin=0.5, vmax=max_grid)
    overlay = np.zeros((len(m2), len(k2), 4))
    overlay[status_grid == 1] = mpl.colors.to_rgba("#BDBDBD")
    overlay[status_grid == 2] = mpl.colors.to_rgba("#D98C8C")
    ax.imshow(overlay, origin="lower", extent=[k2.min(), k2.max(), m2.min(), m2.max()], aspect="auto", interpolation="nearest")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"$\widehat h_{\min}^{\rm num}$ (s)")
    finite_for_contour = np.where(np.isfinite(H), H, max_grid + 0.2)
    ax.contour(k2, m2, finite_for_contour, levels=[operational_h], colors="white", linewidths=1.2)
    handles = [
        mpl.patches.Patch(color="#BDBDBD", label=r"$\widehat h_{\min}^{\rm num}>4.0$ s"),
        mpl.patches.Patch(color="#D98C8C", label="No positive brake authority"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=7)
    ax.set(xlabel=r"Path curvature $|\kappa|$ (m$^{-1}$)", ylabel=r"Friction coefficient $\mu$", title="Numerically screened physical operating boundary")
    save_figure(fig, "S6_P2_curvature_friction_headway_map")
    return p1, speed_df


def service_boundary_experiment():
    regimes = P["communication"]
    names = ["nominal", "moderate", "severe"]
    headways = np.round(np.arange(0.6, 2.61, 0.2), 2)
    budgets = [0, 1, 2, 3]
    seeds = P["monte_carlo"]["service_seeds"]
    physical = robust_headway(0.70, 0.004, seed=30001)["h_min_s"]
    jobs, descriptors = [], []
    for name in names:
        for budget in budgets:
            for h in headways:
                if h + 1e-9 < physical:
                    continue
                for seed in seeds:
                    jobs.append(("proposed", "service", regimes[name], h, budget, seed, "aware", None, None))
                    descriptors.append((name, budget, h, seed))
    metrics = run_jobs(jobs)
    run_rows = []
    for descriptor, m in zip(descriptors, metrics):
        name, budget, h, seed = descriptor
        run_rows.append({"regime": name, "budget": budget, "headway_s": h, "seed": seed, "service_success": service_success(m), **m})
    runs = pd.DataFrame(run_rows)
    runs.to_csv(DATA / "S6_P3_service_runs.csv", index=False)

    rows = []
    th = P["service_thresholds"]
    for name in names:
        for budget in budgets:
            for h in headways:
                phys_ok = h + 1e-9 >= physical
                if not phys_ok:
                    rows.append({"regime": name, "budget": budget, "headway_s": h, "physical_feasible": False, "successes": 0, "runs": 0, "service_success_fraction": math.nan, "wilson_low": math.nan, "wilson_high": math.nan})
                    continue
                md = runs[(runs.regime == name) & (runs.budget == budget) & (runs.headway_s == h)]
                successes = int(md.service_success.sum())
                low, high = wilson_interval(successes, len(md))
                rows.append({
                    "regime": name, "budget": budget, "headway_s": h, "physical_feasible": True,
                    "successes": successes, "runs": len(md), "service_success_fraction": successes / len(md),
                    "wilson_low": low, "wilson_high": high,
                    "mean_gap_rmse_m": md.gap_rmse_m.mean(), "mean_speed_rmse_mps": md.speed_rmse_mps.mean(),
                    "mean_lateral_rmse_m": md.lateral_rmse_m.mean(), "mean_jerk_rms_mps3": md.jerk_rms_mps3.mean(),
                    "mean_computation_p95_ms": md.computation_p95_ms.mean(), "mean_minimum_safe_margin_m": md.minimum_safe_margin_m.mean(),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "S6_P3_service_probability.csv", index=False)

    sensitivity_rows = []
    for scale in [0.9, 1.0, 1.1]:
        for keys, md in runs.groupby(["regime", "budget", "headway_s"]):
            count = sum(service_success(row, scale) for row in md.to_dict("records"))
            low, high = wilson_interval(count, len(md))
            sensitivity_rows.append({"threshold_scale": scale, "regime": keys[0], "budget": keys[1], "headway_s": keys[2], "successes": count, "runs": len(md), "service_success_fraction": count / len(md), "wilson_low": low, "wilson_high": high})
    pd.DataFrame(sensitivity_rows).to_csv(DATA / "S6_service_threshold_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True, sharey=True)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#D9D9D9")
    for ax, name in zip(axes, names):
        q = df[df.regime == name]
        Z = np.full((len(budgets), len(headways)), np.nan)
        for a, b in enumerate(budgets):
            for c, h in enumerate(headways):
                row = q[(q.budget == b) & (q.headway_s == h)].iloc[0]
                if row.physical_feasible:
                    Z[a, c] = row.service_success_fraction
        mesh = ax.pcolormesh(headways, budgets, Z, cmap=cmap, vmin=0, vmax=1, shading="nearest")
        finite = np.where(np.isfinite(Z), Z, -1.0)
        if np.nanmin(Z) <= th["required_success_fraction"] <= np.nanmax(Z):
            ax.contour(headways, budgets, finite, levels=[th["required_success_fraction"]], colors="white", linewidths=1.2)
        ax.axvline(physical, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Headway (s)")
    axes[0].set_ylabel("Attempt budget (sample$^{-1}$)")
    cb = fig.colorbar(mesh, ax=axes, pad=0.02, fraction=0.025)
    cb.set_label(r"Estimated service probability $\widehat p_{\rm svc}$")
    fig.text(0.5, 0.01, "Dashed: numerical physical boundary; white contour: service target 0.80; gray: physically infeasible", ha="center", fontsize=7)
    fig.subplots_adjust(bottom=0.19, wspace=0.12, right=0.88)
    save_figure(fig, "S6_P3_physical_service_probability")
    return df


def diagnose_headway_nonmonotonicity():
    seeds = P["monte_carlo"]["delay_seeds"]
    headways = np.round(np.arange(1.8, 2.61, 0.1), 2)
    jobs, descriptors = [], []
    for gain_mode, override in [("retuned", None), ("fixed_at_2.0_s", 2.0)]:
        for h in headways:
            for seed in seeds:
                jobs.append(("proposed", "service", P["communication"]["moderate"], h, 1, seed, "aware", None, override))
                descriptors.append((gain_mode, h, seed))
    rows = []
    for descriptor, m in zip(descriptors, run_jobs(jobs)):
        rows.append({"gain_mode": descriptor[0], "headway_s": descriptor[1], "seed": descriptor[2], "service_success": service_success(m), **m})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "S6_service_nonmonotonic_diagnosis.csv", index=False)
    return df


def emergency_experiment():
    methods = ["cacc", "dmpc", "rdmpc", "zheng2024", "proposed"]
    h0 = robust_headway(0.62, 0.0045, seed=40001)["h_min_s"]
    headway = min(2.5, math.ceil((h0 + 0.12) * 20) / 20)
    traces = []
    metrics = []
    for method in methods:
        trace, m = simulate(method, "emergency", P["communication"]["severe"], headway, 2, 2103)
        traces.append(trace)
        metrics.append(m)
        trace.to_csv(DATA / f"S6_P4_time_history_{method}.csv", index=False)
    md = pd.DataFrame(metrics)
    md.to_csv(DATA / "S6_P4_emergency_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for trace in traces:
        name = trace.method.iloc[0]
        color = COLORS[name]
        axes[0, 0].plot(trace.time_s, trace.gap_f1_m, label=name, color=color)
        axes[0, 1].plot(trace.time_s, trace.minimum_safe_margin_m, label=name, color=color)
        axes[1, 0].plot(trace.time_s, trace.f1_speed_mps, label=name, color=color)
    prop = traces[-1]
    axes[0, 0].plot(prop.time_s, prop.safe_gap_f1_m, color="black", linestyle="--", linewidth=1.2, label="Required gap")
    axes[1, 0].plot(prop.time_s, prop.leader_speed_mps, color="black", linestyle="--", linewidth=1.2, label="Leader")
    axes[1, 1].plot(prop.time_s, prop.mean_packet_age_s, color=COLORS["Proposed"], label="Mean packet age")
    axes[1, 1].plot(prop.time_s, prop.available_packet_fraction, color="#D17C0B", label="Available fraction")
    for ax in axes.flat:
        ax.axvspan(10, 18, color="#BDBDBD", alpha=0.25, linewidth=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("F1 bumper gap (m)")
    axes[0, 1].set_ylabel("Minimum safety margin (m)")
    axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 1].set_ylabel("Age (s) / available fraction")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    axes[1, 1].legend(frameon=False)
    axes[0, 1].text(0.50, 0.08, "shaded: complete V2V unavailability", transform=axes[0, 1].transAxes, ha="center", fontsize=7.2)
    fig.subplots_adjust(top=0.84, wspace=0.30, hspace=0.16)
    save_figure(fig, "S6_P4_emergency_braking_v2v_unavailability")
    return md, headway


def degradation_experiment(headway):
    methods = ["cacc", "dmpc", "rdmpc", "zheng2024", "proposed"]
    seeds = P["monte_carlo"]["delay_seeds"]
    jobs, descriptors = [], []
    for delay in [0, 1, 2, 3, 4, 5]:
        comm = {"loss_probability": 0.25, "burst_persistence": 0.55, "maximum_delay_steps": delay}
        for method in methods:
            for seed in seeds:
                jobs.append((method, "service", comm, headway, 2, seed, None, None, None))
                descriptors.append((delay, method, seed))
    rows = []
    for descriptor, m in zip(descriptors, run_jobs(jobs)):
        rows.append({"maximum_delay_steps_requested": descriptor[0], "maximum_delay_s": descriptor[0] * P["road_and_operation"]["simulation_step_s"], **m})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "S6_P5_delay_sensitivity_runs.csv", index=False)
    plot_degradation(df)
    return df


def plot_degradation(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    specs = [("gap_rmse_m", "Gap RMSE (m)"), ("speed_rmse_mps", "Speed RMSE (m/s)"), ("minimum_safe_margin_m", "Minimum safety margin (m)")]
    for ax, (metric, ylabel) in zip(axes, specs):
        for name in COLORS:
            z = df[df.method == name].groupby("maximum_delay_s")[metric].agg(["mean", "std", "count"]).reset_index()
            ci = 1.96 * z["std"] / np.sqrt(z["count"])
            ax.plot(z.maximum_delay_s, z["mean"], label=name, color=COLORS[name])
            ax.fill_between(z.maximum_delay_s, z["mean"] - ci, z["mean"] + ci, color=COLORS[name], alpha=0.10, linewidth=0)
        if metric == "minimum_safe_margin_m":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set(xlabel="Maximum delay (s)", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.8)
    save_figure(fig, "S6_P5_delay_sensitivity")


def pareto_experiment(headway):
    seeds = P["monte_carlo"]["resource_seeds"]
    configs = [
        ("Periodic", "proposed", "periodic", 5), ("Reduced rate", "proposed", "reduced", 5),
        ("Fixed event trigger", "cacc", "event", 5), ("Aware B=1", "proposed", "aware", 1),
        ("Aware B=2", "proposed", "aware", 2), ("Aware B=3", "proposed", "aware", 3),
    ]
    jobs, descriptors = [], []
    for label, method, strategy, budget in configs:
        for seed in seeds:
            jobs.append((method, "service", P["communication"]["moderate"], headway, budget, seed, strategy, None, None))
            descriptors.append((label, seed))
    rows = []
    for descriptor, m in zip(descriptors, run_jobs(jobs)):
        rows.append({"policy": descriptor[0], **m})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "S6_P6_resource_runs.csv", index=False)
    summary = df.groupby("policy").agg(packet_mean=("attempts_per_vehicle_km", "mean"), packet_sem=("attempts_per_vehicle_km", "sem"), score_mean=("combined_tracking_score", "mean"), score_sem=("combined_tracking_score", "sem")).reset_index()
    periodic = df[df.policy == "Periodic"].set_index("seed")
    rng = np.random.default_rng(72026)
    comparisons = []
    for label in [c[0] for c in configs[1:]]:
        q = df[df.policy == label].set_index("seed").loc[periodic.index]
        packet_reduction = 100.0 * (periodic.attempts_per_vehicle_km.values - q.attempts_per_vehicle_km.values) / periodic.attempts_per_vehicle_km.values
        score_change = q.combined_tracking_score.values - periodic.combined_tracking_score.values
        boot_idx = rng.integers(0, len(packet_reduction), size=(10000, len(packet_reduction)))
        pr_boot = packet_reduction[boot_idx].mean(axis=1)
        sc_boot = score_change[boot_idx].mean(axis=1)
        comparisons.append({"policy": label, "paired_packet_reduction_percent": packet_reduction.mean(), "packet_reduction_ci_low": np.quantile(pr_boot, 0.025), "packet_reduction_ci_high": np.quantile(pr_boot, 0.975), "paired_score_change": score_change.mean(), "score_change_ci_low": np.quantile(sc_boot, 0.025), "score_change_ci_high": np.quantile(sc_boot, 0.975)})
    summary = summary.merge(pd.DataFrame(comparisons), on="policy", how="left")
    summary.to_csv(DATA / "S6_P6_resource_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.45))
    order = [c[0] for c in configs]
    markers = ["s", "D", "^", "o", "o", "o"]
    colors = ["#777777", "#D17C0B", "#009E73", "#56B4E9", "#0072B2", "#003F5C"]
    for label, marker, color in zip(order, markers, colors):
        r = summary[summary.policy == label].iloc[0]
        ax.errorbar(r.packet_mean, r.score_mean, xerr=1.96 * r.packet_sem, yerr=1.96 * r.score_sem, fmt=marker, color=color, capsize=2.2, label=label)
    aware = summary[summary.policy.str.startswith("Aware")].sort_values("packet_mean")
    ax.plot(aware.packet_mean, aware.score_mean, color="#0072B2", alpha=0.55, linewidth=1.0)
    ax.set(xlabel="Attempted packets (vehicle-km$^{-1}$)", ylabel="Combined tracking score (lower is better)")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "S6_P6_communication_performance_pareto")
    return df


def ablation_experiment(headway):
    seeds = P["monte_carlo"]["ablation_seeds"]
    h0 = robust_headway(0.60, 0.006, seed=60001)["h_min_s"]
    headway = math.ceil((h0 + 0.075) * 20) / 20
    configs = [("Full method", None), ("No timestamp prediction", "no_timestamp"), ("No curve/friction coupling", "no_coupling"), ("No robust tightening", "no_tightening")]
    jobs, descriptors = [], []
    for label, ablation in configs:
        for seed in seeds:
            jobs.append(("proposed", "near_boundary", P["communication"]["severe"], headway, 2, seed, "aware", ablation, None))
            descriptors.append((label, seed))
    rows = []
    for descriptor, m in zip(descriptors, run_jobs(jobs)):
        rows.append({"configuration": descriptor[0], "numerical_boundary_s": h0, **m})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "S6_P7_near_boundary_ablation_runs.csv", index=False)
    summary = df.groupby("configuration").agg(gap=("gap_rmse_m", "mean"), speed=("speed_rmse_mps", "mean"), margin=("minimum_safe_margin_m", "mean"), friction_reserve_p05=("friction_reserve_p05_fraction", "mean"), low_reserve_fraction=("low_friction_reserve_time_fraction", "mean"), packets=("attempts_per_vehicle_km", "mean")).reindex([c[0] for c in configs])
    summary.to_csv(DATA / "S6_P7_near_boundary_ablation_summary.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.65))
    labels = ["Full", "No time", "No coupling", "No tightening"]
    x = np.arange(len(labels))
    for ax, metric, ylabel in zip(axes, ["gap", "speed", "margin", "low_reserve_fraction"], ["Gap RMSE (m)", "Speed RMSE (m/s)", "Minimum safety margin (m)", "Time with reserve $\leq 5\%$"]):
        vals = summary[metric].values
        ax.bar(x, vals, color=["#0072B2", "#9E77B5", "#2A9D8F", "#9E2A2B"], width=0.72)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "margin":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.suptitle(fr"Near-boundary stress: $h={headway:.2f}$ s; numerical boundary $={h0:.2f}$ s", fontsize=8.5)
    fig.subplots_adjust(bottom=0.34, wspace=0.56, top=0.84)
    save_figure(fig, "S6_P7_near_boundary_ablation")
    return df


def write_summary_table(emergency_metrics):
    cols = ["method", "minimum_safe_margin_m", "gap_rmse_m", "lateral_rmse_m", "attempts_per_vehicle_km", "computation_p95_ms"]
    table = emergency_metrics[cols].copy()
    table.to_csv(DATA / "S6_RESULT_TABLE.csv", index=False)
    pretty = table.rename(columns={"method": "Method", "minimum_safe_margin_m": "Minimum safety margin (m)", "gap_rmse_m": "Gap RMSE (m)", "lateral_rmse_m": "Lateral RMSE (m)", "attempts_per_vehicle_km": "Attempts/vehicle-km", "computation_p95_ms": "Computation P95 (ms)"})
    for c in pretty.columns[1:]:
        pretty[c] = pretty[c].map(lambda x: f"{x:.3f}")
    headers = list(pretty.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in pretty.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    (OUT / "S6_RESULT_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_record(start_time):
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "elapsed_seconds": time.perf_counter() - start_time,
        "parameter_file": str(PARAM_PATH.relative_to(ROOT)),
    }
    (OUT / "runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    start = time.perf_counter()
    setup_dirs()
    set_plot_style()
    print("P1/P2 physical boundary")
    physical_boundary_experiments()
    print("P3 service boundary")
    service_boundary_experiment()
    print("P3 diagnostic: controller-gain/headway interaction")
    diagnose_headway_nonmonotonicity()
    print("P4 emergency response")
    emergency, headway = emergency_experiment()
    print(f"Representative headway: {headway:.2f} s")
    print("P5 communication degradation")
    degradation_experiment(headway)
    print("P6 communication-performance Pareto")
    pareto_experiment(headway)
    print("P7 ablation")
    ablation_experiment(headway)
    write_summary_table(emergency)
    write_runtime_record(start)
    print(f"Completed in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
