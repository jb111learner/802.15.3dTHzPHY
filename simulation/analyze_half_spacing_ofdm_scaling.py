"""Carrier-count scan using a high-precision Toeplitz inverse.

After sorting tones and removing unit phases, C_N is real Toeplitz:
r_l = sin(pi*l/2)/(N*sin(pi*l/(2*N))). Even nonzero l give exact zeros.
Levinson recursion and the Gohberg--Semencul formula give its scaled inverse
in O(d^2) arbitrary-precision work, including rectangular sampling N >= d.
Only the largest norm of the scaled inverse is evaluated in float64.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import toeplitz

from half_spacing_ofdm_sim import Config, Waveform, discrete_channel, write_csv
from simulation.analyze_half_spacing_ofdm_matrices import scientific_from_log10


def scaled_toeplitz_inverse(dimension, samples, dps):
    """Return C_N^-1/scale in production group order, log10(scale), checks."""
    if dimension < 2 or dimension % 2 or samples < dimension:
        raise ValueError("Require even d >= 2 and N >= d.")
    with mp.workdps(dps):
        r = [mp.mpf(1)] + [mp.mpf(0) if lag % 2 == 0 else
             (-1) ** ((lag - 1) // 2) / (samples * mp.sin(mp.pi * lag / (2 * samples)))
             for lag in range(1, dimension)]
        coefficients, error = [mp.mpf(1)], mp.mpf(1)
        for degree in range(1, dimension):
            reflection = -(r[degree] + sum(coefficients[j] * r[degree-j]
                                           for j in range(1, degree))) / error
            coefficients = [mp.mpf(1)] + [coefficients[j] + reflection * coefficients[degree-j]
                                          for j in range(1, degree)] + [reflection]
            error *= 1 - reflection ** 2
            if error <= 0:
                raise ArithmeticError("Insufficient precision: nonpositive prediction variance.")
        # T*a = error*e_0 verifies the entire inverse first column in MP.
        residual = max(abs(sum(r[abs(i-j)] * coefficients[j] for j in range(dimension))
                           - (error if i == 0 else 0)) / error for i in range(dimension))
        if residual > mp.mpf("1e-30"):
            raise ArithmeticError(f"Insufficient precision: inverse first-column residual {mp.nstr(residual, 6)}.")
        coefficient_scale = max(abs(value) for value in coefficients)
        a = np.array([float(value / coefficient_scale) for value in coefficients])
        # T^-1 = (L(a)L(a)^T - L(0,a[n-1],...,a[1])L(...)^T)/error.
        lower = toeplitz(a, np.r_[a[0], np.zeros(dimension - 1)])
        reversed_tail = np.r_[0, a[:0:-1]]
        other = toeplitz(reversed_tail, np.zeros(dimension))
        inverse_real = lower @ lower.T - other @ other.T
        log_scale = float(2 * mp.log10(coefficient_scale) - mp.log10(error))
        phases = np.array([ (1, 1j, -1, -1j)[q % 4] for q in range(dimension)])
        inverse_sorted = phases.conj()[:, None] * inverse_real * phases[None, :]
        order = np.r_[2 * np.arange(dimension // 2), 2 * np.arange(dimension // 2) + 1]
        checks = {"decimal_precision": dps, "inverse_first_column_relative_residual": mp.nstr(residual, 12)}
        return inverse_sorted[np.ix_(order, order)], log_scale, checks


def scan_case(reference, dimension, samples, snrs=(30, 60, 120)):
    config = replace(reference, k=dimension // 2, samples=samples, codes=())
    # Keep the reference impulse's physical delays delay/N_ref*T unchanged.
    # Do not round these delays onto each new sampling grid.
    impulse = discrete_channel(reference)
    delays = np.flatnonzero(impulse)
    bins = np.r_[np.arange(config.k), np.arange(config.k) + .5]
    h = np.exp(-2j * np.pi * bins[:, None] * delays[None, :] / reference.samples) @ impulse[delays]
    if np.any(abs(h) == 0):
        raise ValueError("An exactly zero channel response invalidates the full-rank scan.")
    wave = Waveform(config)
    u = wave.transmit(np.eye(dimension)) / (config.mu * math.sqrt(samples))
    c = u.conj().T @ u
    b = u * h[None, :]
    transfer = c * h[None, :]
    dps = max(100, dimension + 100)
    inverse, log_scale, checks = scaled_toeplitz_inverse(dimension, samples, dps)
    confirmation, check_scale, check = scaled_toeplitz_inverse(dimension, samples, dps + 40)
    difference = float(np.linalg.norm(inverse - confirmation) / np.linalg.norm(confirmation))
    if difference > 1e-12 or abs(log_scale - check_scale) > 1e-10:
        raise ArithmeticError("Precision confirmation failed.")
    checks.update(confirmation_precision=dps + 40, scaled_inverse_relative_difference=difference,
                  confirmation_residual=check["inverse_first_column_relative_residual"])
    c_max = float(np.linalg.eigvalsh(c)[-1])
    b_singular = np.linalg.svd(b, compute_uv=False)
    a_max = float(b_singular[0] ** 2)
    inv_c_norm = float(np.linalg.eigvalsh(inverse)[-1])
    inv_a = inverse / h[:, None] / h.conj()[None, :]
    inv_a_norm = float(np.linalg.eigvalsh(inv_a)[-1])
    inv_transfer_norm = float(np.linalg.svd(inverse / h[:, None], compute_uv=False)[0])
    c_min_log = -log_scale - math.log10(inv_c_norm)
    a_min_log = -log_scale - math.log10(inv_a_norm)
    c_cond_log = math.log10(c_max) - c_min_log
    a_cond_log = math.log10(a_max) - a_min_log
    transfer_cond_log = math.log10(float(np.linalg.svd(transfer, compute_uv=False)[0])) + log_scale + math.log10(inv_transfer_norm)
    row = {"total_subcarriers": dimension, "per_group_subcarriers": dimension // 2,
           "samples": samples, "sampling_policy": "fixed_N" if samples == reference.samples else "N_equals_d",
           "C_condition_log10": c_cond_log, "U_condition_log10": .5 * c_cond_log,
           "transfer_condition_log10": transfer_cond_log, "B_condition_log10": .5 * a_cond_log,
           "A_condition_log10": a_cond_log,
           "transfer_condition": scientific_from_log10(transfer_cond_log),
           "A_condition": scientific_from_log10(a_cond_log),
           "C_min_eigenvalue_log10": c_min_log, "A_min_eigenvalue_log10": a_min_log,
           "channel_condition": float(max(abs(h)) / min(abs(h))),
           "channel_only_normal_condition": float((max(abs(h)) / min(abs(h))) ** 2),
           "weakest_mode_transition_db": -10 * (math.log10(config.bits_per_symbol) + a_min_log),
           "float64_transfer_condition_diagnostic": float(np.linalg.cond(transfer)),
           "float64_B_numerical_rank": int(np.count_nonzero(b_singular > dimension * np.finfo(float).eps * b_singular[0]))}
    regularized = []
    for db in snrs:
        nu = 10 ** (-db / 10) / config.bits_per_symbol
        log_min_regularized = float(np.logaddexp(math.log(nu), a_min_log * math.log(10)))
        shrink = b_singular ** 2 / (b_singular ** 2 + nu)
        regularized.append({"total_subcarriers": dimension, "samples": samples,
                            "ebn0_db": db, "condition_2": math.exp(math.log(a_max + nu) - log_min_regularized),
                            "predicted_lmmse_mse": float(np.mean(nu / (b_singular ** 2 + nu))),
                            "effective_degrees_of_freedom": float(shrink.sum())})
    return row, regularized, checks


def run_scan(reference, dimensions, output, no_plots=False):
    reference.validate()
    dimensions = sorted(set(dimensions))
    if not dimensions:
        raise ValueError("At least one carrier count is required.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, regularized, checks = [], [], []
    for dimension in dimensions:
        if dimension > reference.samples:
            raise ValueError("The fixed-N scan requires d <= reference N.")
        for policy, samples in (("fixed_N", reference.samples), ("N_equals_d", dimension)):
            print(f"{policy}: d={dimension}, N={samples}", flush=True)
            # Reuse the identical endpoint instead of recomputing it.
            if policy == "N_equals_d" and dimension == reference.samples:
                row, reg, check = dict(rows[-1]), [dict(item) for item in regularized[-3:]], dict(checks[-1])
            else:
                row, reg, check = scan_case(reference, dimension, samples)
            row["sampling_policy"] = policy
            for item in reg:
                item["sampling_policy"] = policy
            check.update(total_subcarriers=dimension, samples=samples, sampling_policy=policy)
            rows.append(row)
            regularized.extend(reg)
            checks.append(check)
            write_csv(output / "carrier_count_parameters.csv", rows)
            write_csv(output / "regularized_parameters.csv", regularized)
    fits = []
    for policy in ("fixed_N", "N_equals_d"):
        selected = [row for row in rows if row["sampling_policy"] == policy and row["total_subcarriers"] >= 64]
        if len(selected) >= 3:
            x = np.array([row["total_subcarriers"] for row in selected])
            y = np.array([row["transfer_condition_log10"] for row in selected])
            slope, intercept = np.polyfit(x, y, 1)
            fits.append({"sampling_policy": policy, "minimum_carriers": int(x.min()), "maximum_carriers": int(x.max()),
                         "log10_condition_slope_per_carrier": float(slope), "log10_condition_intercept": float(intercept),
                         "R_squared": float(1 - np.sum((y - slope*x - intercept)**2) / np.sum((y-y.mean())**2)),
                         "empirical_fit_only": True})
    report = {"reference_config": asdict(reference), "scan": rows, "regularized": regularized, "checks": checks, "empirical_fits": fits,
              "physical_channel": "Reference delays/N_ref*T and gains fixed; T and half-spacing fixed, occupied bandwidth grows with d.",
              "method": "MP Levinson + Gohberg-Semencul scaled inverse, precision confirmation; float64 largest norm only."}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tap_delays = np.flatnonzero(discrete_channel(reference)).tolist()
    lines = ["# 半子载波系统的条件数随载波数变化", "",
             "总载波数 d=2K。固定 T、半子载波间隔及物理多径信道；增加 d 会扩大占用带宽。",
             f"信道沿用参考 N={reference.samples} 时的物理时延（采样延迟 {tap_delays} 除以 {reference.samples} 再乘 T）与归一化增益，改变 N 时不重新取整时延。",
             f"对比固定 N={reference.samples} 与 N=d 两种采样策略。后者为矩阵层面的同一物理频率响应分析，不是重新量化 FIR 时延后的链路仿真。", "",
             "C_N=UᴴU，B=UD，A=BᴴB；匹配滤波输出传输矩阵为 C_ND。使用高精度 Toeplitz 逆，均经增加 40 位精度复算及完整首列逆关系残差检查。", "",
             "| 总载波 d | N | 策略 | κ(C_ND) | κ(A) | log10 κ(B) | 最弱模式参考 Eb/N0 (dB) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['total_subcarriers']} | {row['samples']} | {row['sampling_policy']} | {row['transfer_condition']} | {row['A_condition']} | {row['B_condition_log10']:.4f} | {row['weakest_mode_transition_db']:.2f} |")
    lines += ["", "当 d 增大时，P 的最大奇异值越来越接近 1，C_N 的最小特征值 1−σ_max(P) 越来越小。",
              "κ(A)=κ(B)²；普通正交 OFDM 的对应矩阵为 D 和 DᴴD，仅受信道起伏影响。",
              "最弱模式参考定义为 ν=λ_min(A)，不是指定 BER 的门限，也不是实用工作信噪比要求。",
              "正则化后的条件数 (λ_max(A)+ν)/(λ_min(A)+ν) 在 λ_min≪ν 时主要由最大特征值和噪声决定；求解稳定不等于弱模式可恢复。",
              "双精度直接条件数仅作为诊断保存在 CSV，不作为真实增长趋势。MSE 为符号检测前理论 MSE，不是 BER。"]
    for fit in fits:
        lines += ["", f"{fit['sampling_policy']}，d={fit['minimum_carriers']}–{fit['maximum_carriers']} 的经验拟合："
                  f"log10 κ(C_ND) ≈ {fit['log10_condition_slope_per_carrier']:.6f} d "
                  f"{fit['log10_condition_intercept']:+.6f}，R²={fit['R_squared']:.8f}。"
                  "这仅描述本次扫描范围，不是渐近定理；固定 N 的扫描必须满足 d≤N。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not no_plots:
        plot_scan(rows, regularized, output)
    return report


def plot_scan(rows, regularized, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for policy, style, marker in (("fixed_N", "-", "o"), ("N_equals_d", "--", "s")):
        selected = [row for row in rows if row["sampling_policy"] == policy]
        x = [row["total_subcarriers"] for row in selected]
        policy_label = f"N={max(row['samples'] for row in rows)}" if policy == "fixed_N" else "N=d"
        for key, color, label in (("transfer_condition_log10", "#D55E00", "C_N D"), ("B_condition_log10", "#0072B2", "B")):
            axes[0].plot(x, [row[key] for row in selected], color=color, linestyle=style, marker=marker,
                         markerfacecolor="white", label=f"{label}, {policy_label}")
        for db, color in ((30, "#009E73"), (60, "#CC79A7"), (120, "#332288")):
            reg = [row for row in regularized if row["sampling_policy"] == policy and row["ebn0_db"] == db]
            axes[1].plot(x, [math.log10(row["condition_2"]) for row in reg], color=color, linestyle=style,
                         marker=marker, markerfacecolor="white", label=f"{db} dB, {policy_label}")
        reg = [row for row in regularized if row["sampling_policy"] == policy and row["ebn0_db"] == 30]
        axes[2].plot(x, [row["predicted_lmmse_mse"] for row in reg], color="#D55E00", linestyle=style,
                     marker=marker, markerfacecolor="white", label=f"LMMSE, 30 dB, {policy_label}")
    axes[0].set(ylabel="log10 condition number", title="Unregularized matrices")
    axes[1].set(ylabel="log10 condition number", title="A + nu I")
    axes[2].set(ylabel="Predicted symbol MSE", title="Weak modes remain below noise")
    for axis in axes:
        axis.set_xlabel("Total subcarriers d = 2K")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.savefig(output / "carrier_count_trend.png", dpi=180)
    fig.savefig(output / "carrier_count_trend.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-report", type=Path, default=Path("simulation_results/half_spacing_ofdm_model_validation/report.json"))
    parser.add_argument("--carriers", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--output", type=Path, default=Path("simulation_results/half_spacing_ofdm_carrier_scaling"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.config_report.read_text(encoding="utf-8"))["config"]
    reference = Config(**payload)
    run_scan(reference, sorted(set(args.carriers)), args.output, args.no_plots)


if __name__ == "__main__":
    main()
