"""High-precision uncoded LMMSE reference and a full-size positive-rank bound.

High SNR uses an equivalent whitened eigenchannel, not a double-precision
time-domain waveform with an artificially increased signal/noise ratio.
No decoder implementation is included here.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import expit

from half_spacing_ofdm_sim import (
    Config, TheoryLMMSE, Waveform, complex_normal, discrete_channel,
    fir_frequency_response, qam_constellation, qam_indices, write_csv, write_summary,
)


def mp_channel(config):
    impulse = discrete_channel(config)
    taps = [(int(delay), mp.mpc(str(impulse[delay].real), str(impulse[delay].imag)))
            for delay in np.flatnonzero(impulse)]
    bins = [mp.mpf(k) for k in range(config.k)] + [mp.mpf(k) + mp.mpf("0.5") for k in range(config.k)]
    return [sum(value * mp.exp(-2j * mp.pi * frequency * delay / config.samples)
                for delay, value in taps) for frequency in bins]


def full_size_rank_bound(config):
    """Cauchy-Binet + a Vandermonde minor: lambda_min >= det_minor / trace**(d-1).

    The log of the bound is evaluated at high precision, without diagonalizing
    the full-size Gram matrix. This is a conservative mathematical bound, not
    an estimate of the actual smallest eigenvalue or practical required SNR.
    """
    with mp.workdps(80):
        h = mp_channel(config)
        powers = [abs(value) ** 2 for value in h]
        dimension, n = 2 * config.k, config.samples
        if any(value == 0 for value in powers):
            return {"subcarriers": dimension, "conditions_hold": False,
                    "reason": "At least one subcarrier channel is exactly zero; the no-floor result does not apply."}
        log_minor = -dimension * mp.log(n) + sum(mp.log(value) for value in powers)
        # Sorted half-spaced tones are q/2, q=0..d-1. The first d sample rows
        # form a Vandermonde matrix; midpoint row phases have unit magnitude.
        log_minor += sum(2 * (dimension - distance) * mp.log(2 * mp.sin(mp.pi * distance / (2 * n)))
                         for distance in range(1, dimension))
        log_lower = log_minor - (dimension - 1) * mp.log(sum(powers))
        minimum_distance_squared = mp.mpf(6) / (config.modulation - 1)
        ber_multiplier = 4 / minimum_distance_squared
        log10_lower = log_lower / mp.log(10)
        sufficient_db = 10 * (mp.log10(ber_multiplier / config.bits_per_symbol) - log10_lower + 6)
        return {"subcarriers": dimension, "samples": n, "conditions_hold": True,
                "minimum_channel_magnitude": float(min(abs(value) for value in h)),
                "lambda_min_lower_bound_log10": float(log10_lower),
                "lambda_min_lower_bound_log10_high_precision": mp.nstr(log10_lower, 40),
                "sufficient_db_for_ber_bound_1e_6": float(sufficient_db),
                "bound_is_conservative_not_actual_transition": True,
                "fixed_size_exact_arithmetic_asymptotic_ber": 0,
                "proof": "Distinct sample nodes and N>=2K imply full column rank. Nonzero D preserves rank. "
                         "MSE=mean(nu/(rho_i+nu))->0; BER<=SER<=4*MSE/dmin^2->0."}


def high_precision_spectrum(config):
    k, dimension = config.k, 2 * config.k
    dps = max(80, 4 * k + 50)
    with mp.workdps(dps):
        h = mp_channel(config)
        magnitudes = [abs(value) for value in h]
        if any(value == 0 for value in magnitudes):
            raise ValueError("High-precision no-floor reference requires nonzero subcarrier channels.")
        normal = mp.matrix(dimension)
        for i in range(dimension):
            normal[i, i] = magnitudes[i] ** 2
        for m in range(k):
            for q in range(k):
                delta = mp.mpf(q - m) + mp.mpf("0.5")
                value = magnitudes[m] * magnitudes[k + q] / (config.samples * mp.sin(mp.pi * delta / config.samples))
                normal[m, k + q] = normal[k + q, m] = value
        eigenvalues, real_vectors = mp.eigsy(normal)
        if eigenvalues[0] <= 0:
            raise ArithmeticError("Precision is insufficient to resolve the smallest positive eigenvalue.")
        # A=D^H C D; Q=diag(I,-jI) converts the imaginary cross block to real.
        diagonal = [mp.conj(h[i] / magnitudes[i]) * (1 if i < k else -1j) for i in range(dimension)]
        vectors = np.array([[complex(diagonal[i] * real_vectors[i, j]) for j in range(dimension)]
                            for i in range(dimension)])
        logs = np.array([float(mp.log(value)) for value in eigenvalues])
        residual = mp.norm(normal * real_vectors - real_vectors * mp.diag(eigenvalues)) / mp.norm(normal)
        # Solve an independently formed noiseless normal equation in arbitrary
        # precision, including channel phases and the Q transformation.
        symbols = [mp.mpc(1 if i % 2 else -1, 1 if i % 3 else -1) / mp.sqrt(2) for i in range(dimension)]
        transformed = mp.matrix([mp.conj(diagonal[i]) * symbols[i] for i in range(dimension)])
        recovered = mp.lu_solve(normal, normal * transformed)
        recovery_error = mp.norm(recovered - transformed) / mp.norm(transformed)
        transition = -10 * (mp.log10(eigenvalues[0]) + mp.log10(config.bits_per_symbol))
        metadata = {"k": k, "subcarriers": dimension, "decimal_precision": dps,
                    "lambda_min_log10": float(mp.log10(eigenvalues[0])),
                    "lambda_max_log10": float(mp.log10(eigenvalues[dimension - 1])),
                    "weakest_mode_transition_db": float(transition),
                    "eigendecomposition_relative_residual": mp.nstr(residual, 12),
                    "noiseless_recovery_relative_error": mp.nstr(recovery_error, 12),
                    "full_rank": True, "noiseless_recovery_passed": bool(recovery_error < mp.mpf("1e-30"))}
    receiver = TheoryLMMSE(Waveform(config), fir_frequency_response(config, discrete_channel(config)))
    discrepancies = []
    for db in (0, 10, 30, 60, 120):
        nu = 10 ** (-db / 10) / config.bits_per_symbol
        prediction = float(np.mean(expit(math.log(nu) - logs)))
        discrepancies.append(abs(receiver.mse(nu) - prediction) / prediction)
    metadata["waveform_SVD_MSE_overlap_relative_error_0_to_120_db"] = max(discrepancies)
    metadata["overlap_passed"] = max(discrepancies) < 1e-7
    return metadata, logs, vectors


def eigenchannel_lmmse(symbols, noise, log_eigenvalues, vectors, log_nu):
    bias = expit(log_nu - log_eigenvalues)
    signal = expit(log_eigenvalues - log_nu)
    noise_weight = np.sqrt(bias * signal)
    error = vectors @ (-bias[:, None] * (vectors.conj().T @ symbols) + noise_weight[:, None] * noise)
    # Computing the error directly avoids subtracting nearly equal quantities
    # or reconstructing a tiny noise term through an ill-conditioned waveform.
    return symbols + error, error, float(np.mean(bias))


def run_analysis(config, ks, output, no_plots=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    full_bound = full_size_rank_bound(config)
    rows, cases = [], []
    points, labels = qam_constellation(config.modulation)
    for k in sorted(set(ks)):
        reference = replace(config, k=k, codes=())
        metadata, logs, vectors = high_precision_spectrum(reference)
        cases.append(metadata)
        stop = max(140, 20 * math.ceil(metadata["weakest_mode_transition_db"] / 20) + 60)
        print(f"High precision: {2*k} tones, dps={metadata['decimal_precision']}, "
              f"log10(lambda_min)={metadata['lambda_min_log10']:.3f}, SNR up to {stop} dB", flush=True)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 41, k]))
        for db in range(0, stop + 1, 20):
            log_nu = -db * math.log(10) / 10 - math.log(config.bits_per_symbol)
            bit_errors, measured_mse = 0, []
            for start in range(0, config.frames, config.batch_size):
                size = min(config.batch_size, config.frames - start)
                indices = rng.integers(len(points), size=(2 * k, size))
                symbols = points[indices]
                estimated, error, predicted = eigenchannel_lmmse(symbols, complex_normal(rng, symbols.shape), logs, vectors, log_nu)
                decisions = qam_indices(estimated, config.modulation)
                bit_errors += int(np.count_nonzero(labels[decisions] != labels[indices]))
                measured_mse.extend(np.mean(abs(error) ** 2, axis=0))
            measured_mse = np.asarray(measured_mse)
            se = float(np.std(measured_mse, ddof=1) / math.sqrt(config.frames)) if config.frames > 1 else None
            observed = float(np.mean(measured_mse))
            passed = bool(abs(observed - predicted) <= max(6 * se, 1e-12 * predicted)) if se is not None else None
            rows.append({"k": k, "subcarriers": 2 * k, "ebn0_db": db,
                         "bit_count": 2 * k * config.frames * config.bits_per_symbol, "bit_errors": bit_errors,
                         "ber": bit_errors / (2 * k * config.frames * config.bits_per_symbol),
                         "mse": observed, "predicted_mse": predicted, "mse_standard_error": se,
                         "mse_theory_passed": passed,
                         "theoretical_ber_upper_bound": min(1.0, 4 * predicted / (6 / (config.modulation - 1))),
                         "model": "high-precision-spectrum equivalent eigenchannel; not float64 waveform"})
        final = rows[-1]
        print(f"  Final: BER={final['ber']:.6g}, MSE={final['mse']:.3e}, "
              f"BER upper bound={final['theoretical_ber_upper_bound']:.3e}", flush=True)
    report = {"config": asdict(config), "full_system_rank_bound": full_bound, "cases": cases, "results": rows,
              "scope": "Uncoded fixed-size exact-model LMMSE. High-SNR MC is an equivalent eigenchannel reference.",
              "full_size_spectrum_computed_at_high_precision": False,
              "passed": full_bound["conditions_hold"] and all(case["noiseless_recovery_passed"] and case["overlap_passed"] for case in cases)
                        and all(row["mse_theory_passed"] is not False for row in rows)}
    write_csv(output / "metrics.csv", rows)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# LMMSE 高信噪比与无误码平台验证", "",
             "条件：固定有限子载波数、N>=2K、充分 CP、准确模型、所有子载波信道非零、精确算术。",
             "采样基含满列秩 Vandermonde 子矩阵，故 A=DᴴC_ND 正定。",
             "MSE=mean[ν/(ρᵢ+ν)]→0；BER≤SER≤4*MSE/dmin²→0。这是无正渐近平台的数学证明，有限样本零误码本身不是证明。", "",
             f"当前完整系统：{2*config.k} 子载波，条件满足 {full_bound['conditions_hold']}。",
             "完整系统仅计算基于 Vandermonde 行列式和迹的保守正特征值下界，未用高精度求完整谱。", ""]
    if full_bound["conditions_hold"]:
        lines += [f"log10(λmin 下界)={full_bound['lambda_min_lower_bound_log10']:.6f}；"
                  f"使该保守 BER 上界≤1e-6 的充分 SNR 为 {full_bound['sufficient_db_for_ber_bound_1e_6']:.2f} dB。",
                  "这个极松的充分界不是实际所需 SNR，不是最弱模态的实测过渡点。", ""]
    lines += ["| 子载波数 | log10(λmin) | 最弱模态过渡参考 (dB) | 最后测试 SNR (dB) | 最后 BER | 最后理论 BER 上界 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for case in cases:
        final = [row for row in rows if row["k"] == case["k"]][-1]
        lines.append(f"| {case['subcarriers']} | {case['lambda_min_log10']:.6f} | {case['weakest_mode_transition_db']:.2f} | "
                     f"{final['ebn0_db']} | {final['ber']:.6g} | {final['theoretical_ber_upper_bound']:.3e} |")
    lines += ["", "参考使用高精度谱和稳定的等价白化特征信道，MSE 从误差表达式直接统计。"
              "它不是声称双精度时域实现能在数百 dB 下正确恢复极弱模态。"
              "小系统同时验证高精度无噪声恢复、谱分解残差、0–120 dB 的原 SVD MSE 一致性。",
              f"检查通过：{report['passed']}。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for case, color, marker in zip(cases, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), ("o", "s", "^", "D")):
            selected = [row for row in rows if row["k"] == case["k"]]
            x = [row["ebn0_db"] for row in selected]
            axes[0].semilogy(x, [row["ber"] if row["bit_errors"] else np.nan for row in selected],
                             color=color, marker=marker, markerfacecolor="white", label=f"{case['subcarriers']} tones")
            axes[0].semilogy(x, [row["theoretical_ber_upper_bound"] for row in selected], ":", color=color, alpha=0.6)
            zeros = [row for row in selected if not row["bit_errors"]]
            axes[0].scatter([row["ebn0_db"] for row in zeros], [1 / row["bit_count"] for row in zeros],
                            marker="v", facecolors="none", edgecolors=color, s=25)
            axes[1].semilogy(x, [row["predicted_mse"] for row in selected], color=color, label=f"{case['subcarriers']} tones")
            axes[1].scatter(x, [row["mse"] for row in selected], color=color, marker=marker, facecolors="none", s=18)
        for axis in axes:
            axis.set_xlabel("Eb/N0 (dB)")
            axis.grid(True, which="both", alpha=0.2)
            axis.legend()
        axes[0].set_ylabel("BER (dotted: theoretical upper bound)")
        axes[0].text(0.98, 0.02, "Open v: zero observed errors (at 1/Nbits)",
                     transform=axes[0].transAxes, ha="right", fontsize=8)
        axes[0].set_title("LMMSE high-precision eigenchannel reference")
        axes[1].set_ylabel("MSE (line: theory; symbols: Monte Carlo)")
        axes[1].set_title("No positive asymptotic floor in the exact model")
        fig.savefig(output / "lmmse_high_snr.png", dpi=180)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("simulation_results/half_spacing_ofdm_comparison/report.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()
    if any(k < 1 or k > 32 for k in args.k):
        parser.error("Reference K must be in [1,32].")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    data = report["config"]
    for name in ("ebn0_db", "tap_delays", "tap_gains", "codes"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    config = Config(**data)
    analysis = run_analysis(config, args.k, args.report.parent / "high_snr_analysis")
    report["high_snr_analysis"] = analysis
    report["simulation_passed"] = report.get("simulation_passed", True) and analysis["passed"]
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_summary(config, report, args.report.parent)
    if not analysis["passed"]:
        raise RuntimeError("High-SNR reference verification failed.")


if __name__ == "__main__":
    main()
