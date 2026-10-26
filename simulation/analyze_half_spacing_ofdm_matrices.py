"""Conditioning and MMSE-mode diagnostics for the validated OFDM model.

For N=2K, a high-precision Lagrange/Vandermonde inverse resolves the smallest
singular value without a large arbitrary-precision eigendecomposition.
The scaled inverse is converted to float64 ONLY to measure its largest norm;
no tiny singular value is inferred from a rounded float64 matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import mpmath as mp
import numpy as np

from half_spacing_ofdm_sim import Config, TheoryLMMSE, Waveform, discrete_channel, write_csv
from simulation.validate_half_spacing_ofdm_model import theory_reference


def scientific_from_log10(value):
    exponent = math.floor(value)
    return f"{10 ** (value - exponent):.8g}e{exponent:+d}"


def scaled_vandermonde_inverse(dimension, dps):
    """Return U^-1/scale for U[n,q]=exp(j*pi*(n+.5)*q/N)/sqrt(N).

    Rows are reordered to the production [integer tones, half-shifted tones].
    Polynomial interpolation gives V^-1[j,n]=coefficient_n(L_j(z)).
    The scale is an explicit bound on the entries, so the complete inverse
    need not be retained as a large arbitrary-precision matrix.
    """
    if dimension < 2 or dimension % 2:
        raise ValueError("Require a positive even dimension >=2.")
    with mp.workdps(dps):
        n = dimension
        roots = [mp.exp(1j * mp.pi * j / n) for j in range(n)]
        polynomial = [mp.mpc(1)]
        for root in roots:
            next_coefficients = [-root * polynomial[0]]
            next_coefficients.extend(polynomial[i - 1] - root * polynomial[i] for i in range(1, len(polynomial)))
            next_coefficients.append(polynomial[-1])
            polynomial = next_coefficients
        products = [mp.mpf(1)]
        for distance in range(1, n):
            products.append(products[-1] * 2 * mp.sin(mp.pi * distance / (2 * n)))
        minimum_denominator = min(products[j] * products[n - 1 - j] for j in range(n))
        scale = mp.sqrt(n) * sum(abs(value) for value in polynomial) / minimum_denominator
        selected = sorted({0, 1, n // 2, n - 1})
        sampled_columns = {column: [] for column in selected}
        inverse = np.empty((n, n), complex)
        for j, root in enumerate(roots):
            # Product_{l!=j}(z_j-z_l): positive magnitudes and an explicit phase.
            # Python complex ** large_int may use a floating trig phase;
            # keep the exact fourth-root-of-unity factor even for large N.
            phase = (1, 1j, -1, -1j)[(n - 1) % 4] * (-1) ** (n - 1 - j) * mp.exp(
                1j * mp.pi * ((n - 2) * j + n * (n - 1) // 2) / (2 * n))
            denominator = phase * products[j] * products[n - 1 - j]
            quotient = [mp.mpc(0)] * n
            quotient[-1] = polynomial[-1]
            for degree in range(n - 2, -1, -1):
                quotient[degree] = polynomial[degree + 1] + root * quotient[degree + 1]
            row_factor = mp.sqrt(n) * mp.exp(-1j * mp.pi * j / (2 * n)) / denominator
            inverse[j] = [complex(row_factor * value / scale) for value in quotient]
            for column in selected:
                sampled_columns[column].append(row_factor * quotient[column])
        residual = mp.mpf(0)
        for row in selected:
            row_basis = [mp.exp(1j * mp.pi * (mp.mpf(row) + mp.mpf("0.5")) * j / n) / mp.sqrt(n) for j in range(n)]
            for column in selected:
                value = sum(row_basis[j] * sampled_columns[column][j] for j in range(n))
                residual = max(residual, abs(value - int(row == column)))
        order = np.r_[2 * np.arange(n // 2), 2 * np.arange(n // 2) + 1]
        return inverse[order], float(mp.log10(scale)), mp.nstr(residual, 12)


def matrix_parameters(name, matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest, smallest = float(singular[0]), float(singular[-1])
    tolerance = max(matrix.shape) * np.finfo(float).eps * largest
    condition = largest / smallest if smallest > 0 else None
    powers = singular ** 2
    probabilities = powers / powers.sum()
    nonzero = probabilities > 0
    row = {"matrix": name, "rows": matrix.shape[0], "columns": matrix.shape[1],
           "sigma_max_float64": largest, "sigma_min_float64": smallest,
           "condition_2_float64": condition,
           "condition_float64_reliable": bool(smallest > 100 * tolerance),
           "numerical_rank_float64": int(np.count_nonzero(singular > tolerance)),
           "rank_threshold": float(tolerance), "frobenius_norm": float(np.linalg.norm(matrix)),
           "stable_rank_energy": float(powers.sum() / powers[0]),
           "entropy_effective_rank_energy": float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))),
           "condition_2_high_precision": None, "condition_2_high_precision_log10": None,
           "sigma_min_high_precision": None, "sigma_min_high_precision_log10": None}
    return row, singular


def high_precision_extremes(config, h, matrices, spectra, dps=None):
    dimension = 2 * config.k
    if config.samples != dimension or np.any(abs(h) == 0):
        return {"computed": False, "reason": "Structured inverse requires N=2K and nonzero H."}
    dps = max(100, dimension + 80) if dps is None else dps
    print(f"High-precision structured inverse: {dimension} x {dimension}, {dps} digits", flush=True)
    inverse, log_scale, residual = scaled_vandermonde_inverse(dimension, dps)
    inverse_check, log_scale_check, residual_check = scaled_vandermonde_inverse(dimension, dps + 40)
    scaled_difference = float(np.linalg.norm(inverse - inverse_check) / np.linalg.norm(inverse_check))
    if max(float(residual), float(residual_check)) > 1e-30 or scaled_difference > 1e-12 or abs(log_scale - log_scale_check) > 1e-10:
        raise ArithmeticError(f"Structured inverse checks failed: residuals={residual}/{residual_check}, "
                              f"scaled difference={scaled_difference}, log-scale difference={abs(log_scale-log_scale_check)}.")
    norm_inverse_u = float(np.linalg.svd(inverse, compute_uv=False)[0])
    inverse_b = inverse / h[:, None]
    norm_inverse_b = float(np.linalg.svd(inverse_b, compute_uv=False)[0])
    inverse_c_scaled = inverse @ inverse.conj().T
    inverse_transfer_scaled = inverse_c_scaled / h[:, None]
    norm_inverse_transfer = float(np.linalg.svd(inverse_transfer_scaled, compute_uv=False)[0])
    log_inv_u = log_scale + math.log10(norm_inverse_u)
    log_inv_b = log_scale + math.log10(norm_inverse_b)
    log_inv_transfer = 2 * log_scale + math.log10(norm_inverse_transfer)
    log_condition_u = math.log10(spectra["U"][0]) + log_inv_u
    log_condition_b = math.log10(spectra["B"][0]) + log_inv_b
    values = {"U": (log_condition_u, -log_inv_u),
              "C_N": (2 * log_condition_u, -2 * log_inv_u),
              "B": (log_condition_b, -log_inv_b),
              "A": (2 * log_condition_b, -2 * log_inv_b),
              "C_N_D": (math.log10(spectra["C_N_D"][0]) + log_inv_transfer, -log_inv_transfer)}
    for row in matrices:
        if row["matrix"] in values:
            log_condition, log_min = values[row["matrix"]]
            row.update(condition_2_high_precision=scientific_from_log10(log_condition),
                       condition_2_high_precision_log10=log_condition,
                       sigma_min_high_precision=scientific_from_log10(log_min), sigma_min_high_precision_log10=log_min)
    return {"computed": True, "decimal_precision": dps, "confirmation_precision": dps + 40,
            "sampled_inverse_identity_max_residual": residual,
            "confirmation_identity_max_residual": residual_check,
            "scaled_inverse_relative_difference_between_precisions": scaled_difference,
            "method": "High-precision Lagrange/Vandermonde inverse; float64 largest norm of scaled inverse.",
            "full_high_precision_eigendecomposition": False,
            "normal_min_eigenvalue_log10": -2 * log_inv_b,
            "normal_max_eigenvalue": float(spectra["B"][0] ** 2),
            "weakest_mode_transition_db": -10 * (math.log10(config.bits_per_symbol) - 2 * log_inv_b),
            "C_gap_1_minus_sigma_max_P_log10": -2 * log_inv_u}


def analyze(config, n0=0.03, high_precision=True, dps=None):
    config.validate()
    wave, impulse = Waveform(config), discrete_channel(config)
    _, c, _, h = theory_reference(config, impulse)
    receiver = TheoryLMMSE(wave, h)
    u = wave.transmit(np.eye(2 * config.k)) / (config.mu * math.sqrt(config.samples))
    a = h.conj()[:, None] * c * h[None, :]
    source = {"P_N": c[:config.k, config.k:], "C_N": c, "D": np.diag(h),
              "C_N_D": c * h[None, :], "U": u, "B": receiver.b, "A": a,
              "R_n": n0 * config.symbol_duration * c}
    rows, spectra = [], {}
    for name, matrix in source.items():
        row, spectra[name] = matrix_parameters(name, matrix)
        rows.append(row)
    hp = high_precision_extremes(config, h, rows, spectra, dps) if high_precision else {"computed": False, "reason": "Disabled."}
    if hp["computed"]:
        c_row = next(row for row in rows if row["matrix"] == "C_N")
        noise_row = next(row for row in rows if row["matrix"] == "R_n")
        noise_row.update(condition_2_high_precision=c_row["condition_2_high_precision"],
                         condition_2_high_precision_log10=c_row["condition_2_high_precision_log10"],
                         sigma_min_high_precision=scientific_from_log10(c_row["sigma_min_high_precision_log10"] + math.log10(n0 * config.symbol_duration)),
                         sigma_min_high_precision_log10=c_row["sigma_min_high_precision_log10"] + math.log10(n0 * config.symbol_duration))
    rho = receiver.singular_values ** 2
    rho_max = float(rho[0])
    log_min_rho = hp["normal_min_eigenvalue_log10"] * math.log(10) if hp["computed"] else math.log(rho[-1]) if rho[-1] > 0 else -np.inf
    regularized = []
    for db in config.ebn0_db:
        nu, _, _ = config.noise_parameters(db)
        log_minimum = float(np.logaddexp(math.log(nu), log_min_rho))
        condition = math.exp(math.log(rho_max + nu) - log_minimum)
        shrink = rho / (rho + nu)
        weights = receiver.singular_values / (rho + nu)
        regularized.append({"ebn0_db": db, "lambda": nu, "max_eigenvalue": rho_max + nu,
                            "min_eigenvalue": math.exp(log_minimum), "condition_2": condition,
                            "modes_below_noise": int(np.count_nonzero(rho < nu)),
                            "effective_degrees_of_freedom": float(shrink.sum()),
                            "predicted_lmmse_mse": receiver.mse(nu),
                            "bias_mse": float(np.mean((1 - shrink) ** 2)),
                            "noise_mse": float(nu * np.mean(weights ** 2)),
                            "detector_operator_norm_2": float(weights.max()),
                            "explicit_normal_roundoff_budget": float(condition * (2 * config.k) * np.finfo(float).eps)})
    eigenvalues = np.linalg.eigvalsh(a)
    report = {"config": asdict(config), "noise_N0": n0, "matrices": rows, "regularized": regularized,
              "high_precision": hp, "diagnostics": {
                  "B_Gram_vs_analytic_A_relative_error": float(np.linalg.norm(receiver.b.conj().T @ receiver.b - a) / np.linalg.norm(a)),
                  "A_float64_min_signed_eigenvalue": float(eigenvalues[0]),
                  "A_float64_max_eigenvalue": float(eigenvalues[-1]),
                  "max_P_singular_value_float64": float(spectra["P_N"][0]),
                  "precision_warning": "Tiny float64 singular/eigenvalues and numerical rank are implementation diagnostics, not exact rank or condition numbers.",
                  "regularization_note": "Uncoded lambda=nu; same formula applies to coded cases using nu=1/(R*b*EbN0)."}}
    return report, spectra


def write_outputs(report, spectra, output, no_plots=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matrix_parameters.csv", report["matrices"])
    write_csv(output / "regularized_parameters.csv", report["regularized"])
    np.savez_compressed(output / "singular_values_float64.npz", **spectra)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 半子载波传输与检测矩阵参数", "",
             "采用验证过的有限采样理论 C_N。条件数均为 2-范数条件数。B=SD/sqrt(N)，A=DᴴC_ND=BᴴB，匹配输出传输矩阵为 C_ND。", "",
             "| 矩阵 | 最大奇异值（双精度） | 最小奇异值（双精度） | 双精度条件数 | 高精度结构估计条件数 | 双精度数值秩 | 双精度条件数可信 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["matrices"]:
        cond = f"{row['condition_2_float64']:.4e}" if row["condition_2_float64"] is not None else "未分辨"
        lines.append(f"| {row['matrix']} | {row['sigma_max_float64']:.5e} | {row['sigma_min_float64']:.5e} | {cond} | "
                     f"{row['condition_2_high_precision'] or '—'} | {row['numerical_rank_float64']} | {row['condition_float64_reliable']} |")
    lines += ["", "U=S/sqrt(N)，C_N=UᴴU；R_n=N0*T*C_N 与 C_N 条件数相同。"
              "P_N 本身的条件数不能代表完整 C_N 的病态程度；C_N 的特征值为 1±sigma_i(P_N)。"
              "sigma_max(P_N) 极接近 1 时，1-sigma_max(P_N) 形成极弱方向。", "",
              "未正则化矩阵的双精度最小奇异值可能完全由舍入决定。数值秩按 max(shape)*eps*sigma_max 计算，"
              "不表示理论秩下降。理论矩阵在当前非零信道下满列秩；浮点 A 的微小负特征值也不表示理论 A 非正定。", ""]
    hp = report["high_precision"]
    if hp["computed"]:
        lines += [f"高精度结构逆使用 {hp['decimal_precision']} 位，并用 {hp['confirmation_precision']} 位复算。"
                  f"两次归一化逆矩阵相对差异 {hp['scaled_inverse_relative_difference_between_precisions']:.3e}，"
                  f"采样逆关系残差 {hp['sampled_inverse_identity_max_residual']}。",
                  "极端奇异值由高精度逆矩阵的最大范数得到；归一化后只用双精度计算最大奇异值，"
                  "不对舍入后的原矩阵取最小奇异值。没有计算完整高精度特征谱。",
                  f"A 的最小特征值 log10(rho_min)={hp['normal_min_eigenvalue_log10']:.8f}；"
                  f"nu≈rho_min 的最弱模式过渡参考约 {hp['weakest_mode_transition_db']:.2f} dB，"
                  "该数值不是达到指定 BER 的门限。", ""]
    lines += ["| Eb/N0 (dB) | lambda | kappa(A+lambda I) | rho_i<nu 的模式数 | 有效自由度 | 理论 LMMSE MSE |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in report["regularized"]:
        lines.append(f"| {row['ebn0_db']:g} | {row['lambda']:.3e} | {row['condition_2']:.5e} | "
                     f"{row['modes_below_noise']} | {row['effective_degrees_of_freedom']:.3f} | {row['predicted_lmmse_mse']:.8g} |")
    lines += ["", "有效自由度为 sum[rho/(rho+nu)]，不是数学秩。MSE 是检测前的单位能量符号 MSE，不是 BER。"
              "正则化改善求解的数值条件，但不会让噪声以下的模式自动可恢复。",
              "正规矩阵直接形成的舍入预算 max(shape)*eps*kappa 单独记录；它是保守诊断，不是统计误差或精确相对误差保证。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for name, color, marker in (("U", "#0072B2", "o"), ("B", "#009E73", "^"), ("C_N_D", "#D55E00", "s")):
            values = spectra[name]
            axes[0].semilogy(np.arange(1, len(values) + 1), values, color=color, marker=marker,
                             markerfacecolor="white", markevery=32, label=name)
        axes[0].axhline(np.finfo(float).eps, color="grey", linestyle=":", label="float64 eps (scale reference)")
        axes[0].set(xlabel="Descending mode index", ylabel="Singular value (float64 diagnostic)")
        axes[0].legend()
        x = [row["ebn0_db"] for row in report["regularized"]]
        axes[1].semilogy(x, [row["condition_2"] for row in report["regularized"]], "o-", color="#332288")
        axes[1].set(xlabel="Eb/N0 (dB)", ylabel="Condition number of A + nu I")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.2)
        fig.savefig(output / "conditioning.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, default=Path("simulation_results/half_spacing_ofdm_model_validation/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/half_spacing_ofdm_matrix_analysis"))
    parser.add_argument("--no-high-precision", action="store_true")
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config_data = json.loads(args.validation_report.read_text(encoding="utf-8"))["config"]
    for name in ("ebn0_db", "tap_delays", "tap_gains", "codes"):
        if config_data.get(name) is not None:
            config_data[name] = tuple(config_data[name])
    report, spectra = analyze(Config(**config_data), high_precision=not args.no_high_precision, dps=args.digits)
    write_outputs(report, spectra, args.output_dir, args.no_plots)
    for row in report["matrices"]:
        print(f"{row['matrix']}: condition64={row['condition_2_float64']}, "
              f"high precision={row['condition_2_high_precision']}, numerical rank={row['numerical_rank_float64']}", flush=True)
    print(f"Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
