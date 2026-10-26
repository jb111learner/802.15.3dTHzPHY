"""Independent signal-only and noise-only tests of the half-spacing OFDM theory.

Run in communication:
    python -m simulation.validate_half_spacing_ofdm_model

Production Waveform/FIRLink implement the actual chain. The reference uses
explicit exponentials, direct tap sums and the analytical sampled P_N, without
calling production coupling_matrix, gram_matrix or fir_frequency_response.
No detector, coding, parameter fitting or signal-plus-noise mixture is used.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from half_spacing_ofdm_sim import Config, FIRLink, Waveform, complex_normal, discrete_channel, qam_constellation, write_csv


def relative_error(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def theory_reference(config, impulse):
    k, n = config.k, config.samples
    frequencies = np.r_[np.arange(k), np.arange(k) + 0.5]
    times = (np.arange(n) + 0.5) / n
    basis = np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
    delta = np.arange(k)[None, :] - np.arange(k)[:, None] + 0.5
    p_sampled = 1j / (n * np.sin(np.pi * delta / n))
    p_continuous = 1j / (np.pi * delta)
    sampled = np.block([[np.eye(k), p_sampled], [p_sampled.conj().T, np.eye(k)]])
    continuous = np.block([[np.eye(k), p_continuous], [p_continuous.conj().T, np.eye(k)]])
    # Independent direct summation, rather than the production H helper.
    response = sum(impulse[delay] * np.exp(-2j * np.pi * frequencies * delay / n)
                   for delay in np.flatnonzero(impulse))
    return basis, sampled, continuous, response


def check(name, value, limit):
    return {"name": name, "value": float(value), "limit": float(limit),
            "passed": bool(np.isfinite(value) and value <= limit)}


def receive_continuously(wave, impulse, symbols, batch_size):
    link = FIRLink(wave, impulse, wave.config.effective_cp_length)
    return np.hstack([link.receive(symbols[:, start:start + batch_size])
                      for start in range(0, symbols.shape[1], batch_size)])


def validate_signal(config, frames=64):
    """No noise: compare real continuous FIR samples and matched outputs with theory."""
    wave, impulse = Waveform(config), discrete_channel(config)
    basis, c, continuous, h = theory_reference(config, impulse)
    transfer = c * h[None, :]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 51]))
    points, _ = qam_constellation(config.modulation)
    payload = points[rng.integers(len(points), size=(2 * config.k, frames))]
    checks = []
    for group in ("both", "group0", "group1"):
        symbols = payload.copy()
        if group == "group0":
            symbols[config.k:] = 0
        if group == "group1":
            symbols[:config.k] = 0
        # Nonzero independent preceding blocks exercise state across boundaries.
        preceding = complex_normal(rng, (2 * config.k, 2))
        sequence = np.hstack((preceding, symbols))
        all_received = receive_continuously(wave, impulse, sequence, min(config.batch_size, 7))
        received = all_received[:, 2:]
        expected_time = config.mu * basis @ (h[:, None] * symbols)
        matched = wave.matched_filter(received)
        expected_matched = config.mu * config.symbol_duration * transfer @ symbols
        checks.append(check(group + "_time_domain_vs_explicit_theory", relative_error(received, expected_time), 3e-12))
        checks.append(check(group + "_raw_matched_vs_muT_CD", relative_error(matched, expected_matched), 3e-12))
        checks.append(check(group + "_normalized_matched_vs_CD", relative_error(
            matched / (config.mu * config.symbol_duration), transfer @ symbols), 3e-12))
        # Independent convolution implementation checks that the exercised link
        # really is the serialized FIR chain, not a frequency-domain shortcut.
        transmitted = wave.with_cp(sequence, config.effective_cp_length)
        serial = np.convolve(transmitted.ravel(order="F"), impulse)[:transmitted.size]
        convolution = serial.reshape(transmitted.shape, order="F")[config.effective_cp_length:, 2:]
        checks.append(check(group + "_FIRLink_vs_numpy_convolve", relative_error(received, convolution), 3e-13))
    # All columns: a random data check alone could conceal a swapped or missing block.
    identity = np.eye(2 * config.k, dtype=complex)
    operator_samples = receive_continuously(wave, impulse, identity, config.batch_size)
    measured_transfer = wave.matched_filter(operator_samples) / (config.mu * config.symbol_duration)
    checks.append(check("complete_transfer_matrix_CD", relative_error(measured_transfer, transfer), 3e-12))
    checks.append(check("sampled_basis_Gram", relative_error(basis.conj().T @ basis / config.samples, c), 3e-12))
    result = {"mode": "signal only; no noise", "frames_per_group_case": frames,
              "full_transfer_columns": 2 * config.k, "checks": checks,
              "passed": all(item["passed"] for item in checks),
              "sampled_vs_continuous_C_relative_difference": relative_error(c, continuous),
              "raw_model": "y=mu*T*C_N*D*a", "normalized_model": "z=C_N*D*a"}
    return result, {"signal_transfer_theory": transfer, "signal_transfer_measured": measured_transfer,
                    "sampled_Gram": c, "continuous_Gram": continuous}


def covariance_checks(empirical, theoretical, mean, pseudo, trials):
    """Complex Gaussian sample-covariance sampling scales; no fitted normalization."""
    diagonal = theoretical.diagonal().real
    rms = float(np.trace(theoretical).real / (math.sqrt(trials - 1) * np.linalg.norm(theoretical)))
    entry_se = np.sqrt(diagonal[:, None] * diagonal[None, :] / (trials - 1))
    pseudo_se = np.sqrt((diagonal[:, None] * diagonal[None, :] + abs(theoretical) ** 2) / (trials - 1))
    pseudo_rms = float(math.sqrt(np.trace(theoretical).real ** 2 + np.linalg.norm(theoretical) ** 2) /
                       (math.sqrt(trials - 1) * np.linalg.norm(theoretical)))
    checks = [check("covariance_relative_Frobenius", relative_error(empirical, theoretical), 3 * rms),
              check("covariance_max_entry_in_sampling_SE", np.max(abs(empirical - theoretical) / entry_se), 8),
              check("zero_mean_max_in_sampling_SE", np.max(abs(mean) / np.sqrt(diagonal / trials)), 8),
              check("proper_noise_pseudocovariance_relative_norm", np.linalg.norm(pseudo) / np.linalg.norm(theoretical), 3 * pseudo_rms),
              check("proper_noise_pseudocovariance_max_entry_in_sampling_SE", np.max(abs(pseudo) / pseudo_se), 8),
              check("empirical_covariance_Hermitian", relative_error(empirical, empirical.conj().T), 3e-14)]
    return checks, rms


def validate_noise(config, n0=0.03, trials=131072, batch_size=512):
    """Zero symbols through FIR; receiver AWGN only, followed by CP discard and MF.

    E[w w^H]=(N*N0/T) I at the FIR OUTPUT. Noise is not filtered by the
    propagation channel. Raw matched covariance is N0*T*C_N; normalized
    covariance is nu*C_N, nu=N0/(mu**2*T). Prefix noise is discarded.
    """
    if not np.isfinite(n0) or n0 <= 0 or trials < 1000 or batch_size < 1:
        raise ValueError("Require finite N0>0, trials>=1000 and positive batch_size.")
    wave, impulse = Waveform(config), discrete_channel(config)
    _, c, _, _ = theory_reference(config, impulse)
    dimension, n, t, cp = 2 * config.k, config.samples, config.symbol_duration, config.effective_cp_length
    theoretical = n0 * t * c
    time_variance = n * n0 / t
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 52]))
    link = FIRLink(wave, impulse, cp)
    total, second, pseudo_second = np.zeros(dimension, complex), np.zeros((dimension, dimension), complex), np.zeros((dimension, dimension), complex)
    input_energy, input_samples, zero_signal_max = 0.0, 0, 0.0
    for start in range(0, trials, batch_size):
        size = min(batch_size, trials - start)
        signal = link.receive(np.zeros((dimension, size), complex))
        zero_signal_max = max(zero_signal_max, float(np.max(abs(signal))))
        # Generate noise over the full received block including the prefix,
        # then retain exactly the same core used by the actual receiver.
        noise = math.sqrt(time_variance) * complex_normal(rng, (n + cp, size))
        input_energy += float(np.sum(abs(noise) ** 2))
        input_samples += noise.size
        matched_noise = wave.matched_filter(signal + noise[cp:])
        total += matched_noise.sum(axis=1)
        second += matched_noise @ matched_noise.conj().T
        pseudo_second += matched_noise @ matched_noise.T
    mean = total / trials
    empirical = (second - trials * np.outer(mean, mean.conj())) / (trials - 1)
    pseudo = (pseudo_second - trials * np.outer(mean, mean)) / (trials - 1)
    checks, rms = covariance_checks(empirical, theoretical, mean, pseudo, trials)
    checks.append(check("zero_transmitted_signal_through_FIR", zero_signal_max, 0))
    input_error = abs(input_energy / input_samples / time_variance - 1)
    checks.append(check("time_noise_variance_relative_error", input_error, 8 / math.sqrt(input_samples)))
    # Exact operator propagation complements the finite Monte Carlo estimate.
    matched_operator = wave.matched_filter(np.eye(n, dtype=complex))
    exact_covariance = time_variance * matched_operator @ matched_operator.conj().T
    checks.append(check("exact_actual_MF_operator_covariance_vs_N0T_C", relative_error(exact_covariance, theoretical), 3e-12))
    scale = (config.mu * t) ** 2
    nu = n0 / (config.mu ** 2 * t)
    normalized = empirical / scale
    checks.append(check("normalized_covariance_vs_nu_C", relative_error(normalized, nu * c), 3 * rms))
    k = config.k
    for name, block in (("group0", np.s_[:k, :k]), ("group1", np.s_[k:, k:]), ("cross01", np.s_[:k, k:]), ("cross10", np.s_[k:, :k])):
        block_se = np.sqrt(theoretical.diagonal().real[:, None] * theoretical.diagonal().real[None, :] / (trials - 1))[block]
        checks.append(check(name + "_covariance_max_entry_in_sampling_SE", np.max(abs(empirical[block] - theoretical[block]) / block_se), 8))
    result = {"mode": "noise only; transmitted symbols zero", "N0": n0, "trials": trials,
              "time_noise_variance": time_variance, "normalized_nu": nu,
              "raw_covariance_model": "N0*T*C_N", "normalized_covariance_model": "nu*C_N",
              "noise_injection_point": "receiver, after propagation FIR; prefix discarded",
              "centering": "sample mean removed, covariance divided by trials-1",
              "expected_relative_Frobenius_RMS_sampling_error": rms,
              "checks": checks, "passed": all(item["passed"] for item in checks),
              "wrong_white_model_relative_error": relative_error(empirical, n0 * t * np.eye(dimension)),
              "wrong_conjugated_model_relative_error": relative_error(empirical, theoretical.conj())}
    return result, {"noise_covariance_theory_raw": theoretical, "noise_covariance_empirical_raw": empirical,
                    "noise_covariance_exact_operator_raw": exact_covariance, "noise_mean_raw": mean,
                    "noise_pseudocovariance_raw": pseudo,
                    "noise_covariance_theory_normalized": nu * c, "noise_covariance_empirical_normalized": normalized}


def plot_comparison(theory, measured, output, name, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    limit = max(float(np.max(abs(theory.real))), float(np.max(abs(theory.imag))))
    difference = measured - theory
    error_limit = max(float(np.max(abs(difference.real))), float(np.max(abs(difference.imag))), 1e-16 * limit)
    for row, component in enumerate((np.real, np.imag)):
        for col, (matrix, heading) in enumerate(((theory, "Theory"), (measured, "Actual chain"), (difference, "Actual - theory"))):
            bound = error_limit if col == 2 else limit
            im = axes[row, col].imshow(component(matrix), origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
            axes[row, col].set(title=("Re: " if row == 0 else "Im: ") + heading, xlabel="Column", ylabel="Row")
            fig.colorbar(im, ax=axes[row, col], shrink=0.85)
    fig.suptitle(title)
    fig.savefig(Path(output) / name, dpi=160)
    plt.close(fig)


def run_validation(config, output, signal_frames=64, noise_trials=131072, n0=0.03, no_plots=False):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signal, signal_arrays = validate_signal(config, signal_frames)
    print(f"Signal only: {'PASS' if signal['passed'] else 'FAIL'}", flush=True)
    noise, noise_arrays = validate_noise(config, n0, noise_trials, config.batch_size)
    print(f"Noise only: {'PASS' if noise['passed'] else 'FAIL'}; covariance relative error="
          f"{noise['checks'][0]['value']:.6g}, expected sampling RMS="
          f"{noise['expected_relative_Frobenius_RMS_sampling_error']:.6g}", flush=True)
    report = {"config": asdict(config), "signal": signal, "noise": noise, "passed": signal["passed"] and noise["passed"],
              "reference_independence": "Explicit frequency exponentials, analytical P_N and direct FIR tap sum; no production theory helpers.",
              "continuous_theory_note": "Finite midpoint sampling uses P_N; continuous P is its N->infinity limit, not exactly equal at finite N."}
    write_csv(output / "signal_checks.csv", signal["checks"])
    write_csv(output / "noise_checks.csv", noise["checks"])
    np.savez_compressed(output / "matrices.npz", **signal_arrays, **noise_arrays)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 半子载波理论：信号与噪声独立链路验证", "",
             f"两组合计 {2*config.k} 子载波，N={config.samples}，T={config.symbol_duration:g}，"
             f"CP={config.effective_cp_length}，信道={config.channel}。", "",
             "## 无噪声信号", "",
             "独立两组 QAM 数据与单组激励通过实际 IFFT、组内 CP 后半频移、连续 FIR 和去 CP，再执行实际 FFT 匹配滤波。"
             "先行非零块和跨批次状态保留；理论侧独立计算指数基、抽头频响与解析耦合矩阵，不复用生产理论辅助函数。",
             "比较时域 r=mu*S*D*a、原始匹配输出 y=mu*T*C_N*D*a、归一化 z=C_N*D*a；"
             "逐列基激励还验证完整 CD 传输矩阵，独立 numpy.convolve 核对实际 FIR。", "",
             "| 检查 | 相对误差 | 限值 | 通过 |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {item['name']} | {item['value']:.3e} | {item['limit']:.3e} | {item['passed']} |" for item in signal["checks"])
    lines += ["", "## 纯噪声", "",
              f"发送全零，经实际 FIR 后在接收端加入复白噪声，方差 N*N0/T={noise['time_noise_variance']:.6g}，"
              f"N0={n0:g}；丢弃前缀噪声，匹配滤波后统计 {noise_trials} 个独立块。",
              "原始理论协方差 R_n=N0*T*C_N；归一化为 R_eta=nu*C_N，nu=N0/(mu²*T)。"
              "移除样本均值并使用 M-1 分母，分别检查两组内部协方差和两个跨组块，也检查均值与伪协方差。",
              f"实测相对 Frobenius 误差 {noise['checks'][0]['value']:.6g}，"
              f"理论抽样误差 RMS {noise['expected_relative_Frobenius_RMS_sampling_error']:.6g}。"
              "有限 Monte Carlo 协方差不应要求机器精度；额外用实际匹配滤波线性算子传播协方差作确定性检查。", "",
              "| 检查 | 数值 | 限值 | 通过 |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {item['name']} | {item['value']:.3e} | {item['limit']:.3e} | {item['passed']} |" for item in noise["checks"])
    lines += ["", "## 有限采样与连续公式", "",
              "原理论 P[m,k]=j/[pi*(k-m+1/2)]；中点采样的精确 P_N[m,k]=j/[N*sin(pi*(k-m+1/2)/N)]。"
              f"本次 C_N 与连续 C 的相对差异为 {signal['sampled_vs_continuous_C_relative_difference']:.6g}，"
              "它是采样积分差异，不是链路模型误差。未用经验归一化或矩阵拟合消除此差异。", "",
              f"总体通过：{report['passed']}。",
              "矩阵原始数据：matrices.npz；完整指标：report.json、signal_checks.csv、noise_checks.csv；"
              "图：signal_model_comparison.png、noise_covariance_comparison.png（未禁用绘图时）。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not no_plots:
        plot_comparison(signal_arrays["signal_transfer_theory"], signal_arrays["signal_transfer_measured"], output,
                        "signal_model_comparison.png", "Signal only: matched transfer C_N D")
        plot_comparison(noise_arrays["noise_covariance_theory_raw"], noise_arrays["noise_covariance_empirical_raw"], output,
                        "noise_covariance_comparison.png", f"Noise only: covariance N0 T C_N | {noise_trials} independent trials")
    if not report["passed"]:
        raise RuntimeError("Signal/noise theory validation failed; inspect report.json.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subcarriers", type=int, default=512)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--signal-frames", type=int, default=64)
    parser.add_argument("--noise-trials", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n0", type=float, default=0.03)
    parser.add_argument("--symbol-duration", type=float, default=1.0)
    parser.add_argument("--block-energy", type=float, default=1.0)
    parser.add_argument("--modulation", type=int, default=4)
    parser.add_argument("--channel", choices=("multipath", "awgn"), default="multipath")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/half_spacing_ofdm_model_validation"))
    args = parser.parse_args()
    if args.subcarriers < 2 or args.subcarriers % 2 or args.signal_frames < 1:
        parser.error("Require positive even subcarriers and positive signal-frames.")
    config = Config(k=args.subcarriers // 2, samples=args.samples, modulation=args.modulation,
                    frames=args.signal_frames, batch_size=args.batch_size, noise_trials=args.noise_trials,
                    symbol_duration=args.symbol_duration, block_energy=args.block_energy,
                    channel=args.channel, seed=args.seed, codes=())
    try:
        run_validation(config, args.output_dir, args.signal_frames, args.noise_trials, args.n0, args.no_plots)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
