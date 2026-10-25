"""独立的浮点/定点成对比较；它不是第三种运行模式。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from fixedpoint.config import FixedPointConfig
from fixedpoint.context import FixedPointContext


@dataclass
class ComparisonStimulus:
    seed: int
    channel: dict = field(default_factory=dict)


def _hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def nmse(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {"value": None, "db": None, "status": "shape_mismatch",
                "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    energy = float(np.sum(np.abs(reference) ** 2))
    error = float(np.sum(np.abs(candidate - reference) ** 2))
    if not math.isfinite(energy) or not math.isfinite(error):
        return {"value": None, "db": None, "status": "nonfinite"}
    value = error / energy if energy > 0 else None
    return {"value": value, "db": 10 * math.log10(value) if value and value > 0 else None,
            "error_energy": error, "reference_energy": energy,
            "status": "ok" if value is not None else "zero_reference"}


def evm(reference_symbols, recovered_symbols):
    reference, recovered = np.asarray(reference_symbols).reshape(-1), np.asarray(recovered_symbols).reshape(-1)
    result = nmse(reference, recovered)
    value = result.get("value")
    return {"rms": math.sqrt(value) if value is not None else None,
            "percent": 100 * math.sqrt(value) if value is not None else None,
            "status": result["status"], "symbols": len(reference)}


def binomial_interval(errors, total, confidence=0.95):
    if not total:
        return {"lower": None, "upper": None, "confidence": confidence}
    if not 0 < confidence < 1 or not 0 <= errors <= total:
        raise ValueError("invalid binomial counts/confidence")
    # 精确零错误单侧上界；有错误时使用 Wilson 双侧区间。
    if errors == 0:
        return {"lower": 0.0, "upper": -math.expm1(math.log(1 - confidence) / total),
                "confidence": confidence, "method": "zero_error_one_sided"}
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p, z2 = errors / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return {"lower": max(0., center - half), "upper": min(1., center + half),
            "confidence": confidence, "method": "wilson_two_sided"}


def bit_block_metrics(tx_bits, rx_bits, block_size, frame_size, valid_bits=None):
    tx, rx = np.asarray(tx_bits).reshape(-1), np.asarray(rx_bits).reshape(-1)
    if block_size <= 0 or frame_size <= 0:
        raise ValueError("block/frame size must be positive")
    if valid_bits is None:
        valid_bits = len(tx)
    if not 0 <= valid_bits <= len(tx):
        raise ValueError("invalid information-bit count")
    difference = np.ones(valid_bits, dtype=bool)
    available = min(valid_bits, len(rx))
    difference[:available] = tx[:available] != rx[:available]
    # 缺失比特保守计为错误，不能通过短长度截取隐去。
    bit_errors = int(np.count_nonzero(difference))
    block_errors = sum(bool(np.any(difference[start:start + block_size]))
                       for start in range(0, valid_bits, block_size))
    frame_errors = sum(bool(np.any(difference[start:start + frame_size]))
                       for start in range(0, valid_bits, frame_size))
    blocks, frames = math.ceil(valid_bits / block_size), math.ceil(valid_bits / frame_size)
    extra = max(0, len(rx) - len(tx))
    if extra:
        # 多余输出也是映射异常，不能称为无误块/帧。
        block_errors, frame_errors = blocks, frames
    return {"ber": bit_errors / valid_bits if valid_bits else None,
            "bit_errors": bit_errors, "bits": valid_bits,
            "bler": block_errors / blocks if blocks else None,
            "block_errors": block_errors, "blocks": blocks,
            "fer": frame_errors / frames if frames else None,
            "frame_errors": frame_errors, "frames": frames,
            "missing_bits": max(0, valid_bits - len(rx)), "extra_bits": extra,
            "length_mismatch": len(tx) != len(rx),
            "ber_interval": binomial_interval(bit_errors, valid_bits),
            "bler_interval": binomial_interval(block_errors, blocks),
            "missing_bit_policy": "count_as_error"}


def snr_loss(points, target=1e-3, metric="ber", confidence=.95):
    """经验交点用于诊断，计数置信区间形成推荐验收使用的交点范围。"""
    if not math.isfinite(target) or not 0 < target < 1 or metric not in {"ber", "bler"}:
        raise ValueError("target must be in (0, 1); metric is ber or bler")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    points = list(points)
    snrs = [float(point["snr_db"]) for point in points]
    if any(not math.isfinite(snr) for snr in snrs):
        raise ValueError("SNR points must be finite")
    if len(set(snrs)) != len(snrs):
        raise ValueError("duplicate SNR points must be merged before computing loss")
    def crossing(side):
        pairs = sorted((float(point["snr_db"]), point[side][metric]) for point in points)
        if any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for _, v in pairs):
            return None, "invalid_metric"
        values = [v for _, v in pairs]
        if any(values[i + 1] > values[i] for i in range(len(values) - 1)):
            return None, "nonmonotone_requires_more_samples"
        for x, value in pairs:
            if value == target:
                return {"lower": x, "upper": x, "estimate": x}, "sampled_target"
        for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
            if y0 > target > y1:
                estimate = None
                if y1 > 0:
                    estimate = x0 + (x1 - x0) * (math.log(target) - math.log(y0)) / (math.log(y1) - math.log(y0))
                return {"lower": x0, "upper": x1, "estimate": estimate}, "bracketed"
        return None, "target_not_bracketed"
    floating, fs = crossing("float")
    fixed, qs = crossing("fixed")
    result = {"metric": metric, "target": target, "float_crossing": floating,
              "fixed_crossing": fixed, "float_status": fs, "fixed_status": qs,
              "loss_db": None, "loss_interval_db": None}
    if floating and fixed:
        result["loss_interval_db"] = [fixed["lower"] - floating["upper"], fixed["upper"] - floating["lower"]]
        if floating["estimate"] is not None and fixed["estimate"] is not None:
            result["loss_db"] = fixed["estimate"] - floating["estimate"]
    # 两条曲线、所有采样点的二项计数区间使用 Bonferroni 分配失败概率。
    # 不依赖每个点预先报告的 Wilson 区间；这里使用精确 Clopper-Pearson。
    point_confidence = 1 - (1 - confidence) / (2 * len(points)) if points else None
    errors_key, total_key = {"ber": ("bit_errors", "bits"),
                             "bler": ("block_errors", "blocks")}[metric]
    def confidence_crossing(side, empirical_status):
        if not points:
            return None, "no_points", []
        if empirical_status in {"invalid_metric", "nonmonotone_requires_more_samples"}:
            return None, empirical_status, []
        from scipy.stats import beta
        intervals = []
        alpha = 1 - point_confidence
        for point in sorted(points, key=lambda item: item["snr_db"]):
            values = point[side]
            if errors_key not in values or total_key not in values:
                return None, "missing_counts", intervals
            errors, total = values[errors_key], values[total_key]
            if (not isinstance(errors, (int, np.integer)) or not isinstance(total, (int, np.integer))
                    or total <= 0 or not 0 <= errors <= total):
                return None, "invalid_counts", intervals
            if not math.isclose(values[metric], errors / total, rel_tol=1e-10, abs_tol=1e-15):
                return None, "metric_count_mismatch", intervals
            if errors == 0:
                lower, upper = 0., -math.expm1(math.log(alpha) / total)
                method = "zero_error_exact_one_sided"
            elif errors == total:
                lower, upper = math.exp(math.log(alpha) / total), 1.
                method = "all_error_exact_one_sided"
            else:
                lower = float(beta.ppf(alpha / 2, errors, total - errors + 1))
                upper = float(beta.ppf(1 - alpha / 2, errors + 1, total - errors))
                method = "clopper_pearson_two_sided"
            intervals.append({"snr_db": float(point["snr_db"]), "lower": lower, "upper": upper,
                              "errors": int(errors), "total": int(total), "method": method,
                              "confidence": point_confidence})
        above = [point["snr_db"] for point in intervals if point["lower"] > target]
        below = [point["snr_db"] for point in intervals if point["upper"] <= target]
        if not above:
            return None, "no_statistically_supported_above_target", intervals
        if not below:
            return None, "no_statistically_supported_below_target", intervals
        lower, upper = max(above), min(below)
        if lower >= upper:
            return None, "inconsistent_confidence_bands", intervals
        return {"lower": lower, "upper": upper}, "count_supported_bracket", intervals
    float_confidence, float_confidence_status, float_intervals = confidence_crossing("float", fs)
    fixed_confidence, fixed_confidence_status, fixed_intervals = confidence_crossing("fixed", qs)
    result.update({"confidence": confidence, "per_point_confidence": point_confidence,
                   "confidence_method": "Bonferroni exact binomial intervals; monotone curves assumed",
                   "confidence_float_crossing": float_confidence,
                   "confidence_fixed_crossing": fixed_confidence,
                   "confidence_float_status": float_confidence_status,
                   "confidence_fixed_status": fixed_confidence_status,
                   "confidence_point_intervals": {"float": float_intervals, "fixed": fixed_intervals},
                   "confidence_loss_interval_db": None})
    if float_confidence and fixed_confidence:
        result["confidence_loss_interval_db"] = [fixed_confidence["lower"] - float_confidence["upper"],
                                                  fixed_confidence["upper"] - float_confidence["lower"]]
    result["statistical_status"] = ("count_supported_loss_bracket" if result["confidence_loss_interval_db"] is not None
                                     else "insufficient_count_evidence; empirical_loss_is_diagnostic")
    return result


def compare_module(float_function, fixed_function, module_input):
    """隔离模块误差，函数必须具有相同输入/输出契约。"""
    state = np.random.get_state()
    try:
        floating = float_function(deepcopy(module_input))
        np.random.set_state(state)
        fixed = fixed_function(deepcopy(module_input))
    finally:
        np.random.set_state(state)
    reference = floating["signal_stream"] if isinstance(floating, dict) else floating
    candidate = fixed["signal_stream"] if isinstance(fixed, dict) else fixed
    return {"nmse": nmse(reference, candidate), "float_output": floating, "fixed_output": fixed}


def _run_link(params, stimulus, enable_fixed):
    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    from receiver.THzReceiver import THzReceiver
    config = deepcopy(params.get("fixed_point", {}))
    config["enable_fixed"] = bool(enable_fixed)
    params.update(fixed_point=config, random_seed=stimulus.seed, seed_strategy="固定种子")
    if "large_scale_seed" in params._params and params.get("large_scale_seed") is None:
        params.update(large_scale_seed=stimulus.seed)
    context = FixedPointContext.from_params(params)
    context.stimulus = stimulus.channel
    tx = THzTransmitter(params)
    signal = tx.run()
    from .fixedpoint_observations import observe_transmit_regions
    observe_transmit_regions(context, tx, signal)
    channel = THzChannel(params)
    received = channel.run(signal)
    rx = THzReceiver(params, tx)
    failure = None
    try:
        decoded = rx.run(received)
    except (ValueError, IndexError) as exc:
        # 确认属于接收失帧才计数；配置/内核错误必须直接抛出。
        text = str(exc)
        if not any(token in text for token in ("帧", "信号长度", "SYNC", "SFD", "同步", "采样率与时长", "length mismatch")):
            raise
        decoded = {"signal_stream": np.array([], dtype=np.uint8)}
        failure = {"type": type(exc).__name__, "message": text}
    info = np.asarray(tx.data_bits_dict["signal_stream"])
    padding = int(tx.data_bits_dict.get("padding_bit_num", 0))
    original_symbols = np.asarray(tx.modulated_data_dict["signal_stream"])
    stages = {"tx.waveform": signal["signal_stream"], "channel.output": received["signal_stream"]}
    for name, attribute in (("rx.matched_filter", "rx_matched"), ("rx.equalizer", "rx_equalized"),
                            ("rx.channel_estimation", "rx_estimated"), ("rx.llr", "llr_dict")):
        value = getattr(rx, attribute, None)
        if isinstance(value, dict) and "signal_stream" in value:
            stages[name] = value["signal_stream"]
        if isinstance(value, dict) and "channel_freq_response" in value:
            stages[name + ".H"] = np.asarray(value["channel_freq_response"])
    output = np.asarray(decoded["signal_stream"])
    coder = getattr(rx.decoder, "decoder", None)
    bits_per_code_symbol = int(getattr(coder, "c_exp", 1)) if params.get("code_type") == "RS" else 1
    block_size = int(getattr(coder, "k", rx.decoder.packet_size)) * bits_per_code_symbol
    coded_block_size = int(getattr(coder, "n", block_size)) * bits_per_code_symbol
    frame_size = int(tx.data_bits_dict["frame_bit_num"])
    metrics = bit_block_metrics(info, output, block_size, frame_size, len(info) - padding)
    length_events = getattr(context, "length_events", [])
    if length_events:
        # 原链路可能自动补齐，未知映射下采用整次试次失败的保守口径。
        metrics.update(ber=1.0, bit_errors=metrics["bits"], bler=1.0,
                       block_errors=metrics["blocks"], fer=1.0, frame_errors=metrics["frames"],
                       length_mismatch=True, mapping_failure_policy="fail_entire_trial")
        metrics["ber_interval"] = binomial_interval(metrics["bit_errors"], metrics["bits"])
        metrics["bler_interval"] = binomial_interval(metrics["block_errors"], metrics["blocks"])
    metrics["length_events"] = length_events
    recovered = stages.get("rx.equalizer", np.array([], dtype=complex))
    # 排除完全属于末尾填充的调制符号；混合码块仍保留。
    ncbps = int(params.get("NCBPS"))
    coded_valid = math.ceil((len(info) - padding) / block_size) * coded_block_size
    valid_symbols = min(len(original_symbols), math.ceil(coded_valid / ncbps))
    metrics["evm"] = evm(original_symbols[:valid_symbols], np.asarray(recovered).reshape(-1)[:valid_symbols])
    metrics["receiver_failure"] = failure
    return {"metrics": metrics, "stages": stages, "bits": info, "decoded": output,
            "symbols": original_symbols, "context": context, "report": context.report(),
            "generated_channel": getattr(channel, "chan_true", None)}


def run_float_fixed_comparison(phy_params, fixed_config=None, comparison_config=None):
    """配对测试同一 SNR；不改变调用者参数或全局 NumPy RNG 状态。"""
    from params.PHYParams import PHYParams
    comparison = dict(comparison_config or {})
    allowed = {"seed", "range_analysis", "output_dir", "include_arrays"}
    if set(comparison) - allowed:
        raise ValueError("unknown comparison fields: " + ", ".join(sorted(set(comparison) - allowed)))
    params = phy_params.clone() if isinstance(phy_params, PHYParams) else PHYParams()
    if not isinstance(phy_params, PHYParams):
        params.update(**dict(phy_params))
    cfg = FixedPointConfig.from_dict(fixed_config if fixed_config is not None else params.get("fixed_point", {})).to_dict()
    cfg["enable_fixed"] = True
    params.update(fixed_point=cfg)
    range_config = deepcopy(params.get("range_analysis", {}))
    range_config["enabled"] = bool(comparison.get("range_analysis", True))
    params.update(range_analysis=range_config)
    seed = int(comparison.get("seed", params.get("random_seed") if params.get("random_seed") is not None else 2026))
    stimulus = ComparisonStimulus(seed)
    random_state = np.random.get_state()
    try:
        np.random.seed(seed % 2**32)
        floating = _run_link(params.clone(), stimulus, False)
        np.random.seed(seed % 2**32)
        fixed = _run_link(params.clone(), stimulus, True)
    finally:
        np.random.set_state(random_state)
    if not np.array_equal(floating["bits"], fixed["bits"]):
        raise RuntimeError("float/fixed information bits differ")
    if params.get("enable_awgn") and "noise" not in stimulus.channel:
        raise RuntimeError("channel did not capture paired noise")
    channel_float, channel_fixed = floating["generated_channel"], fixed["generated_channel"]
    if channel_float is not None and channel_fixed is not None and not np.array_equal(channel_float, channel_fixed):
        raise RuntimeError("float/fixed generated environmental channel realizations differ")
    module_metrics = {name: nmse(reference, fixed["stages"][name])
                      for name, reference in floating["stages"].items() if name in fixed["stages"]}
    result = {"seed": seed, "snr_db": float(params.get("SNRdB")),
              "float": floating["metrics"], "fixed": fixed["metrics"],
              "nmse": module_metrics, "float_report": floating["report"], "fixed_report": fixed["report"],
              "fixed_config": cfg, "phy_params": deepcopy(params._params),
              "stimulus": {"information_bits_sha256": _hash(floating["bits"]),
                           "noise_sha256": _hash(stimulus.channel["noise"]) if "noise" in stimulus.channel else None,
                           "generated_channel_sha256": _hash(channel_float) if channel_float is not None else None,
                           "noise_power": stimulus.channel.get("noise_power"),
                           "noise_policy": "shared_absolute_variance_from_float_reference"}}
    if comparison.get("include_arrays", False):
        result["arrays"] = {"float": floating["stages"], "fixed": fixed["stages"]}
    if comparison.get("output_dir"):
        from .fixedpoint_report import save_comparison_report
        save_comparison_report(result, Path(comparison["output_dir"]))
    return result


def run_comparison_sweep(phy_params, snr_values, fixed_config=None, *, runs=1, seed=2026,
                         target=1e-3, target_metric="ber", output_dir=None):
    if runs < 1:
        raise ValueError("runs must be positive")
    points = []
    for snr in snr_values:
        params = phy_params.clone()
        params.update(SNRdB=float(snr))
        trials = [run_float_fixed_comparison(params, fixed_config, {"seed": seed + i, "range_analysis": False})
                  for i in range(runs)]
        point = {"snr_db": float(snr), "paired_runs": runs}
        for side in ("float", "fixed"):
            values = {field: sum(trial[side][field] for trial in trials)
                      for field in ("bit_errors", "bits", "block_errors", "blocks", "frame_errors", "frames", "missing_bits", "extra_bits")}
            for metric, numerator, denominator in (("ber", "bit_errors", "bits"), ("bler", "block_errors", "blocks"), ("fer", "frame_errors", "frames")):
                values[metric] = values[numerator] / values[denominator] if values[denominator] else None
                values[metric + "_interval"] = binomial_interval(values[numerator], values[denominator])
            point[side] = values
            evms = [trial[side]["evm"]["rms"] for trial in trials]
            values["max_evm_rms"] = max(evms) if all(v is not None for v in evms) else None
            values["max_overflow_rate"] = max((stats["overflow_rate"]
                for trial in trials for stats in trial[side + "_report"]["diagnostics"].values()), default=0.)
        if runs >= 2:
            differences = np.array([trial["fixed"]["ber"] - trial["float"]["ber"] for trial in trials])
            rng = np.random.default_rng(seed + 0x465850)
            means = np.mean(differences[rng.integers(0, runs, size=(2000, runs))], axis=1)
            point["paired_ber_delta"] = {"mean": float(np.mean(differences)),
                                         "bootstrap_95_percentile": np.percentile(means, [2.5, 97.5]).tolist(),
                                         "resampling_unit": "paired_run", "runs": runs}
        points.append(point)
    result = {"points": points, "snr_loss": snr_loss(points, target, target_metric),
              "paired_seeds": list(range(seed, seed + runs)), "stopping_rule": "same_fixed_run_count"}
    if output_dir:
        from .fixedpoint_report import save_sweep_report
        save_sweep_report(result, output_dir)
    return result


def evaluate_qformat_candidates(phy_params, candidates, snr_values, *, probe_runs=2,
                               confirmation_runs=4, seed=2026, target=1e-3,
                               max_snr_loss_db=.2, max_evm_rms=.03,
                               max_overflow_rate=1e-5, output_dir=None):
    """独立确认候选；没有满足条件的候选时不输出已验证推荐。"""
    snr_values = list(snr_values)
    if probe_runs < 1 or confirmation_runs < 1:
        raise ValueError("probe/confirmation runs must be positive")
    def audit(report):
        loss = report["snr_loss"]
        interval = loss.get("confidence_loss_interval_db")
        reasons = []
        if interval is None or interval[1] > max_snr_loss_db:
            reasons.append("confidence SNR loss not bounded within budget")
        for point in report["points"]:
            fixed = point["fixed"]
            if fixed["max_evm_rms"] is None or fixed["max_evm_rms"] > max_evm_rms:
                reasons.append("EVM exceeds budget or mapping failed")
            if fixed["max_overflow_rate"] > max_overflow_rate:
                reasons.append("overflow exceeds budget")
        # 达到目标必须有足够统计，而不能以零错误样本点代替确认。
        if not any(p["fixed"]["ber_interval"]["upper"] is not None
                   and p["fixed"]["ber_interval"]["upper"] <= target for p in report["points"]):
            reasons.append("no statistically supported fixed BER target point")
        return sorted(set(reasons))
    results = []
    for index, candidate in enumerate(candidates):
        cfg = FixedPointConfig.from_dict(candidate).to_dict()
        probe = run_comparison_sweep(phy_params, snr_values, cfg, runs=probe_runs, seed=seed, target=target)
        reasons = audit(probe)
        entry = {"index": index, "config": cfg, "probe": probe, "rejection_reasons": reasons, "validated": False}
        if not reasons:
            confirmation = run_comparison_sweep(phy_params, snr_values, cfg, runs=confirmation_runs,
                                                 seed=seed + 1_000_000, target=target)
            entry["confirmation"] = confirmation
            entry["rejection_reasons"] = audit(confirmation)
            entry["validated"] = not entry["rejection_reasons"]
        results.append(entry)
    accepted = [entry for entry in results if entry["validated"]]
    # 默认字长不是硬件面积模型；只在满足约束候选间提供可解释的排序。
    accepted.sort(key=lambda entry: entry["config"]["default_format"]["word_length"])
    result = {"candidates": results, "recommended": accepted[0]["config"] if accepted else None,
              "validated": bool(accepted), "constraints": {"target_ber": target,
              "max_snr_loss_db": max_snr_loss_db, "max_evm_rms": max_evm_rms,
              "max_overflow_rate": max_overflow_rate},
              "selection_rule": "smallest default word length among independently confirmed candidates"}
    if output_dir:
        from .fixedpoint_report import write_json
        write_json(Path(output_dir) / "recommended_qformats.json", result)
    return result
