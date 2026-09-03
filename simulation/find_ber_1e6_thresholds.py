"""搜索四种 SISO AWGN 配置达到 BER <= 1e-6 的最小整数 SNR。

结果按配置增量写入 JSON；每个候选门限及其前一整数点均累计至少
ceil(-ln(0.05) / 1e-6) 个信息比特。零误码时，该样本量对应约 95%
单侧 BER 上限不超过 1e-6。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from channel.THzChannel import THzChannel
from params.PHYParams import PHYParams
from receiver.THzReceiver import THzReceiver
from transmitter.THzTransmitter import THzTransmitter


TARGET_BER = 1e-6
CONFIDENCE = 0.95
VERIFY_BITS = math.ceil(-math.log(1.0 - CONFIDENCE) / TARGET_BER)
QUICK_RUNS = 1
OUTPUT_PATH = Path("simulation_results/ber_1e6_awgn_thresholds.json")

CONFIGS = {
    "16qam_sc": {"label": "16QAM SC LDPC(1344,1440) AWGN", "mode": "sc-fde", "ncbps": 4, "start": 14, "stop": 23},
    "16qam_ofdm": {"label": "16QAM OFDM LDPC(1344,1440) AWGN", "mode": "ofdm", "ncbps": 4, "start": 14, "stop": 23},
    "64qam_sc": {"label": "64QAM SC LDPC(1344,1440) AWGN", "mode": "sc-fde", "ncbps": 6, "start": 20, "stop": 31},
    "64qam_ofdm": {"label": "64QAM OFDM LDPC(1344,1440) AWGN", "mode": "ofdm", "ncbps": 6, "start": 20, "stop": 31},
}


def _params(config: dict, snr_db: int, seed: int) -> PHYParams:
    params = PHYParams()
    params.update(
        data_source="PRBS",
        duration=1e-7,  # 小于一帧；组帧器仍生成一个完整有效帧
        sample_length=None,
        random_seed=seed,
        seed_strategy="固定种子",
        link_mode=config["mode"],
        NCBPS=config["ncbps"],
        code_type="LDPC",
        ldpc_matrix_type="ieee802153d_1440",
        ldpc_standard_rate="14/15",
        SNRdB=float(snr_db),
        enable_awgn=True,
        enable_multipath=False,
        enable_phase_noise=False,
        enable_cfo=False,
        enable_cfo_compensation=False,
        enable_iq_imbalance=False,
        enable_iq_compensation=False,
        enable_pa=False,
        enable_channel_equalization=True,
    )
    return params


def run_one(config: dict, snr_db: int, run_index: int) -> dict:
    seed = 20260902 + run_index
    params = _params(config, snr_db, seed)
    # AWGN 当前使用 numpy 全局随机源；显式设种子保证结果可复现。
    np.random.seed(91000000 + 1000 * snr_db + run_index)
    started = time.perf_counter()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        transmitter = THzTransmitter(params)
        transmitter.run()
        channel = THzChannel(params)
        received = channel.run(transmitter.tx_signal_dict)
        receiver = THzReceiver(params, transmitter)
        decoded = receiver.run(received)

    tx_bits = np.asarray(transmitter.data_bits_dict["signal_stream"]).ravel()
    rx_bits = np.asarray(decoded["signal_stream"]).ravel()
    compared_bits = min(len(tx_bits), len(rx_bits))
    errors = int(np.count_nonzero(tx_bits[:compared_bits] != rx_bits[:compared_bits]))
    return {
        "run_index": run_index,
        "seed": seed,
        "bits": compared_bits,
        "errors": errors,
        "ber": errors / compared_bits,
        "waveform_duration_seconds": float(transmitter.tx_signal_dict["duration_seconds"]),
        "elapsed_seconds": time.perf_counter() - started,
        "captured_log_tail": captured.getvalue().splitlines()[-4:],
    }


def accumulate(config: dict, snr_db: int, minimum_bits: int, minimum_runs: int = 1) -> dict:
    trials = []
    total_bits = total_errors = 0
    total_duration = 0.0
    run_index = 0
    while total_bits < minimum_bits or run_index < minimum_runs:
        trial = run_one(config, snr_db, run_index)
        trials.append(trial)
        total_bits += trial["bits"]
        total_errors += trial["errors"]
        total_duration += trial["waveform_duration_seconds"]
        run_index += 1
        print(
            f"  SNR={snr_db:2d} dB run={run_index:2d} "
            f"errors={total_errors}/{total_bits} BER={total_errors / total_bits:.3e}",
            flush=True,
        )

    bandwidth = 30e9
    raw_throughput = total_bits / total_duration
    nominal_se = raw_throughput / bandwidth
    ebn0_db = snr_db - 10.0 * math.log10(nominal_se)
    zero_error_upper_95 = (
        -math.log(1.0 - CONFIDENCE) / total_bits if total_errors == 0 else None
    )
    return {
        "SNRdB": snr_db,
        "EbN0_dB": ebn0_db,
        "BER": total_errors / total_bits,
        "total_bits": total_bits,
        "total_errors": total_errors,
        "independent_runs": run_index,
        "total_waveform_duration_seconds": total_duration,
        "raw_throughput_bps": raw_throughput,
        "nominal_spectral_efficiency_bps_per_hz": nominal_se,
        "zero_error_ber_upper_95": zero_error_upper_95,
        "elapsed_seconds": sum(t["elapsed_seconds"] for t in trials),
        "trials": trials,
    }


def save_document(document: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, OUTPUT_PATH)


def run_config(key: str, document: dict) -> None:
    config = CONFIGS[key]
    print(f"\n[{key}] {config['label']}", flush=True)
    quick = []
    candidate = None
    for snr_db in range(config["start"], config["stop"] + 1):
        point = accumulate(config, snr_db, minimum_bits=1, minimum_runs=QUICK_RUNS)
        quick.append(point)
        if point["BER"] <= TARGET_BER:
            candidate = snr_db
            break
    if candidate is None:
        raise RuntimeError(f"{key}: 扫描至 {config['stop']} dB 仍未找到候选门限")

    verified = []
    # 先验证候选点；若未达标则逐整数上移。最后补验门限前一点。
    threshold = candidate
    while True:
        point = accumulate(config, threshold, minimum_bits=VERIFY_BITS)
        verified.append(point)
        if point["BER"] <= TARGET_BER:
            break
        threshold += 1
        if threshold > config["stop"]:
            raise RuntimeError(f"{key}: 验证至 {config['stop']} dB 仍未达到目标 BER")

    previous = threshold - 1
    if not any(point["SNRdB"] == previous for point in verified):
        # 前一整数点只需证明明显高于目标。两帧中一旦出现多个误码，
        # 已足以排除 BER <= 1e-6，无需继续耗时累计到 300 万比特。
        verified.append(accumulate(config, previous, minimum_bits=1, minimum_runs=2))
    previous_point = next(point for point in verified if point["SNRdB"] == previous)
    threshold_point = next(point for point in verified if point["SNRdB"] == threshold)

    document["results"][key] = {
        "configuration": config,
        "threshold_SNRdB": threshold,
        "threshold_EbN0_dB": threshold_point["EbN0_dB"],
        "threshold_BER": threshold_point["BER"],
        "previous_integer_point_passes": previous_point["BER"] <= TARGET_BER,
        "quick_scan": quick,
        "verified_points": sorted(verified, key=lambda item: item["SNRdB"]),
    }
    save_document(document)
    print(
        f"[{key}] threshold: SNR={threshold} dB, "
        f"Eb/N0={threshold_point['EbN0_dB']:.6f} dB, "
        f"BER={threshold_point['BER']:.3e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    args = parser.parse_args()
    selected = list(CONFIGS) if args.config == "all" else [args.config]
    existing_results = {}
    if OUTPUT_PATH.exists():
        try:
            existing_results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")).get("results", {})
        except (OSError, json.JSONDecodeError):
            existing_results = {}
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_BER": TARGET_BER,
        "confidence": CONFIDENCE,
        "minimum_verification_bits": VERIFY_BITS,
        "decision": "最小整数 SNR 点的累计实测 BER <= 1e-6；并验证前一整数点",
        "EbN0_definition": "SNR - 10log10(raw_information_throughput / bandwidth)",
        "channel": "AWGN only",
        "receiver": {
            "cfo_compensation": False,
            "iq_compensation": False,
            "channel_estimation_and_equalization": True,
        },
        "results": existing_results,
    }
    for key in selected:
        run_config(key, document)
    print(f"\nSaved: {OUTPUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
