"""MIMO-OFDM BER、吞吐率、NMSE 与条件数 SNR 扫描工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from params.PHYParams import PHYParams
from simulation.SimulationManager import SimulationManager


def build_mimo_params(snr_db, detector="mmse", csi_mode="estimated", **overrides):
    params = PHYParams()
    values = dict(
        enable_mimo=True,
        link_mode="ofdm",
        num_tx=2,
        num_rx=2,
        num_spatial_streams=2,
        mimo_detector=detector,
        mimo_csi_mode=csi_mode,
        mimo_channel_model="iid_rayleigh",
        enable_multipath=True,
        mimo_num_taps=4,
        enable_awgn=True,
        SNRdB=float(snr_db),
    )
    values.update(overrides)
    params.update(**values)
    return params


def run_sweep(snr_values, detector="mmse", csi_mode="estimated", **overrides):
    results = []
    for snr_db in snr_values:
        params = build_mimo_params(snr_db, detector, csi_mode, **overrides)
        manager = SimulationManager(base_params=params, save_plots=False)
        run = manager.run_once()
        results.append({"snr_db": float(snr_db), **run["metrics"]})
    return results


def plot_sweep(results, output_dir="simulation/simulation_results", prefix="mimo"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snr = np.array([item["snr_db"] for item in results])
    plots = (
        ("ber", "BER", True),
        ("effective_throughput_bps", "有效吞吐率 (bit/s)", False),
        ("mimo_channel_nmse", "信道估计 NMSE", True),
        ("mimo_mean_condition_number", "平均信道条件数", False),
    )
    saved = []
    for key, ylabel, logarithmic in plots:
        values = np.array([item.get(key, np.nan) for item in results], dtype=float)
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        if logarithmic:
            axis.semilogy(snr, np.maximum(values, 1e-15), "o-")
        else:
            axis.plot(snr, values, "o-")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        path = output / f"{prefix}_{key}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        saved.append(path)
    return saved


if __name__ == "__main__":
    sweep = run_sweep(range(0, 31, 5))
    for saved_path in plot_sweep(sweep):
        print(saved_path)
