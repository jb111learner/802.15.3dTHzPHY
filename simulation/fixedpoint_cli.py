"""数值仿真与分析命令入口；运行模式仍只有浮点/定点。"""
import argparse
import json
from pathlib import Path

import numpy as np

from params.PHYParams import PHYParams
from fixedpoint import FixedPointConfig, recommend_qformats
from .SimulationManager import SimulationManager
from .fixedpoint_comparison import run_float_fixed_comparison, run_comparison_sweep
from .fixedpoint_report import save_fixedpoint_report, write_json


OPERATION_MODULES = {
    "ofdm": ["ofdm.tx.ifft", "ofdm.rx.fft", "ofdm.rx.channel_estimation", "ofdm.rx.equalizer"],
    "sc-fde": ["sc.tx.pulse_shaping", "sc.rx.matched_filter", "sc.rx.coarse_sync",
               "sc.rx.fine_sync", "sc.rx.channel_estimation", "sc.rx.equalizer", "sc.rx.cfo"],
    "mimo": ["mimo.tx.ifft", "mimo.rx.fft", "mimo.rx.channel_estimation", "mimo.rx.detector"],
}


def demo_params(link="ofdm", snr=30., seed=2026):
    """一帧工程 LDPC 小样本，用于功能验证，不作为性能确认。"""
    params = PHYParams()
    values = dict(random_seed=seed, seed_strategy="固定种子", NCBPS=4, code_type="LDPC",
                  ldpc_matrix_type="engineering", ldpc_n=64, ldpc_k=32,
                  duration=None, SNRdB=snr, enable_multipath=False,
                  enable_cfo=False, enable_iq_imbalance=False, enable_pa=False)
    values["enable_iq_compensation"] = False
    if link == "mimo":
        values.update(enable_mimo=True, link_mode="ofdm", num_tx=2, num_rx=2,
                      num_spatial_streams=2, subwave_num=64, gi_length=8,
                      subframe_ofdm_num=4, pilot_block_indexes=[],
                      mimo_channel_model="identity", mimo_num_taps=1,
                      oversampling=1, sample_length=128)
    else:
        values.update(link_mode=link, sample_length=11520 if link == "ofdm" else 12240)
    params.update(**values)
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description="浮点/定点仿真及独立分析工具")
    parser.add_argument("task", choices=["run", "compare", "sweep", "recommend"])
    parser.add_argument("--link", choices=["ofdm", "sc-fde", "mimo"], default="ofdm")
    parser.add_argument("--config", type=Path, help="定点配置 JSON")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fixed", dest="enable_fixed", action="store_true")
    modes.add_argument("--float", dest="enable_fixed", action="store_false")
    parser.set_defaults(enable_fixed=None)
    parser.add_argument("--operation", action="store_true", help="优先模块启用内部整数内核")
    parser.add_argument("--range-analysis", action="store_true")
    parser.add_argument("--snr", type=float, default=30.)
    parser.add_argument("--snr-values", type=float, nargs="+", default=[10., 15., 20., 25., 30.])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("simulation_results/fixedpoint/latest"))
    parser.add_argument("--range-report", type=Path)
    args = parser.parse_args(argv)
    if args.task == "recommend":
        if args.range_report is None:
            parser.error("recommend requires --range-report from floating reference")
        ranges = json.loads(args.range_report.read_text(encoding="utf-8"))
        result = recommend_qformats(ranges)
        write_json(args.output / "recommended_qformats.json", result)
        print("已生成范围候选；尚未通过独立性能验证。")
        return result
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.enable_fixed is not None:
        config["enable_fixed"] = args.enable_fixed
    if args.operation:
        modules = config.setdefault("modules", {})
        for name in OPERATION_MODULES[args.link]:
            modules.setdefault(name, {})["arithmetic_level"] = "operation"
    FixedPointConfig.from_dict(config)
    params = demo_params(args.link, args.snr, args.seed)
    params.update(fixed_point=config, range_analysis={"enabled": args.range_analysis})
    if args.task == "compare":
        result = run_float_fixed_comparison(params, config, {"seed": args.seed, "output_dir": str(args.output)})
        print("浮点 BER:", result["float"]["ber"], "定点 BER:", result["fixed"]["ber"])
        print("定点 EVM(%):", result["fixed"]["evm"]["percent"])
    elif args.task == "sweep":
        result = run_comparison_sweep(params, args.snr_values, config, runs=args.runs, seed=args.seed,
                                      target=args.target, output_dir=args.output)
        print("SNR loss:", result["snr_loss"])
    else:
        np.random.seed(args.seed % 2**32)
        manager = SimulationManager(params, output_dir=args.output, save_plots=False)
        result = manager.run_once()
        report = result["result"].get("fixed_point_report")
        if report is not None:
            save_fixedpoint_report(report, args.output)
        write_json(args.output / "metrics.json", result["metrics"])
        print("数值模式:", "定点" if config.get("enable_fixed", False) else "浮点", "BER:", result["metrics"]["ber"])
    return result


if __name__ == "__main__":
    main()
