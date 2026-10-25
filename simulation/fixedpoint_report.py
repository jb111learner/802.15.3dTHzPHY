"""定点报告与比较图导出，JSON 禁止不可移植的 NaN/Infinity。"""
from pathlib import Path
import json
import math
import numpy as np

from fixedpoint.qformat_recommender import recommend_qformats


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def save_fixedpoint_report(report, output_dir):
    directory = Path(output_dir)
    if report["config"]["enable_fixed"]:
        recommended = {"validated": False, "nodes": {},
                       "status": "golden_float_ranges_required",
                       "note": "请使用浮点范围分析或配对比较生成Q格式候选；定点输出范围可能已饱和。"}
    else:
        recommended = recommend_qformats(report["ranges"])
    return {"config": write_json(directory / "fixedpoint_config.json", report["config"]),
            "coverage": write_json(directory / "coverage.json", report["coverage"]),
            "ranges": write_json(directory / "range_analysis.json", report["ranges"]),
            "diagnostics": write_json(directory / "quantization_diagnostics.json", report["diagnostics"]),
            "recommended": write_json(directory / "recommended_qformats.json", recommended),
            "semantics": write_json(directory / "arithmetic_semantics.json", report["arithmetic_semantics"])}


def save_comparison_report(result, output_dir):
    paths = save_fixedpoint_report(result["fixed_report"], output_dir)
    # 推荐应使用 golden 未饱和范围，不能由 fixed 范围反推动态范围。
    paths["recommended"] = write_json(Path(output_dir) / "recommended_qformats.json",
                                      recommend_qformats(result["float_report"]["ranges"]))
    paths["comparison"] = write_json(Path(output_dir) / "link_comparison.json",
                                      {k: v for k, v in result.items() if k != "arrays"})
    paths["modules"] = write_json(Path(output_dir) / "module_comparison.json", result["nmse"])
    return paths


def save_sweep_report(result, output_dir):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "link_comparison.json", result)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                                "mathtext.fontset": "dejavusans"}):
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        for axis, metric in zip(fig.subplots(1, 2), ("ber", "bler")):
            for side, label in (("float", "Float"), ("fixed", "Fixed")):
                x = [point["snr_db"] for point in result["points"]]
                y = [point[side][metric] if point[side][metric] else np.nan for point in result["points"]]
                line, = axis.semilogy(x, y, marker="o", label=label)
                zeros = [point for point in result["points"] if point[side][metric] == 0]
                if zeros:
                    axis.scatter([p["snr_db"] for p in zeros],
                                 [p[side][metric + "_interval"]["upper"] for p in zeros],
                                 marker="v" if side == "float" else "^", facecolors="none",
                                 edgecolors=line.get_color(), linewidths=1.2,
                                 label=label + " zero-error upper bound")
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(metric.upper())
            axis.grid(True, alpha=.3)
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / "comparison_curves.png", dpi=160)
    return str(directory / "comparison_curves.png")
