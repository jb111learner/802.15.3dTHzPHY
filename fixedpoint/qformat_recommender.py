"""范围分析只产生候选格式，必须通过链路比较后才能称为验证推荐。"""
import math
from .format import FixedPointFormat


def recommend_qformats(range_report, constraints=None):
    constraints = constraints or {}
    word_lengths = constraints.get("word_lengths", [12, 16, 18, 24])
    headroom_db = float(constraints.get("headroom_db", 3.0))
    if headroom_db < 0:
        raise ValueError("headroom_db must be nonnegative")
    multiplier = 10.0 ** (headroom_db / 20.0)
    nodes = range_report.get("ranges", range_report)
    results = {}
    for node, components in nodes.items():
        statistics = [components.get("real", {}), components.get("imag", {})]
        extrema = [float(s[k]) for s in statistics for k in ("min", "max") if s.get(k) is not None]
        if not extrema:
            continue
        bound = max(abs(value) for value in extrema) * multiplier
        # signed Q整数位包含符号；正上界严格小于2**(m-1)。
        integer_bits = max(1, int(math.floor(math.log2(bound))) + 2) if bound else 1
        candidates = []
        for word in word_lengths:
            if word >= integer_bits:
                fmt = FixedPointFormat(int(word), int(word) - integer_bits)
                if bound > fmt.maximum:
                    continue
                candidates.append({**fmt.to_dict(), "q_format": fmt.q_format,
                                   "lsb": 2.0 ** -fmt.fractional_length})
        results[node] = {"peak_bound": bound, "headroom_db": headroom_db,
                         "integer_bits_including_sign": integer_bits, "candidates": candidates,
                         "validated": False, "status": "range_candidates_only" if candidates else "no_feasible_word_length",
                         "validation_required": ["module_error", "EVM", "BER", "BLER", "SNR_loss", "overflow_rate"]}
    return {"validated": False, "nodes": results,
            "note": "范围候选尚未通过独立链路性能验证；FFT/累加器还需理论增长分析。"}
