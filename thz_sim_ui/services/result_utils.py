from pathlib import Path
import math
import re


def select_rx_constellation_data(result: dict) -> dict:
    """选择结果页星座图数据：IQ 补偿结果优先，均衡结果兜底。"""
    return (result.get("rx_iq_compensated")
            or result.get("rx_equalized")
            or {})


def build_simulation_result_path(
    project_folder: str | Path,
    task_name: str,
    run_id: int,
) -> Path:
    """为一次单方案仿真生成工程内独立且稳定的结果目录。"""
    safe_task_name = re.sub(r"[^0-9A-Za-z._-]+", "_", task_name).strip("._")
    safe_task_name = safe_task_name or "simulation"
    return Path(project_folder) / "runs" / f"{safe_task_name}_{int(run_id)}"


def compute_theoretical_phy_rate(params) -> float:
    """按调制、编码和实际帧结构计算理论物理层速率。"""
    symbol_rate = float(params.get("sample_rate"))
    bits_per_symbol = int(params.get("NCBPS"))
    streams = int(params.get("num_spatial_streams", 1)) if params.get("enable_mimo") else 1
    if str(params.get("code_type", "")).upper() == "RS":
        k = float(params.get("rs_packet_size"))
        n = k + float(params.get("rs_nsym"))
    else:
        from utils.LDPCMatrix import ieee802153d_1440_dimensions
        n, k, _ = ieee802153d_1440_dimensions(str(params.get("ldpc_standard_rate")))
    coding_efficiency = k / n

    gi = int(params.get("gi_length"))
    mode = str(params.get("link_mode")).lower()
    if mode == "ofdm":
        n_sc = int(params.get("subwave_num"))
        n_symbols = int(params.get("subframe_ofdm_num"))
        if params.get("enable_mimo"):
            n_pilots = len(list(params.get("pilot_block_indexes", [])))
            data_symbols = n_symbols - n_pilots
            blocks_per_stream = math.ceil(data_symbols / streams)
            frame_symbols = int(params.get("mimo_guard_samples", 32)) * 2
            frame_symbols += (2 + 2 + blocks_per_stream) * (n_sc + gi)
            payload_symbols_per_stream = data_symbols * n_sc / streams
        else:
            preamble = 3328 if str(params.get("Preamble_type")) == "short" else 5120
            n_pilots = len(list(params.get("pilot_block_indexes", [])))
            frame_symbols = preamble + n_symbols * (n_sc + gi)
            payload_symbols_per_stream = (n_symbols - n_pilots) * n_sc
    else:
        n_sc = int(params.get("subframe_length"))
        n_symbols = int(params.get("subframe_num"))
        preamble = 3328 if str(params.get("Preamble_type")) == "short" else 5120
        frame_symbols = preamble + n_symbols * (n_sc + gi)
        payload_symbols_per_stream = n_symbols * n_sc

    frame_efficiency = payload_symbols_per_stream / frame_symbols
    return symbol_rate * bits_per_symbol * streams * coding_efficiency * frame_efficiency
