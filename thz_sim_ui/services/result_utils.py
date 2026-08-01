from pathlib import Path
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
