"""MIMO 链路共用的数组维度与参数校验工具。"""

from __future__ import annotations

import numpy as np


def ensure_antenna_axis(signal, expected_antennas=None, name="signal_stream"):
    """将 SISO/MIMO 信号统一成 ``(天线, 采样点)``。"""
    array = np.asarray(signal)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是一维或二维数组，实际维度为 {array.ndim}")
    if expected_antennas is not None and array.shape[0] != int(expected_antennas):
        raise ValueError(
            f"{name} 天线数不匹配：期望 {expected_antennas}，实际 {array.shape[0]}"
        )
    return array


def validate_mimo_params(params):
    """校验当前实现支持的 MIMO 组合。"""
    num_tx = int(params.get("num_tx", 1))
    num_rx = int(params.get("num_rx", 1))
    num_streams = int(params.get("num_spatial_streams", 1))
    if num_tx < 1 or num_rx < 1 or num_streams < 1:
        raise ValueError("num_tx、num_rx 和 num_spatial_streams 必须为正整数")
    if num_streams > min(num_tx, num_rx):
        raise ValueError("num_spatial_streams 不能超过 min(num_tx, num_rx)")
    if params.get("enable_mimo", False):
        if str(params.get("link_mode", "")).lower() != "ofdm":
            raise ValueError("当前 MIMO 实现仅支持 link_mode='ofdm'")
        if num_streams != num_tx:
            raise ValueError("当前空间复用实现要求 num_spatial_streams == num_tx")
        if str(params.get("mimo_scheme", "spatial_multiplexing")).lower() != "spatial_multiplexing":
            raise ValueError("当前仅支持 mimo_scheme='spatial_multiplexing'")
    return num_tx, num_rx, num_streams

