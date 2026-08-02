"""正交训练符号驱动的 MIMO LS 信道估计。"""

from __future__ import annotations

import numpy as np


class MIMOChannelEstimator:
    @staticmethod
    def estimate(training_frequency, pilot_frequency, threshold=1e-12):
        training_frequency = np.asarray(training_frequency, dtype=np.complex128)
        pilot = np.asarray(pilot_frequency, dtype=np.complex128).reshape(-1)
        if training_frequency.ndim != 3:
            raise ValueError("training_frequency 必须为 (num_rx, num_tx, nfft)")
        if training_frequency.shape[-1] != len(pilot):
            raise ValueError("训练符号与导频长度不一致")
        safe = pilot.copy()
        weak = np.abs(safe) < threshold
        if np.any(weak):
            raise ValueError("导频包含零或过小元素，无法执行 LS 估计")
        return training_frequency / safe[np.newaxis, np.newaxis, :]

    @staticmethod
    def nmse(estimated, reference):
        estimated = np.asarray(estimated)
        reference = np.asarray(reference)
        numerator = np.linalg.norm(estimated - reference) ** 2
        denominator = np.linalg.norm(reference) ** 2 + np.finfo(float).eps
        return float(numerator / denominator)

