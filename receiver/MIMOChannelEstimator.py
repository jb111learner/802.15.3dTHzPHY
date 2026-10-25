"""正交训练符号驱动的 MIMO LS 信道估计。"""

from __future__ import annotations

import numpy as np
from receiver.FixedReceiverSupport import FixedReceiverMixin


class MIMOChannelEstimator(FixedReceiverMixin):
    def __init__(self, params=None):
        self._init_fixed(params, "mimo.rx.channel_estimation")

    def estimate_with_config(self, training_frequency, pilot_frequency, threshold=1e-12):
        if self._fixed_enabled():
            training = np.asarray(training_frequency, dtype=np.complex128)
            pilot = np.asarray(pilot_frequency, dtype=np.complex128).reshape(-1)
            # Reuse original validation before selecting the fixed division.
            self.estimate(training, pilot, threshold)
            return self.fixed_context.div(
                training, pilot[None, None, :], self.fixed_module_id,
                node=self.fixed_module_id + ".H"
            )
        result = self.estimate(training_frequency, pilot_frequency, threshold)
        self._watch(training_frequency, self.fixed_module_id + ".input")
        self._watch(result, self.fixed_module_id + ".H")
        return result

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

