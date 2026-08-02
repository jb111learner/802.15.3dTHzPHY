"""逐子载波 ZF/MMSE MIMO 检测。"""

from __future__ import annotations

import numpy as np


class MIMODetector:
    def __init__(self, method="mmse"):
        self.method = str(method).lower()
        if self.method not in {"zf", "mmse"}:
            raise ValueError("MIMO 检测算法仅支持 'zf' 或 'mmse'")

    def detect(self, received, channel, noise_var=0.0):
        """
        received: (num_rx, num_symbols, nfft)
        channel:  (num_rx, num_streams, nfft)
        """
        received = np.asarray(received, dtype=np.complex128)
        channel = np.asarray(channel, dtype=np.complex128)
        if received.ndim != 3 or channel.ndim != 3:
            raise ValueError("received 和 channel 均必须是三维数组")
        num_rx, num_symbols, nfft = received.shape
        if channel.shape[0] != num_rx or channel.shape[2] != nfft:
            raise ValueError("接收资源网格与 MIMO 信道矩阵维度不一致")
        num_streams = channel.shape[1]
        detected = np.empty((num_streams, num_symbols, nfft), dtype=np.complex128)
        post_variances = np.empty((num_streams, nfft), dtype=float)
        condition_numbers = np.empty(nfft, dtype=float)
        identity = np.eye(num_streams, dtype=np.complex128)

        for subcarrier in range(nfft):
            h = channel[:, :, subcarrier]
            y = received[:, :, subcarrier]
            condition_numbers[subcarrier] = np.linalg.cond(h)
            if self.method == "zf":
                weight = np.linalg.pinv(h)
            else:
                gram = h.conj().T @ h + float(noise_var) * identity
                try:
                    weight = np.linalg.solve(gram, h.conj().T)
                except np.linalg.LinAlgError:
                    weight = np.linalg.pinv(gram) @ h.conj().T
            detected[:, :, subcarrier] = weight @ y
            covariance = float(noise_var) * (weight @ weight.conj().T)
            post_variances[:, subcarrier] = np.maximum(np.real(np.diag(covariance)), 0.0)

        diagnostics = {
            "post_noise_variance": post_variances,
            "condition_numbers": condition_numbers,
            "mean_condition_number": float(np.mean(condition_numbers)),
        }
        return detected, diagnostics

