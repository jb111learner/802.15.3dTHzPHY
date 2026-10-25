"""逐子载波 ZF/MMSE MIMO 检测。"""

from __future__ import annotations

import numpy as np
from receiver.FixedReceiverSupport import FixedReceiverMixin


class MIMODetector(FixedReceiverMixin):
    def __init__(self, method="mmse", *, params=None):
        self._init_fixed(params, "mimo.rx.detector")
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
        if self._fixed_enabled() and self.fixed_context.arithmetic_level(self.fixed_module_id) == "operation":
            return self._detect_fixed(received, channel, noise_var)
        self._boundary_only("pinv/solve and singular-matrix fallback are not implemented in integer arithmetic")
        received = self._fq(received, role="input", node="mimo.rx.detector.received")
        channel = self._fq(channel, role="coefficient", node="mimo.rx.detector.H")
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
        detected = self._fq(detected, node="mimo.rx.detector.output")
        if self._fixed_enabled():
            self.fixed_context.record_coverage(self.fixed_module_id, "boundary", "matrix solve computed in float; input/H/output quantized")
        return detected, diagnostics

    def _detect_fixed(self, received, channel, noise_var):
        """满秩 ZF/MMSE 矩阵运算；量化后奇异时明确失败。"""
        if received.ndim != 3 or channel.ndim != 3:
            raise ValueError("received 和 channel 均必须是三维数组")
        num_rx, num_symbols, nfft = received.shape
        if channel.shape[0] != num_rx or channel.shape[2] != nfft:
            raise ValueError("接收资源网格与 MIMO 信道矩阵维度不一致")
        ctx, mid = self.fixed_context, self.fixed_module_id
        num_streams = channel.shape[1]
        detected = np.empty((num_streams, num_symbols, nfft), dtype=np.complex128)
        variances = np.empty((num_streams, nfft), dtype=float)
        conditions = np.empty(nfft, dtype=float)
        for subcarrier in range(nfft):
            h = ctx.quantize(channel[:, :, subcarrier], mid, role="matrix", node=mid + ".H")
            y = ctx.quantize(received[:, :, subcarrier], mid, role="input", node=mid + ".received")
            conditions[subcarrier] = np.linalg.cond(h)  # diagnostic only
            if self.method == "zf":
                weight = ctx.pinv(h, mid, input_role="matrix", node=mid + ".weight")
            else:
                hermitian = h.conj().T
                gram = ctx.matmul(hermitian, h, mid, a_role="matrix", b_role="matrix", node=mid + ".gram")
                regularization = ctx.quantize(float(noise_var) * np.eye(num_streams), mid,
                                              role="noise", node=mid + ".regularization")
                gram = ctx.add(gram, regularization, mid, role="matrix", a_role="matrix", b_role="noise")
                # Root integer solver handles full rank; there is no hidden float pinv fallback.
                weight = ctx.solve(gram, hermitian, mid, a_role="matrix", b_role="matrix", node=mid + ".weight")
            detected[:, :, subcarrier] = ctx.matmul(weight, y, mid, role="output",
                                                     a_role="matrix", b_role="input", node=mid + ".output")
            covariance = ctx.matmul(weight, weight.conj().T, mid, a_role="matrix", b_role="matrix")
            covariance = ctx.mul(float(noise_var), covariance, mid, role="noise",
                                 a_role="noise", b_role="matrix", node=mid + ".noise_covariance")
            variances[:, subcarrier] = np.maximum(np.real(np.diag(covariance)), 0.0)
        ctx.record_coverage(mid, "operation", "full-rank integer ZF/MMSE; rank-deficient fixed solve raises; condition number is float diagnostic")
        return detected, {
            "post_noise_variance": variances,
            "condition_numbers": conditions,
            "mean_condition_number": float(np.mean(conditions)),
            "singular_matrix_policy": "error",
        }

