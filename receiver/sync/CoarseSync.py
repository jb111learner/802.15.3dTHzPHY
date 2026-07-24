import numpy as np
from scipy.signal import correlate


class CoarseSync:
    """
    粗同步：基于 SYNC 互相关的帧起始定位

    处理流程（放在 RxMatchedFilter 与 CFO 粗估计之间）：
      1. 输入过采样信号（匹配滤波后）
      2. 本地 SYNC 序列与接收信号互相关 → 定位 SYNC 起始位置
      3. 从检测位置截取有效信号段
      4. 输出过采样信号字典 + sync_offset（供下游 FineSync 使用）

    注意：本类不下采样，下采样由 Downsampler 完成。
    """

    def __init__(self, transmitter):
        self.params = transmitter.params
        self.oversampling = self.params.get("oversampling")

        # ———— 本地 SYNC 参考（过采样域） ————
        self.sync_ref = transmitter.sync_upsampled    # SYNC 序列（过采样）
        self.sync_len_oversample = len(self.sync_ref)

        # ———— 输出参数 ————
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
        self.sync_offset = None       # 检测到的 SYNC 起始位置（过采样索引）
        self.corr_peak = None         # 互相关峰值

    @staticmethod
    def _valid_correlation(signal, reference):
        """使用 FFT 计算与 np.correlate(..., mode='valid') 等价的互相关。"""
        return correlate(signal, reference, mode="valid", method="fft")

    def _verification_data(self, data_dict):
        """校验输入数据字典（过采样域）"""
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        total_len = data_dict["signal_length"]
        base_rate = data_dict["sample_rate_Hz"]

        if round(base_rate * data_dict["duration_seconds"]) != total_len:
            raise ValueError("采样率与时长不匹配")
        if self.sync_len_oversample >= total_len:
            raise ValueError("SYNC 参考序列长度超过信号总长度")

        self.padding_bit_num = data_dict["padding_bit_num"]

    def detect_sync(self, data_dict):
        """
        执行 SYNC 互相关定位 + 信号对齐

        :param data_dict: 过采样接收信号字典（来自 RxMatchedFilter）
        :return: result_dict — 过采样信号字典（从 SYNC 起始处截取），
                 包含 sync_offset 供下游 FineSync 使用
        """
        self._verification_data(data_dict)
        rx_signal = data_dict["signal_stream"]

        # ———— 1. SYNC 互相关 → 帧起始 ————
        search_len = min(len(rx_signal), self.sync_len_oversample * 5)
        corr = np.abs(self._valid_correlation(
            rx_signal[:search_len], self.sync_ref
        ))
        sync_start = np.argmax(corr)
        self.sync_offset = sync_start
        self.corr_peak = corr[sync_start]
        print(f"  [CoarseSync] SYNC start @ index {sync_start}, "
              f"corr peak = {self.corr_peak:.1f}")

        # ———— 2. 从 SYNC 起始处截取信号 ————
        aligned_signal = rx_signal[sync_start:]

        # ———— 3. 输出 ————
        self.signal_length = len(aligned_signal)
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.duration = self.signal_length / self.sample_rate

        result_dict = {
            "signal_stream": aligned_signal,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
            "sync_offset": self.sync_offset,
        }
        return result_dict


# ==================== 测试 ====================
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    from receiver.MatchedFilter import RxMatchedFilter

    for mode in ["sc-fde", "ofdm"]:
        print(f"\n{'='*50}")
        print(f"     CoarseSync 测试 — {mode.upper()} 模式")
        print(f"{'='*50}")

        params = PHYParams()
        params.update(link_mode=mode)
        tx = THzTransmitter(params)
        tx.run()
        ch = THzChannel(params)
        rx_dict = ch.run(tx.tx_signal_dict)

        # 匹配滤波
        mf = RxMatchedFilter(tx)
        mf_out = mf.matched_filter(rx_dict)

        # 粗同步（仅定位，不下采样）
        cs = CoarseSync(tx)
        sync_out = cs.detect_sync(mf_out)

        # 验证：从检测位置开始的信号应与 TX 对齐
        tx_sync_oversampled = tx.sync_upsampled
        rx_seg = sync_out["signal_stream"][:len(tx_sync_oversampled)]
        corr = np.abs(np.dot(rx_seg.conj(), tx_sync_oversampled))
        print(f"  SYNC corr after alignment: {corr:.1f}")
        print(f"  sync_offset = {cs.sync_offset}")
        print(f"  Output length = {sync_out['signal_length']}")
        print(f"  OK" if corr > 100 else "  CHECK")

    print("\nDone")
