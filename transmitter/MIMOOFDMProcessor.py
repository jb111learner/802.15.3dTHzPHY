"""2×2/一般 Nt×Nr 空间复用 OFDM 发射波形生成。"""

from __future__ import annotations

import numpy as np

from transmitter.MIMOLayerMapper import MIMOLayerMapper
from utils.MIMOUtils import validate_mimo_params


class MIMOOFDMProcessor:
    """生成重复同步、正交训练和多空间流 OFDM 数据块。"""

    def __init__(self, params):
        self.params = params
        self.num_tx, _, self.num_streams = validate_mimo_params(params)
        self.nfft = int(params.get("subwave_num"))
        self.cp_length = int(params.get("gi_length"))
        if self.nfft < 2:
            raise ValueError("subwave_num 必须不小于 2")
        if not 0 <= self.cp_length < self.nfft:
            raise ValueError("gi_length 必须满足 0 <= gi_length < subwave_num")
        self.tx_scale = 1.0 / np.sqrt(self.num_tx)
        self.sync_frequency = self._build_bpsk_sequence(seed=1701)
        self.pilot_frequency = self._build_bpsk_sequence(seed=3203)

    def _build_bpsk_sequence(self, seed):
        rng = np.random.default_rng(seed + self.nfft)
        return (2 * rng.integers(0, 2, self.nfft) - 1).astype(np.complex128)

    def _with_cp(self, time_blocks):
        if self.cp_length == 0:
            return time_blocks
        return np.concatenate((time_blocks[..., -self.cp_length:], time_blocks), axis=-1)

    def process(self, modulated_dict):
        symbols = np.asarray(modulated_dict["signal_stream"], dtype=np.complex128).reshape(-1)
        frame_symbol_num = int(modulated_dict.get("frame_symbol_num", len(symbols)))
        frame_num = int(modulated_dict.get("frame_num", 1))
        if frame_num < 1:
            frame_num = 1

        guard_length = int(self.params.get("mimo_guard_samples", 32))
        if guard_length < 0:
            raise ValueError("mimo_guard_samples 不能为负")
        guard = np.zeros((self.num_tx, guard_length), dtype=np.complex128)

        # 每帧固定前导结构：2 SYNC + num_tx 训练（与 SISO 逐帧插前导一致）
        sync_grid = np.zeros((self.num_tx, 2, self.nfft), dtype=np.complex128)
        sync_grid[0, :, :] = self.sync_frequency

        training_grid = np.zeros(
            (self.num_tx, self.num_tx, self.nfft), dtype=np.complex128
        )
        for tx_index in range(self.num_tx):
            training_grid[tx_index, tx_index, :] = self.pilot_frequency

        # 逐帧组帧：每帧 = 保护间隔 + 2 SYNC + 训练 + 该帧数据块（含 CP）
        mapper = MIMOLayerMapper(self.num_streams)
        frame_waveforms = []
        layer_lengths = None
        num_data_symbols = None
        for frame_index in range(frame_num):
            frame_symbols = symbols[
                frame_index * frame_symbol_num:(frame_index + 1) * frame_symbol_num
            ]
            layers = mapper.map(frame_symbols)
            layer_lengths = [len(layer) for layer in layers]
            num_data_symbols = max(
                1, max((length + self.nfft - 1) // self.nfft for length in layer_lengths)
            )

            data_grid = np.zeros(
                (self.num_tx, num_data_symbols, self.nfft), dtype=np.complex128
            )
            for tx_index, layer in enumerate(layers):
                flat = data_grid[tx_index].reshape(-1)
                flat[:len(layer)] = layer * self.tx_scale

            frequency_grid = np.concatenate(
                (sync_grid, training_grid, data_grid), axis=1
            )
            time_blocks = np.fft.ifft(frequency_grid, axis=-1)
            blocks_with_cp = self._with_cp(time_blocks)
            frame_wave = blocks_with_cp.reshape(self.num_tx, -1)
            if guard_length:
                frame_wave = np.concatenate((guard, frame_wave), axis=1)
            frame_waveforms.append(frame_wave)

        waveform = np.concatenate(frame_waveforms, axis=1)
        if guard_length:
            waveform = np.concatenate((waveform, guard), axis=1)

        sample_rate = float(self.params.get("sample_rate"))
        frame_info = {
            "nfft": self.nfft,
            "cp_length": self.cp_length,
            "block_length": self.nfft + self.cp_length,
            "num_sync_symbols": 2,
            "num_training_symbols": self.num_tx,
            "num_data_symbols": num_data_symbols,       # 每帧数据块数
            "original_symbol_count": frame_symbol_num,  # 每帧符号数
            "layer_lengths": layer_lengths,             # 每帧每流符号数
            "frame_num": frame_num,
            "tx_scale": self.tx_scale,
            "guard_length": guard_length,
            "sync_frequency": self.sync_frequency.copy(),
            "pilot_frequency": self.pilot_frequency.copy(),
        }
        return {
            "signal_stream": waveform,
            "sample_rate_Hz": sample_rate,
            "duration_seconds": waveform.shape[1] / sample_rate,
            "signal_length": waveform.shape[1],
            "padding_bit_num": modulated_dict.get("padding_bit_num", 0),
            "num_tx": self.num_tx,
            "antenna_axis": 0,
            "mimo_frame": frame_info,
        }
