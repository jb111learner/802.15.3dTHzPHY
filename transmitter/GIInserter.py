import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler 
from transmitter.Encoder import Encoder
from transmitter.Modulator import THzModulator
class GIInserter:
    """
    CP 插入器 — SC-FDE / OFDM 双模式

    SC-FDE: 每 block = subframe_length 符号, 每帧 subframe_num 个 block
    OFDM:   每 block = subwave_num 符号 (一个OFDM符号), 每帧 subframe_ofdm_num 个 block
    """
    def __init__(self, params):
        self.params = params
        self.link_mode = (params.get("link_mode") or "sc-fde").lower()
        self.gi_length = params.get("gi_length")
        self.gi_type = params.get("gi_type")

        if self.link_mode == "ofdm":
            self.block_len = params.get("subwave_num")           # 512
            self.blocks_per_frame = params.get("subframe_ofdm_num")  # 48
        else:
            self.block_len = params.get("subframe_length")       # 480
            self.blocks_per_frame = params.get("subframe_num")   # 51

        # 输出参数
        self.sample_rate = None
        self.duration = None
        self.symbol_length = None
        self.padding_bit_num = 0
        self.frame_symbol_num = None
        self.frame_num = None

    def _verification_data(self, data_dict):
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num", "frame_symbol_num", "frame_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")

        fsym = data_dict["frame_symbol_num"]
        if fsym % self.block_len != 0:
            raise ValueError(f"frame_symbol_num({fsym}) 必须是 block_len({self.block_len}) 的整数倍")
        if fsym // self.block_len != self.blocks_per_frame:
            raise ValueError(f"frame_symbol_num/block_len({fsym//self.block_len}) != blocks_per_frame({self.blocks_per_frame})")

        n_blocks = data_dict["signal_length"] // self.block_len
        self.symbol_length = data_dict["signal_length"] + self.gi_length * n_blocks
        self.duration = data_dict["duration_seconds"]
        self.sample_rate = self.symbol_length / self.duration
        self.padding_bit_num = data_dict["padding_bit_num"]
        self.frame_symbol_num = fsym + self.gi_length * self.blocks_per_frame
        self.frame_num = data_dict["frame_num"]

    def insert_gi(self, data_dict):
        self._verification_data(data_dict)
        data = data_dict["signal_stream"]

        data_blocks = data.reshape(-1, self.block_len).T
        gi_blocks = np.concatenate([data_blocks[-self.gi_length:, :], data_blocks], axis=0)
        data_with_gi = gi_blocks.T.flatten()

        if len(data_with_gi) != self.symbol_length:
            raise ValueError(f"插入GI后长度不匹配: 预期{self.symbol_length}, 实际{len(data_with_gi)}")

        result_dict = {
            "signal_stream": data_with_gi,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
            "frame_symbol_num": self.frame_symbol_num,
            "frame_num": self.frame_num,
        }
        return result_dict

if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    scrambler = Scrambler(params)
    scrambled_data_dict = scrambler.scramble(res1)
    coder = Encoder(params)
    encoded_data_dict = coder.encode(scrambled_data_dict)
    modulator = THzModulator(params)
    symbols_dict = modulator.modulate(encoded_data_dict)
    print(f"符号长度：{symbols_dict['signal_length']}")
    print(f"采样率：{symbols_dict['sample_rate_Hz']} Hz，时长：{symbols_dict['duration_seconds']} 秒")
    print(f"补零数量：{symbols_dict['padding_bit_num']} bit")    
    gi_inserter = GIInserter(params)
    data_with_gi_dict = gi_inserter.insert_gi(symbols_dict)
    signal_power = np.mean(np.abs(data_with_gi_dict["signal_stream"]) ** 2)
    print(f"插入GI后信号功率：{signal_power}")
    print(f"符号长度：{data_with_gi_dict['signal_length']}")
    print(f"采样率：{data_with_gi_dict['sample_rate_Hz']} Hz，时长：{data_with_gi_dict['duration_seconds']} 秒")
    print(f"补零数量：{data_with_gi_dict['padding_bit_num']} bit")