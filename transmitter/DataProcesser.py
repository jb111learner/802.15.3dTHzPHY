import os
import random
import numpy as np
from params.PHYParams import PHYParams

class BitStreamProcessor:
    """
    比特流处理
    """
    def __init__(self, params):
        self.params = params
        self.file_path = self.params.get("file_path")      # 文件路径
        self.sample_rate = self.params.get("sample_rate")  # 采样率 (sps)
        self.duration = self.params.get("duration")        # 数据时长 (s)
        self.sample_length = self.params.get("sample_length")  # 采样点总数
        self.data_source = self.params.get("data_source")    # 数据源类型 ("PRBS"或"文件输入")
        self.subframe_num = self.params.get("subframe_num")  # 单数据帧子帧数量(frames)
        self.subframe_length = self.params.get("subframe_length")  # 数据子帧长度(symbols)
        self.NCBPS = self.params.get("NCBPS")  # 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
        self.code_type = self.params.get("code_type")  # 编码类型 ("RS"或"LDPC")
        if self.code_type == "RS":
            self.code_rate = self.params.get("rs_packet_size") / (self.params.get("rs_packet_size") + self.params.get("rs_nsym"))  # RS编码包大小
        elif self.code_type == "LDPC":
            self.code_rate = 14 / 15  # LDPC编码率(目前固定为14/15)
        link_mode = (self.params.get("link_mode")).lower()
        if link_mode == "ofdm":
            n_pilots = len(self.params.get("pilot_block_indexes"))
            n_data_syms = self.params.get("subframe_ofdm_num") - n_pilots
            n_sc = self.params.get("subwave_num")
            self.frame_bit_num = self.NCBPS * n_sc * n_data_syms * self.code_rate
        else:
            self.frame_bit_num = self.NCBPS * self.subframe_length * self.subframe_num * self.code_rate
        self.seed_strategy = self.params.get("seed_strategy")  # 随机种子策略
        self.random_seed = self.params.get("random_seed")
        self._seed_counter = 0
        self.save_big_bitstream = True  # 是否开启超大比特流保存
        self.big_bit_threshold = 10**10    # 超大比特流阈值(默认1000万bit)

        # 核心公共参数
        self.bit_stream = None    # 原始比特流
        self.bit_length = None      # 比特流长度
        self.is_big_bitstream = False  # 是否为超大比特流        
        self.padded_bitstream = None  # 补零后的比特流
        self.frame_num = None      # 最终的帧数
        self.padding_bit_num = 0   # 补零的比特数

    # ==================== 参数校验 ====================
    def _verification_param(self):
        """
        统一参数校验（适配 PRBS / 文件输入 两种数据源）

        PRBS 模式:
          - sample_rate、NCBPS 必须设置
          - duration 与 sample_length 二选一
        文件输入模式:
          - file_path 必须存在
        """
        # ———— 公共必设参数 ————
        if self.sample_rate is None or self.sample_rate <= 0:
            raise ValueError("sample_rate 必须设置为正数")
        if not isinstance(self.sample_rate, (int, float)):
            raise ValueError("sample_rate 必须是数值")
        if self.NCBPS is None or self.NCBPS <= 0:
            raise ValueError("NCBPS 必须设置为正整数")
        if not isinstance(self.NCBPS, int):
            raise ValueError("NCBPS 必须是整数")

        data_src = self.data_source

        if data_src == "PRBS":
            # duration 与 sample_length 二选一
            has_dur = self.duration is not None and self.duration > 0
            has_len = self.sample_length is not None and self.sample_length > 0

            if has_dur and has_len:
                raise ValueError("duration 与 sample_length 不能同时设置，请二选一")
            if not has_dur and not has_len:
                raise ValueError("duration 与 sample_length 必须设置其中一个")

            if has_len:
                if not isinstance(self.sample_length, int):
                    raise ValueError("sample_length 必须是整数")
                self.bit_length = self.sample_length * self.NCBPS
            else:
                if not isinstance(self.duration, (int, float)):
                    raise ValueError("duration 必须是数值")
                # sample_rate 表示调制后的基础符号率。每个符号承载 NCBPS
                # 个信息比特，因此给定持续时间内的符号数为 Rs*T，比特数
                # 为 Rs*T*NCBPS。
                self.sample_length = int(np.round(
                    self.sample_rate * self.duration))
                self.bit_length = self.sample_length * self.NCBPS

        elif data_src == "文件输入":
            if not self.file_path or not os.path.exists(self.file_path):
                raise FileNotFoundError(
                    f"文件路径无效或文件不存在：{self.file_path}")

        else:
            raise ValueError(f"不支持的数据源类型：{data_src}，仅支持 'PRBS' 或 '文件输入'")

    # ==================== PRBS 比特流生成 ====================
    def generate_random_bitstream(self):
        """PRBS 随机比特流生成（校验由 _verification_param 完成）"""
        strategy = str(self.seed_strategy or "").strip()
        configured_seed = self.params.get("random_seed")

        if configured_seed is not None:
            seed = int(configured_seed)
        elif strategy == "固定种子":
            seed = 0
        elif strategy == "递增种子":
            self._seed_counter += 1
            seed = self._seed_counter
        elif strategy == "时间种子" or strategy == "":
            import time
            seed = int((time.time() * 1e6) % (2 ** 32))
        else:
            raise ValueError(f"未知随机种子策略: {strategy}")

        rng = np.random.default_rng(seed)
        self.bit_stream = rng.integers(0, 2, size=self.bit_length, dtype=np.uint8)
        return self

    def file2bitstream(self):
        """文件转比特流"""
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件路径无效或文件不存在：{self.file_path}")

        with open(self.file_path, 'rb') as f:
            file_bytes = f.read()

        bit_list = []
        for byte in file_bytes:
            bit_str = bin(byte)[2:].zfill(8)
            bit_list.extend([int(bit) for bit in bit_str])   
        self.bit_stream = np.array(bit_list, dtype=np.uint8) 

        self.bit_length = len(bit_list)
        if self.duration is not None:
            self.sample_rate = self.bit_length / (self.duration * self.NCBPS)
        else:
            self.duration = self.bit_length / (self.sample_rate * self.NCBPS)    
        return self

    def frame_process(self):
        """比特流分帧处理"""
        if self.bit_stream is None:
            raise RuntimeError("请先生成随机比特流或读取文件比特流，再执行分帧操作")
        # frame_bit_num 可能是 Python/NumPy 的整数或浮点数。整数本身没有
        # float.is_integer()，因此应先统一转换再检查，不能依赖具体数值类型。
        try:
            frame_bit_num_value = float(self.frame_bit_num)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("单帧比特数必须是有限数值") from exc
        if not np.isfinite(frame_bit_num_value) or not frame_bit_num_value.is_integer():
            raise ValueError("请检查调制方式、帧结构和编码效率的配置，确保单帧比特数为整数")
        frame_bit_num = int(frame_bit_num_value)
        if frame_bit_num <= 0:
            raise ValueError("单帧比特数必须为正整数")
        self.frame_bit_num = frame_bit_num

        bit_stream_data = self.bit_stream
        
        total_bit = len(bit_stream_data)
        self.padding_bit_num = 0

        if total_bit % self.frame_bit_num != 0:
            self.padding_bit_num = self.frame_bit_num - (total_bit % self.frame_bit_num)
        padded_bitstream = np.pad(bit_stream_data, (0, self.padding_bit_num), mode='constant', constant_values=0)
        self.padded_bitstream = padded_bitstream
        self.frame_num = padded_bitstream.reshape(-1, self.frame_bit_num).shape[0]        
        self.bit_length = len(self.padded_bitstream) 
        self.duration = self.bit_length / (self.sample_rate * self.NCBPS)    
        return self

    def _get_result(self):
        """封装完整返回结果字典"""
        if self.save_big_bitstream and self.bit_length >= self.big_bit_threshold:
            save_path = f"big_bitstream_{self.bit_length}bits_{random.randint(1000,9999)}.npy"
            np.save(save_path, self.padded_bitstream)
            self.padded_bitstream = save_path
            self.is_big_bitstream = True

        result_dict = {
            "signal_stream": self.padded_bitstream,
            "sample_rate_Hz": self.sample_rate * self.NCBPS, 
            "duration_seconds": self.duration,
            "signal_length": self.bit_length,
            "frame_num": self.frame_num,
            "padding_bit_num": self.padding_bit_num,
            "is_big_bitstream": self.is_big_bitstream,
            "frame_bit_num": self.frame_bit_num,
        }
        return result_dict

    def run(self):
        """
        统一执行入口
          ① _verification_param: 参数校验 + 计算 bit_length
          ② PRBS / 文件输入 → 比特流
          ③ 分帧
        """
        self._verification_param()

        data_src = self.data_source
        if data_src == "PRBS":
            self.generate_random_bitstream()
        elif data_src == "文件输入":
            self.file2bitstream()
        else:
            raise ValueError(f"不支持的数据源类型：{data_src}")

        self.frame_process()
        return self._get_result()

    @staticmethod
    def load_big_bitstream(file_path: str):
        """加载比特流数组"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"比特流文件不存在：{file_path}")
        return np.load(file_path)
    
# ===================== 完整测试用例（含【文件模式自定义采样率】所有场景，必看） =====================
if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    print("=== 比特流处理测试 ===")
    print(f"前100比特流：{res1['signal_stream'][:100] if not res1['is_big_bitstream'] else '超大比特流，已保存为文件'}")
    print(f"比特流长度：{res1['signal_length']} bit")
    print(f"分帧配置：每帧{res1['frame_bit_num']}bit，共{res1['frame_num']}帧")
    print(f"采样率：{res1['sample_rate_Hz']} Hz，时长：{res1['duration_seconds']} 秒")
    print(f"补零数量：{res1['padding_bit_num']} bit")
    print(f"是否超大比特流：{res1['is_big_bitstream']}")
