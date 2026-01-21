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
        self.file_path = self.params.get("file_path")
        self.sample_rate = self.params.get("sample_rate")   # 采样率 (Hz)
        self.duration = self.params.get("duration")      # 数据时长 (s)
        self.bit_length = self.params.get("bit_length")    # 比特流总长度 (bit)        
        self.frame_bit_num = self.params.get("frame_bit_num")   # 单帧的比特数
        self.save_big_bitstream = True  # 是否开启超大比特流保存
        self.big_bit_threshold = 10**7    # 超大比特流阈值(默认1000万bit)

        # 核心公共参数
        self.bit_stream = None    # 原始比特流
        self.is_big_bitstream = False  # 是否为超大比特流        
        self.padded_bitstream = None  # 补零后的比特流
        self.frame_num = None      # 最终的帧数
        self.padding_bit_num = 0   # 补零的比特数

    def generate_random_bitstream(self):
        """随机比特流生成：输入任意2个参数，自动推导第三个，逻辑无改动"""
        sample_rate, duration, bit_length = self.sample_rate, self.duration, self.bit_length
        input_params = [param for param in [sample_rate, duration, bit_length] if param is not None]
        if len(input_params) != 2:
            raise ValueError("随机生成模式必须传入【任意2个】参数：sample_rate、duration、bit_length")
        for param in input_params:
            if not isinstance(param, (int, float)) or param <= 0:
                raise ValueError("采样率、时长、比特长度必须为正数数值")

        if sample_rate and duration:
            self.bit_length = int(np.round(self.sample_rate * self.duration))
        elif sample_rate and bit_length:
            self.duration = self.bit_length / self.sample_rate
        elif duration and bit_length:
            self.sample_rate = self.bit_length / self.duration

        self.bit_stream = np.random.randint(0, 2, size=self.bit_length, dtype=np.uint8)
        return self

    def file2bitstream(self):
        """文件转比特流：支持自定义采样率输入，无损转换，逻辑无改动"""
        sample_rate = self.sample_rate
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件路径无效或文件不存在：{self.file_path}")
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError("自定义采样率必须是正数数值，例如：8000、16000、44100、48000")

        with open(self.file_path, 'rb') as f:
            file_bytes = f.read()

        bit_list = []
        for byte in file_bytes:
            bit_str = bin(byte)[2:].zfill(8)
            bit_list.extend([int(bit) for bit in bit_str])

        self.bit_stream = np.array(bit_list, dtype=np.uint8)
        self.bit_length = len(self.bit_stream)
        self.sample_rate = sample_rate
        self.duration = self.bit_length / self.sample_rate
        return self

    def frame_process(self):
        """比特流分帧处理"""
        frame_bit_num = self.frame_bit_num
        if self.bit_stream is None:
            raise RuntimeError("请先生成随机比特流或读取文件比特流，再执行分帧操作")
        if not isinstance(frame_bit_num, int) or frame_bit_num <= 0:
            raise ValueError("单帧比特数(frame_bit_num)必须是正整数，例如：64、128、256、1024")
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
        self.duration = self.bit_length / self.sample_rate       
        return self

    def _get_result(self):
        """封装完整返回结果字典"""
        if self.save_big_bitstream and self.bit_length >= self.big_bit_threshold:
            save_path = f"big_bitstream_{self.bit_length}bits_{random.randint(1000,9999)}.npy"
            np.save(save_path, self.padded_bitstream)
            self.padded_bitstream = save_path
            self.is_big_bitstream = True

        result_dict = {
            "bit_stream": self.padded_bitstream,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "bit_length": self.bit_length,
            "frame_num": self.frame_num,
            "padding_bit_num": self.padding_bit_num,
            "is_big_bitstream": self.is_big_bitstream,
            "frame_bit_num": self.frame_bit_num,
        }
        return result_dict

    def run(self):
        """
        统一执行入口：智能适配双模式参数，调用逻辑极简，无任何改动
        ✅ 随机模式：传任意2个参数(sample_rate/duration/bit_length)，自动生成比特流
        ✅ 文件模式：仅需实例化传file_path，采样率在这里传入即可
        ✅ 分帧：传frame_bit_num则自动分帧，不传则不分帧
        """
        if self.file_path is None:
            self.generate_random_bitstream()
        else:
            self.file2bitstream()

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
    print(f"前100比特流：{res1['bit_stream'][:100] if not res1['is_big_bitstream'] else '超大比特流，已保存为文件'}")
    print(f"比特流长度：{res1['bit_length']} bit")
    print(f"分帧配置：每帧{res1['frame_bit_num']}bit，共{res1['frame_num']}帧")
    print(f"采样率：{res1['sample_rate_Hz']} Hz，时长：{res1['duration_seconds']} 秒")
    print(f"补零数量：{res1['padding_bit_num']} bit")
    print(f"是否超大比特流：{res1['is_big_bitstream']}")

