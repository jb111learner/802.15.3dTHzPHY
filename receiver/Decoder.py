import numpy as np
from transmitter.THzTransmitter import THzTransmitter
from params.PHYParams import PHYParams
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.DeModulator import THzDemodulator
from utils.Coder import RSCoder, LDPCCoder

class Decoder:
    """
    支持按字节分包编码/解码。
    """
    def __init__(self, params):
        self.params = params
        self.code_type = self.params.get("code_type")   # 编码类型
        if self.code_type not in ["RS", "LDPC"]:
            raise ValueError("当前仅支持RS解码, LDPC解码")
        if self.code_type == "RS":
            self.decoder = RSCoder(self.params)
            self.nsym = self.decoder.nsym
            self.packet_size = self.decoder.packet_size
            self.efficiency = self.packet_size / (self.packet_size + self.nsym)
            self.decode_mode = self.params.get("decode_mode")
        elif self.code_type == "LDPC":
            self.decoder = LDPCCoder(self.params)
            self.nsym = self.decoder.r          # parity bits
            self.packet_size = self.decoder.k    # info bits
            self.efficiency = self.decoder.k / self.decoder.n

        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.bit_length = None  # 比特长度
        self.padding_bit_num = 0   # 补零的比特数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 比特长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        self.sample_rate = data_dict["sample_rate_Hz"] * self.efficiency
        self.duration = data_dict["duration_seconds"]
        self.bit_length = data_dict["signal_length"] * self.efficiency
        self.padding_bit_num = data_dict["padding_bit_num"]   

    def decode(self, data_dict):
        """
        输入：数据字典
        输出：解码后数据字典
        """
        self._verification_data(data_dict)
        llr = data_dict["signal_stream"]
        if self.code_type == "RS":
            if self.decode_mode == "hard":
                bits_encoded = self.decoder.decode(llr, int(self.bit_length))
            elif self.decode_mode == "chase":
                bits_encoded = self.decoder.decode_chase(
                    llr, int(self.bit_length), num_chase_per_packet=self.params.get("chase_num_per_packet")
                )
        elif self.code_type == "LDPC":
            bits_encoded, _ = self.decoder.decode(llr, original_bit_len=int(self.bit_length))
        if len(bits_encoded) < round(self.bit_length):
            bits_encoded = np.pad(bits_encoded, (0, round(self.bit_length) - len(bits_encoded)))
        elif len(bits_encoded) > round(self.bit_length):
            bits_encoded = bits_encoded[:round(self.bit_length)]

        result_dict = {
            "signal_stream": bits_encoded,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": round(self.bit_length),
            "padding_bit_num": self.padding_bit_num,
        }   
        # 转换为比特流返回
        return result_dict

    # def decode_with_auto_erase(self, data_dict, num_erasures_per_packet=2):
    #     """
    #     输入：数据字典
    #     输出：解码后数据字典
    #     """
    #     self._verification_data(data_dict)
    #     llr = data_dict["signal_stream"]
    #     bits_encoded = self.decoder.decode_with_auto_erase(llr, int(self.bit_length), num_erasures_per_packet)
    #     # bits_encoded = self.decoder.decode(llr, int(self.bit_length), decode_mode=self.decode_mode)
    #     if len(bits_encoded) != round(self.bit_length):
    #         raise ValueError(f"解码后比特长度不匹配：期望 {round(self.bit_length)}，实际 {len(bits_encoded)}")

    #     result_dict = {
    #         "signal_stream": bits_encoded,
    #         "sample_rate_Hz": self.sample_rate,
    #         "duration_seconds": self.duration,
    #         "signal_length": round(self.bit_length),
    #         "padding_bit_num": self.padding_bit_num,
    #     }   
    #     # 转换为比特流返回
    #     return result_dict
    
# if __name__ == "__main__":
#     # 初始化参数和发射机
#     params = PHYParams()
#     params.update(SNRdB=15)
#     transmitter = THzTransmitter(params)    
#     # 执行完整发射流程  
#     tx_signal_dict = transmitter.run() 
#     # 初始化信道并生成接收信号
#     channel = THzChannel(params)
#     rx_signal_dict = channel.run(tx_signal_dict)
#     # 初始化接收端匹配滤波器
#     rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
#     # 进行匹配滤波并获得中间信号
#     rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)
#     # 初始化GI移除器
#     gi_remover = GIRemover(params)
#     # 进行GI移除
#     rx_data_no_gi_dict = gi_remover.remove_gi(rx_matched_dict)
#     # 解调
#     snr_db = params.get("SNRdB")    
#     snr_linear = 10 ** (snr_db / 10)
#     sigma = np.sqrt(1.00 / (2 * snr_linear))
#     demodulator = THzDemodulator(params)
#     llr_dict = demodulator.demodulate(rx_data_no_gi_dict, sigma)
#     # 初始化解码器
#     decoder = Decoder(params)
#     # 进行解码
#     decoded_bits_dict = decoder.decode(llr_dict)
#     data_scrambled_dict = transmitter.data_scrambled_dict
#     print(f"\n===== 解码结果分析 =====")
#     print(f"解码比特长度：{len(decoded_bits_dict['signal_stream'])}, 原始比特长度：{len(data_scrambled_dict['signal_stream'])}")
#     print(f"原始比特（前10）：{data_scrambled_dict['signal_stream'][0:10]}")
#     print(f"解码比特（前10）：{decoded_bits_dict['signal_stream'][0:10]}")
#     print(f"比特错误数：{np.sum(data_scrambled_dict['signal_stream'] != decoded_bits_dict['signal_stream'])}")
#     print(f"误码率：{np.sum(data_scrambled_dict['signal_stream'] != decoded_bits_dict['signal_stream'])/len(data_scrambled_dict['signal_stream']):.6f}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

    def run_simulation(snr_db, num_frames=10):
        """
        运行一次仿真，返回普通译码和擦除译码的 BER
        num_frames: 发送的数据帧数（每帧长度由 PHYParams 决定）
        """
        params = PHYParams()
        params.update(SNRdB=snr_db)
        
        total_bits = 0
        err_plain = 0
        err_erase = 0
        
        for _ in range(num_frames):
            # 每次循环重新创建发射机、信道等，确保数据随机性
            transmitter = THzTransmitter(params)
            tx_signal_dict = transmitter.run()
            
            channel = THzChannel(params)
            rx_signal_dict = channel.run(tx_signal_dict)
            
            rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
            rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)
            
            gi_remover = GIRemover(params)
            rx_data_no_gi_dict = gi_remover.remove_gi(rx_matched_dict)
            
            snr_linear = 10 ** (snr_db / 10)
            sigma = np.sqrt(1.00 / (2 * snr_linear))
            demodulator = THzDemodulator(params)
            llr_dict = demodulator.demodulate(rx_data_no_gi_dict, sigma)
            
            original_bits = transmitter.data_scrambled_dict['signal_stream']
            total_bits += len(original_bits)
            
            # 普通译码
            decoder = Decoder(params)
            decoded_plain_dict = decoder.decode(llr_dict)
            err_plain += np.sum(original_bits != decoded_plain_dict['signal_stream'])
            
            # 擦除译码（每个包擦除2个最不可靠符号）
            decoded_erase_dict = decoder.decode_with_auto_erase(llr_dict, num_erasures_per_packet=2)
            err_erase += np.sum(original_bits != decoded_erase_dict['signal_stream'])
        
        ber_plain = err_plain / total_bits if total_bits > 0 else 1.0
        ber_erase = err_erase / total_bits if total_bits > 0 else 1.0
        return ber_plain, ber_erase

    # 设置 SNR 测试点
    snr_dbs = np.arange(-5, 8, 2)   # [-5, -3, -1, 1, 3, 5, 7]
    ber_plain_list = []
    ber_erase_list = []

    # 每个 SNR 点运行 10 帧（若帧内数据量较小可增加）
    num_frames_per_snr = 2

    print("开始多信噪比仿真...")
    for snr in tqdm(snr_dbs):
        ber_p, ber_e = run_simulation(snr, num_frames=num_frames_per_snr)
        ber_plain_list.append(ber_p)
        ber_erase_list.append(ber_e)
        print(f"SNR={snr} dB: Plain BER={ber_p:.2e}, Erasure BER={ber_e:.2e}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_dbs, ber_plain_list, 'o-', label='Plain Decoding')
    plt.semilogy(snr_dbs, ber_erase_list, 's-', label='Erasure Decoding (auto 2 erasures)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('RS Code Performance Comparison')
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.show()
    