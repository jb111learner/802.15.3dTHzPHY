import numpy as np
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler 
from transmitter.Encoder import Encoder
from transmitter.Modulator import THzModulator
from transmitter.GIInserter import GIInserter
from transmitter.TxOFDMProcesser import TxOFDMProcesser
from transmitter.PreambleInsertor import PreambleInsertor
from scipy.signal import firwin
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False 

class TxPulseShaper:
    """
    发送端脉冲成型类：RC/RRC/矩形滤波器
    """

    def __init__(self, params):
        self.sps = params.get("oversampling")
        self.rolloff = params.get("rolloff")
        self.filter_type = params.get("filter_type").lower()
        self.filter_length = params.get("filter_length")
        self.link_mode = params.get("link_mode").lower()   # 新增        
        self._validate_params()

        self.filter_coeffs = None

        # 输出参数    
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.symbol_length = None  # 符号长度
        self.padding_bit_num = 0   # 补零的比特数   

    def _validate_params(self):
        """参数校验（适配两种模式）"""
        if self.sps < 1 or not isinstance(self.sps, int):
            raise ValueError(f"上采样率必须为正整数，当前值：{self.sps}")

        if self.link_mode not in ("sc-fde", "ofdm"):
            raise ValueError(f"link_mode 必须为 'sc-fde' 或 'ofdm'，当前值：{self.link_mode}")

        if self.link_mode == "sc-fde":
            if not (0 <= self.rolloff <= 1):
                raise ValueError(f"滚降系数必须在0~1之间，当前值：{self.rolloff}")
            if self.filter_type not in ("rc", "rrc", "rect"):
                raise ValueError(f"SC-FDE模式仅支持 rc/rrc/rect，当前值：{self.filter_type}")
        else:   # OFDM 模式
            # 忽略 filter_type 和 rolloff，内部强制使用低通
            pass
    
    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 符号流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 符号长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        self.symbol_length = data_dict["signal_length"] * self.sps
        self.duration = data_dict["duration_seconds"]
        self.sample_rate = data_dict["sample_rate_Hz"] * self.sps
        self.padding_bit_num = data_dict["padding_bit_num"]     

    def _rrc_impulse_response(self, span_symbols, sps, beta, T):
        """
        返回长度 = span_symbols*sps + 1 的 RRC (root-raised-cosine) 冲激响应（矢量化）
        """
        num = int(span_symbols * sps)
        m = np.arange(-num // 2, num // 2 + 1)
        t = m * T / sps                      # 时间坐标（秒）

        h = np.zeros_like(t, dtype=np.float64)
        pi = np.pi
        for idx, ti in enumerate(t):
            if abs(ti) < 1e-12:
                # t = 0
                h[idx] = (1.0 / np.sqrt(T)) * (1.0 + beta * (4 / pi - 1.0))
            elif beta > 0 and abs(abs(ti) - T / (4 * beta)) < 1e-12:
                # 奇异点 t = ±T/(4β)
                h[idx] = (beta / (np.sqrt(2 * T))) * (
                    (1 + 2 / pi) * np.sin(pi / (4 * beta)) +
                    (1 - 2 / pi) * np.cos(pi / (4 * beta))
                )
            else:
                numerator = np.sin(pi * ti * (1 - beta) / T) + \
                            4 * beta * ti / T * np.cos(pi * ti * (1 + beta) / T)
                denominator = pi * ti / T * (1 - (4 * beta * ti / T) ** 2) * np.sqrt(T)
                h[idx] = numerator / denominator

        # 去掉可能的 NaN 或 Inf（极小分母产生的）
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h


    def _rc_impulse_response(self, span_symbols, sps, beta, T):
        """
        返回长度 = span_symbols*sps + 1 的 RC (raised-cosine) 冲激响应（直接公式）
        RC(t) = sinc(t/T) * cos(π β t/T) / (1 - (2β t/T)^2)
        为保证与 RRC 相同长度和对称性，直接采样并归一化
        """
        num = int(span_symbols * sps)
        m = np.arange(-num // 2, num // 2 + 1)
        t = m * T / sps                      # 时间坐标（秒）

        h = np.zeros_like(t, dtype=np.float64)
        pi = np.pi
        for idx, ti in enumerate(t):
            if abs(ti) < 1e-12:
                # t = 0
                h[idx] = 1.0   # 未归一化的峰值（后续会整体归一化）
            elif beta > 0 and abs(abs(ti) - T / (2 * beta)) < 1e-12:
                # 奇异点 t = ±T/(2β)
                # 使用极限公式：h = (π/4) * sinc(1/(2β)) ?? 为简化，取相邻两点平均
                # 更精确：极限值为 (π/4) * sinc(1/(2β))，但直接相邻平均足够稳定
                left = ti - 1e-12 * T
                right = ti + 1e-12 * T
                # 递归计算两个临近点的值（避免重复代码，直接调用自身但不推荐，这里简单插值）
                # 采用近似：忽略该点，令其为左右均值（数值上误差极小，归一化后无影响）
                # 实际上因为后续会整体归一化，且该点能量占比极小，置0亦可
                # 为保证分母不为零，直接计算极限表达式：
                arg = pi * ti / T
                val = np.sin(arg) / arg * np.cos(pi * beta * ti / T) / (1 - (2 * beta * ti / T) ** 2)
                # 但这里分母为零，需要单独计算极限：h = (π/4) * sinc(1/(2β)) * (β? )
                # 为简洁且稳定，直接计算左右两点均值：
                t_left = ti - 1e-9
                t_right = ti + 1e-9
                def _safe_rc(t_):
                    if abs(t_) < 1e-12:
                        return 1.0
                    arg = pi * t_ / T
                    sinc = np.sin(arg) / arg
                    cos_term = np.cos(pi * beta * t_ / T)
                    denom = 1 - (2 * beta * t_ / T) ** 2
                    if abs(denom) < 1e-12:
                        return 0.0
                    return sinc * cos_term / denom
                h[idx] = (_safe_rc(t_left) + _safe_rc(t_right)) / 2.0
            else:
                arg = pi * ti / T
                sinc = np.sin(arg) / arg
                cos_term = np.cos(pi * beta * ti / T)
                denom = 1 - (2 * beta * ti / T) ** 2
                # denom 理论上在非奇异点不为零，但数值上可能很小，加保护
                if abs(denom) < 1e-12:
                    h[idx] = 0.0
                else:
                    h[idx] = sinc * cos_term / denom

        # 替换 NaN 或 Inf
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h
    
    # ---------- OFDM 上采样低通滤波器设计 ----------
    def _design_lowpass_filter(self):
        """OFDM 模式下统一使用通用低通滤波器"""
        # sps=1 时没有插零产生的频谱镜像，无需插值低通。此时若继续按
        # 1/sps 设计，cutoff 会等于 Nyquist（归一化频率 1.0），而
        # scipy.signal.firwin 要求截止频率严格小于 Nyquist。
        if self.sps == 1:
            return np.array([1.0], dtype=np.float64)

        # 1. 设计低通，截止频率 = 1 / sps
        num_taps = self.filter_length * self.sps + 1
        if num_taps % 2 == 0:
            num_taps += 1  # 保持奇数，方便延迟对齐
        
        cutoff = 1.0 / self.sps
        h = firwin(num_taps, cutoff, window='hamming')
        
        # 2. 关键：补偿插零带来的幅度衰减（直流增益 = sps）
        h = h / np.sum(h) * self.sps
        return h

    # ---------- 统一滤波器设计入口 ----------
    def _design_tx_filter(self):
        """根据 link_mode 选择滤波器设计策略"""
        if self.link_mode == "ofdm":
            return self._design_lowpass_filter()

        # 以下为 SC-FDE 原有逻辑
        span_symbols = int(self.filter_length)
        sps = self.sps
        T = (1.0 / self.sample_rate) * sps   # 符号周期（秒）
        if self.filter_type == 'rrc':
            h = self._rrc_impulse_response(span_symbols, sps, self.rolloff, T)
        elif self.filter_type == 'rc':
            h = self._rc_impulse_response(span_symbols, sps, self.rolloff, T)
        elif self.filter_type == 'rect':
            num_taps = span_symbols * sps + 1
            m = np.arange(-num_taps // 2, num_taps // 2 + 1)
            t = m * T / sps
            h = np.where(np.abs(t) <= T / 2, 1.0, 0.0)
        else:
            raise ValueError("Unsupported filter type")
        # 能量归一化（SC-FDE 成型滤波器标准做法）
        h = h / np.sqrt(np.sum(h ** 2) + 1e-15)
        return h    


    def upsample_symbols(self, symbols):
        """上采样：符号间插0"""
        up = np.zeros(len(symbols)*self.sps, dtype=complex)
        up[::self.sps] = symbols
        return up

    def shape_pulse(self, data_dict):
        """
        脉冲成型 / 上采样主函数
        SC-FDE 模式：完成成型滤波；OFDM 模式：仅插值抗镜像
        """
        self._verification_data(data_dict)
        self.filter_coeffs = self._design_tx_filter()
        symbols = data_dict["signal_stream"]
        up = self.upsample_symbols(symbols)

        # 卷积 + 延迟对齐
        sig = np.convolve(up, self.filter_coeffs, mode="full")
        delay = len(self.filter_coeffs) // 2
        shaped = sig[delay : delay + len(up)]   # 对齐后截取有效长度

        if len(shaped) != self.symbol_length:
            raise ValueError(
                f"成型后信号长度不匹配:预期长度={self.symbol_length}, 实际长度={len(shaped)}"
            )

        result_dict = {
            "signal_stream": shaped,
            "sample_rate_Hz": self.sample_rate,
            "base_sample_rate_Hz": data_dict["sample_rate_Hz"],
            "oversampling_factor": self.sps,
            "duration_seconds": self.duration,
            "signal_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
            "up":up,
        }
        return result_dict    

    def get_freq_response(self):
        """幅频响应（修正频率轴计算）"""
        n_fft = 4096
        H = np.fft.fft(self.filter_coeffs, n_fft)
        f = np.fft.fftfreq(n_fft, 1/self.sample_rate)
        pos = f >= 0  # 只取正频率
        return f[pos], 20*np.log10(np.abs(H[pos]) + 1e-12)

    # 眼图绘制接口
    def plot_eye_diagram(self, result_dict, num_symbols=500):
        """生成成型信号并绘制眼图"""
        # 生成测试信号
        shaped_signal = result_dict["signal_stream"][:num_symbols * self.sps]
        # 绘制眼图
        fig, ax = plt.subplots(figsize=(8, 6))
        # 按符号周期重排数据
        samples_per_symbol = self.sps
        num_rows = len(shaped_signal) // samples_per_symbol
        eye_data = shaped_signal[:num_rows*samples_per_symbol].reshape(-1, samples_per_symbol)
        # 叠加绘制
        for i in range(num_rows):
            ax.plot(np.real(eye_data[i]), alpha=0.1, color='blue')
        ax.set_title(f"{self.filter_type.upper()}滤波器眼图（滚降系数={self.rolloff}）")
        ax.set_xlabel("符号周期内采样点")
        ax.set_ylabel("幅值（实部）")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
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
    # gi_inserter = GIInserter(params)
    # data_with_gi_dict = gi_inserter.insert_gi(symbols_dict)
    # print(f"符号长度：{data_with_gi_dict['signal_length']}")
    # print(f"采样率：{data_with_gi_dict['sample_rate_Hz']} Hz，时长：{data_with_gi_dict['duration_seconds']} 秒")
    # print(f"补零数量：{data_with_gi_dict['padding_bit_num']} bit")
    ofdm_processer = TxOFDMProcesser(params)
    data_ofdm_dict = ofdm_processer.ofdm_process(symbols_dict)
    signal_power = np.mean(np.abs(data_ofdm_dict["signal_stream"]) ** 2)
    print(f"ofdm信号功率：{signal_power}")
    plt.plot(symbols_dict['signal_stream'][:200])
    plt.title("OFDM符号（前200点）")
    plt.xlabel("符号索引")
    plt.ylabel("幅度")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    preamble_gen = PreambleGenerator(params)
    sync, sfd, ces = preamble_gen.generate()
    preamble = np.concatenate([sync, sfd, ces])
    signal_power = np.mean(np.abs(preamble) ** 2)
    print(f"前导信号功率：{signal_power}")
    
    data_with_preamble_dict = {
    "signal_stream": np.concatenate([preamble, data_ofdm_dict['signal_stream']]),
    "sample_rate_Hz": data_ofdm_dict['sample_rate_Hz'],
    "duration_seconds": (len(data_ofdm_dict['signal_stream']) + len(preamble)) / data_ofdm_dict['sample_rate_Hz'],
    "signal_length": len(data_ofdm_dict['signal_stream']) + len(preamble),
    "padding_bit_num": data_ofdm_dict['padding_bit_num'],
    }
    signal_power = np.mean(np.abs(data_with_preamble_dict["signal_stream"]) ** 2)
    print(f"信号功率：{signal_power}")

    pulse_shaper = TxPulseShaper(params)
    shaped_signal_dict = pulse_shaper.shape_pulse(data_with_preamble_dict)
    signal_power = np.mean(np.abs(shaped_signal_dict["signal_stream"]) ** 2)
    print(f"成型后信号功率：{signal_power}")
    print(f"成型后信号前100点：{shaped_signal_dict['signal_stream'][:100]}")
    print(f"成型后信号长度：{shaped_signal_dict['signal_length']}")
    print(f"采样率：{shaped_signal_dict['sample_rate_Hz']} Hz，时长：{shaped_signal_dict['duration_seconds']} 秒")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].plot(shaped_signal_dict['signal_stream'][:500])
    axes[0].set_title("成形滤波符号（前100点）")
    axes[0].set_xlabel("符号索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(shaped_signal_dict['up'][:500], color='orange')
    axes[1].set_title("上采样符号（前100点）")
    axes[1].set_xlabel("符号索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
