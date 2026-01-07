import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
# 设置中文字体和编码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class TxPulseShaper:
    """
    发送端脉冲成型类：RC/RRC/矩形滤波器
    使用严格数学公式实现 RC/RRC
    """

    def __init__(self, params):
        self.symbol_rate = params.get("symbol_rate")
        self.oversampling = params.get("oversampling")
        self.sample_rate = self.symbol_rate * self.oversampling
        self.rolloff = params.get("rolloff")
        self.filter_type = params.get("filter_type").lower()
        self.filter_length = params.get("filter_length")
        self.chan_max_delay = max(params.get("chan_delays"))

        self.sps = int(self.sample_rate / self.symbol_rate)
        self._validate_params()

        # 用数学公式生成滤波器
        self.filter_coeffs = self._design_tx_filter()

    def _validate_params(self):
        """参数合法性校验（修复命名+逻辑）"""
        if self.oversampling < 1 or not isinstance(self.oversampling, int):
            raise ValueError(f"上采样率必须为正整数，当前值：{self.oversampling}")
        if not (0 <= self.rolloff <= 1):
            raise ValueError(f"滚降系数必须在0~1之间，当前值：{self.rolloff}")
        if self.filter_type not in ["rc", "rrc", "rect"]:
            raise ValueError(f"仅支持 rc/rrc/rect，当前值：{self.filter_type}")

    def _rrc_impulse_response(self, span_symbols, sps, beta, T):
        """
        返回长度 = span_symbols*sps + 1 的 RRC (root-raised-cosine) 冲激响应（矢量化）
        参考标准公式：
        h(t) = (4*beta / (pi*sqrt(T))) * (cos((1+beta)*pi*t/T) + (sin((1-beta)*pi*t/T) / (4*beta*t/T))) / (1 - (4*beta*t/T)**2)
        处理 t=0 和 t=±T/(4β) 的极限值
        """
        num = int(span_symbols * sps)
        # time vector centered at 0, step = 1/sample_rate
        # t in units of seconds
        # use samples index m = -(num//2) ... +(num//2)
        m = np.arange(-num//2, num//2 + 1)
        t = m / (sps * (1.0 / T))  # since sample_rate = sps/T -> t = m / sample_rate = m/(sps/T) = m * T / sps
        # equivalently: t = m * T / sps
        t = m * T / sps

        h = np.zeros_like(t, dtype=np.float64)
        pi = np.pi
        # Avoid division by zero: handle special points analytically
        # General formula for RRC (with sqrt(T) normalization) - use stable implementation:
        for idx, ti in enumerate(t):
            if abs(ti) < 1e-12:
                # limit t->0
                h[idx] = (1.0 / np.sqrt(T)) * (1.0 + beta * (4/np.pi - 1.0))
            elif beta > 0 and abs(abs(ti) - T/(4*beta)) < 1e-12:
                # t = ± T/(4β) special value
                h[idx] = (beta / (np.sqrt(2*T))) * (
                    (1 + 2/np.pi) * np.sin(pi / (4*beta)) +
                    (1 - 2/np.pi) * np.cos(pi / (4*beta))
                )
            else:
                numerator = np.sin(pi * ti * (1 - beta) / T) + 4 * beta * ti / T * np.cos(pi * ti * (1 + beta) / T)
                denominator = pi * ti / T * (1 - (4 * beta * ti / T)**2) * np.sqrt(T)
                h[idx] = numerator / denominator

        # L2 normalization to unit energy
        h = h / np.sqrt(np.sum(h**2) + 1e-15)
        return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


    def _rc_impulse_response(self, span_symbols, sps, beta, T):
        """直接由 RRC 卷积得到 RC，或者实现 RC 的标准公式（此处建议用 RRC 与自身卷积得到 RC）"""
        # RC = convolution(RRC, RRC) in continuous sense; discretely you can convolve sampled RRC with itself.
        rrc = self._rrc_impulse_response(span_symbols, sps, beta, T)
        rc = np.convolve(rrc, rrc, mode='full')
        # crop to desired length (take center part)
        center = len(rc) // 2
        desired_len = len(rrc)  # keep same length
        half = desired_len // 2
        rc_cropped = rc[center - half : center - half + desired_len]
        rc_cropped = rc_cropped / np.sqrt(np.sum(rc_cropped**2) + 1e-15)
        return rc_cropped


    def _design_tx_filter(self):
        """重新设计：明确 filter_length 表示符号长度 span (in symbols)"""
        # 解释：self.filter_length 被解释为跨越的符号数（span），必须为正整数
        span_symbols = int(self.filter_length)
        sps = self.sps
        T = 1.0 / self.symbol_rate
        if self.filter_type == 'rrc':
            h = self._rrc_impulse_response(span_symbols, sps, self.rolloff, T)
        elif self.filter_type == 'rc':
            h = self._rc_impulse_response(span_symbols, sps, self.rolloff, T)
        elif self.filter_type == 'rect':
            # rectangle spanning 1 symbol (or span_symbols symbols)
            num_taps = span_symbols * sps + 1
            m = np.arange(-num_taps//2, num_taps//2 + 1)
            t = m * T / sps
            h = np.where(np.abs(t) <= T/2, 1.0, 0.0)
            h = h / np.sqrt(np.sum(h**2) + 1e-15)
        else:
            raise ValueError("Unsupported filter type")
        return h


    def upsample_symbols(self, symbols):
        """上采样：符号间插0"""
        up = np.zeros(len(symbols)*self.sps, dtype=complex)
        up[::self.sps] = symbols
        return up

    def shape_pulse(self, symbols):
        """脉冲成型：上采样 + 卷积 + 对齐延迟"""
        up = self.upsample_symbols(symbols)
        # 卷积（full模式）+ 延迟对齐（去除滤波器引入的延迟）
        sig = np.convolve(up, self.filter_coeffs, mode="full")
        delay = len(self.filter_coeffs) // 2
        # 截断到原上采样长度，保证输出长度 = 输入符号数 × 上采样率
        return sig[delay:delay+len(up)]

    def get_freq_response(self):
        """幅频响应（修正频率轴计算）"""
        n_fft = 4096
        H = np.fft.fft(self.filter_coeffs, n_fft)
        f = np.fft.fftfreq(n_fft, 1/self.sample_rate)
        pos = f >= 0  # 只取正频率
        return f[pos], 20*np.log10(np.abs(H[pos]) + 1e-12)

    # 新增：眼图绘制接口（适配发射机代码）
    def plot_eye_diagram(self, symbols, num_symbols=500):
        """生成成型信号并绘制眼图"""
        # 生成测试信号
        shaped_signal = self.shape_pulse(symbols[:num_symbols])
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
        return shaped_signal
    
# ===================== 简化版主程序（核心：参数调用+绘图） =====================
if __name__ == "__main__":
    # ========== 1. 定义核心参数（可根据需求修改） ==========
    pulse_params = {
        "symbol_rate": 1e6,          # 符号速率 1MHz
        "oversampling": 8,           # 8倍上采样（眼图更平滑）
        "rolloff": 0.35,             # 滚降系数（0.35~0.5最优）
        "filter_type": "rrc",        # 滤波器类型：rrc/rc/rect
        "filter_length": 63,         # 滤波器长度（奇数个符号）
        "chan_delays": [0]           # 信道延迟（默认0）
    }

    # ========== 2. 初始化脉冲成型器 ==========
    print("=== 初始化脉冲成型滤波器 ===")
    print(f"滤波器类型：{pulse_params['filter_type'].upper()}")
    print(f"滚降系数：{pulse_params['rolloff']}")
    print(f"上采样率：{pulse_params['oversampling']}")
    tx_shaper = TxPulseShaper(pulse_params)

    # ========== 3. 绘制滤波器幅频响应曲线 ==========
    print("\n=== 绘制滤波器幅频响应曲线 ===")
    freq, mag = tx_shaper.get_freq_response()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq/1e6, mag, color='blue', linewidth=1.5)
    # 标注关键频率
    nyquist_freq = tx_shaper.symbol_rate / 2
    cutoff_freq = nyquist_freq * (1 + tx_shaper.rolloff)
    ax.axvline(nyquist_freq/1e6, color='r', linestyle='--', alpha=0.7, 
               label=f'奈奎斯特频率: {nyquist_freq/1e6:.2f}MHz')
    ax.axvline(cutoff_freq/1e6, color='g', linestyle='--', alpha=0.7, 
               label=f'截止频率: {cutoff_freq/1e6:.2f}MHz')
    # 图表美化
    ax.set_title(f"{tx_shaper.filter_type.upper()}滤波器幅频响应（滚降系数={tx_shaper.rolloff}）")
    ax.set_xlabel("频率 (MHz)")
    ax.set_ylabel("幅度 (dB)")
    ax.set_xlim(0, 1.5)  # 聚焦0~1.5MHz频段
    ax.set_ylim(-80, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========== 4. 生成测试符号并绘制眼图 ==========
    print("\n=== 绘制滤波器眼图 ===")
    # 生成QPSK测试符号（模拟真实调制符号）
    num_test_symbols = 2000  # 用于绘制眼图的符号数
    test_symbols = (np.random.randint(0, 2, num_test_symbols)*2 - 1) + \
                   1j*(np.random.randint(0, 2, num_test_symbols)*2 - 1)
    # 绘制眼图并返回成型后的信号
    shaped_signal = tx_shaper.plot_eye_diagram(test_symbols, num_symbols=1000)

    # ========== 5. 输出核心信息（可选） ==========
    print(f"\n=== 核心参数汇总 ===")
    print(f"滤波器抽头数：{len(tx_shaper.filter_coeffs)}")
    print(f"每符号采样数：{tx_shaper.sps}")
    print(f"成型后信号长度：{len(shaped_signal)} 采样点")
    print("=== 绘图完成 ===")