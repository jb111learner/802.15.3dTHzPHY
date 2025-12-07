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
        if self.sample_rate % self.symbol_rate != 0:
            raise ValueError("采样率必须为符号率整数倍")
        if not (0 <= self.rolloff <= 1):
            raise ValueError("滚降系数必须在0~1之间")
        if self.filter_type not in ["rc", "rrc", "rect"]:
            raise ValueError("仅支持 rc/rrc/rect")
        if self.filter_length % 2 == 0:
            raise ValueError("滤波器长度必须为奇数个符号")

    # ----------------------- RC / RRC 数学公式 -----------------------
    def _raised_cosine(self, t, beta, T):
        h = np.zeros_like(t)

        for i, ti in enumerate(t):
            if abs(ti) < 1e-12:
                # t = 0
                h[i] = 1.0
            elif abs(1 - (2 * beta * ti / T)**2) < 1e-12:
                # t = ± T/(2β)
                h[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
            else:
                num = np.sin(np.pi * ti / T * (1 - beta)) + \
                    4 * beta * ti / T * np.cos(np.pi * ti / T * (1 + beta))
                den = np.pi * ti / T * (1 - (4 * beta * ti / T)**2)
                h[i] = num / den

        return h


    def _root_raised_cosine(self, t, beta, T):

        h = np.zeros_like(t)

        for i, ti in enumerate(t):

            if abs(ti) < 1e-12:   # t = 0
                h[i] = (1 + beta * (4/np.pi - 1)) / np.sqrt(T)

            elif abs(abs(ti) - T/(4*beta)) < 1e-12:  # t = ±T/(4β)
                h[i] = (beta / (np.sqrt(2*T))) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
                )

            else:
                num = ( np.sin(np.pi*ti*(1-beta)/T) / (np.pi*ti/T)
                    + 4*beta*(ti/T)*np.cos(np.pi*ti*(1+beta)/T) /
                    (1 - (4*beta*ti/T)**2) )

                h[i] = num / np.sqrt(T)

        return h


    def _design_tx_filter(self):
        """根据 filter_type 生成滤波器系数"""
        taps = self.filter_length * self.sps
        T = 1.0 / self.symbol_rate  # symbol period

        # 时间向量 (对称抽头)
        t = np.arange(-taps//2, taps//2 + 1) / self.sample_rate

        if self.filter_type == "rect":
            h = np.ones_like(t)
            h = h / np.sqrt(np.sum(h*h))   # 能量归一化
            return h

        elif self.filter_type == "rc":
            h = self._raised_cosine(t, self.rolloff, T)

        elif self.filter_type == "rrc":
            h = self._root_raised_cosine(t, self.rolloff, T)

        # 归一化
        h = h / np.sqrt(np.sum(h*h))
        return h

    def upsample_symbols(self, symbols):
        up = np.zeros(len(symbols)*self.sps, dtype=complex)
        up[::self.sps] = symbols
        return up

    def shape_pulse(self, symbols):
        up = self.upsample_symbols(symbols)
        sig = np.convolve(up, self.filter_coeffs, mode="full")
        delay = len(self.filter_coeffs)//2
        return sig[delay:delay+len(up)]

    def get_freq_response(self):
        n_fft = 4096
        H = np.fft.fft(self.filter_coeffs, n_fft)
        fs_rrc = self.symbol_rate * self.sps  
        f = np.fft.fftfreq(n_fft, 1/fs_rrc)

        # f = np.fft.fftfreq(n_fft, 1/self.sample_rate)
        pos = f >= 0
        return f[pos], 20*np.log10(np.abs(H[pos]) + 1e-12)



# ===================== 测试主程序（全修正版） =====================
if __name__ == "__main__":
    print("=== 发送端脉冲成型类测试程序（修正版） ===\n")

    # ========== 测试1：参数合法性校验 ==========
    print("=== 测试1：参数合法性校验 ===")
    # 测试非法滚降系数
    try:
        invalid_rolloff_params = {
            "symbol_rate": 1e6,
            "sample_rate": 4e6,
            "rolloff": 1.5,  # 非法值（>1）
            "filter_type": "rrc",
            "filter_length": 63
        }
        TxPulseShaper(invalid_rolloff_params)
    except ValueError as e:
        print(f"✓ 非法滚降系数校验通过：{e}")

    # 测试非整数倍采样率
    try:
        invalid_sample_rate_params = {
            "symbol_rate": 1e6,
            "sample_rate": 3e6,  # 非4倍（非法）
            "rolloff": 0.2,
            "filter_type": "rrc",
            "filter_length": 63
        }
        TxPulseShaper(invalid_sample_rate_params)
    except ValueError as e:
        print(f"✓ 非整数倍采样率校验通过：{e}")

    # 测试偶数滤波器长度
    try:
        even_length_params = {
            "symbol_rate": 1e6,
            "sample_rate": 4e6,
            "rolloff": 0.2,
            "filter_type": "rrc",
            "filter_length": 64  # 偶数（非法）
        }
        TxPulseShaper(even_length_params)
    except ValueError as e:
        print(f"✓ 偶数滤波器长度校验通过：{e}")
    print()

    # ========== 测试2：不同滤波器类型的设计 ==========
    print("=== 测试2：滤波器设计验证 ===")
    # 配置参数（RRC滤波器）
    rrc_params = {
        "symbol_rate": 1e6,
        "sample_rate": 4e6,
        "rolloff": 0.35,
        "filter_type": "rrc",
        "filter_length": 63
    }
    tx_rrc = TxPulseShaper(rrc_params)
    
    # 配置参数（RC滤波器）
    rc_params = {
        "symbol_rate": 1e6,
        "sample_rate": 4e6,
        "rolloff": 0.35,
        "filter_type": "rc",
        "filter_length": 63
    }
    tx_rc = TxPulseShaper(rc_params)
    
    # 配置参数（矩形滤波器）
    rect_params = {
        "symbol_rate": 1e6,
        "sample_rate": 4e6,
        "filter_type": "rect",
        "filter_length": 63
    }
    tx_rect = TxPulseShaper(rect_params)
    
    print(f"RRC滤波器抽头数：{len(tx_rrc.filter_coeffs)}（预期：63×4=252）")
    print(f"RC滤波器抽头数：{len(tx_rc.filter_coeffs)}（预期：252）")
    print(f"矩形滤波器抽头数：{len(tx_rect.filter_coeffs)}（预期：252）")
    print(f"每符号采样数：{tx_rrc.sps}（预期：4）")
    print()

    # ========== 测试3：脉冲成型效果验证 ==========
    print("=== 测试3：脉冲成型效果验证 ===")
    # 生成测试符号（QPSK）
    num_symbols = 50
    symbols = (np.random.randint(0, 2, num_symbols) * 2 - 1) + \
              1j * (np.random.randint(0, 2, num_symbols) * 2 - 1)
    
    # 上采样测试
    upsampled = tx_rrc.upsample_symbols(symbols)
    print(f"原始符号长度：{len(symbols)} → 上采样后长度：{len(upsampled)}（预期：50×4=200）")
    
    # 脉冲成型
    shaped_signal = tx_rrc.shape_pulse(symbols)
    print(f"成型后信号长度：{len(shaped_signal)}（预期：200）")
    print()

    # ========== 测试4：滤波器时域响应可视化 ==========
    print("=== 测试4：滤波器时域响应可视化 ===")
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RRC滤波器脉冲响应
    t_rrc = np.linspace(-tx_rrc.filter_length/2, tx_rrc.filter_length/2, len(tx_rrc.filter_coeffs))
    axes[0,0].plot(t_rrc, tx_rrc.filter_coeffs)
    axes[0,0].set_title("RRC滤波器脉冲响应（滚降系数=0.35）")
    axes[0,0].set_xlabel("符号周期")
    axes[0,0].set_ylabel("幅度")
    axes[0,0].grid(True)
    
    # RC滤波器脉冲响应
    t_rc = np.linspace(-tx_rc.filter_length/2, tx_rc.filter_length/2, len(tx_rc.filter_coeffs))
    axes[0,1].plot(t_rc, tx_rc.filter_coeffs)
    axes[0,1].set_title("RC滤波器脉冲响应（滚降系数=0.35）")
    axes[0,1].set_xlabel("符号周期")
    axes[0,1].grid(True)
    
    # 矩形滤波器脉冲响应
    t_rect = np.linspace(-tx_rect.filter_length/2, tx_rect.filter_length/2, len(tx_rect.filter_coeffs))
    axes[1,0].plot(t_rect, tx_rect.filter_coeffs)
    axes[1,0].set_title("矩形滤波器脉冲响应")
    axes[1,0].set_xlabel("符号周期")
    axes[1,0].grid(True)
    
    # 原始符号与成型后信号对比
    axes[1,1].stem(np.real(symbols[:10]), label="原始符号")
    # 采样点索引对齐
    axes[1,1].plot(np.arange(40), np.real(shaped_signal[:40]), 'r-', label="成型后信号")
    axes[1,1].set_title("原始符号与RRC成型后信号对比（实部）")
    axes[1,1].set_xlabel("符号索引/采样点")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig("tx_pulse_shaper_time_domain.png", dpi=300)
    print("时域响应图已保存为 tx_pulse_shaper_time_domain.png")

    # ========== 测试5：单滤波器幅频响应分析 ==========
    print("\n=== 测试5：RRC滤波器幅频响应分析 ===")
    # 获取幅频响应
    freq_pos, mag_pos = tx_rrc.get_freq_response()
    # 计算理论截止频率
    nyquist_freq = tx_rrc.symbol_rate / 2
    cutoff_freq = nyquist_freq * (1 + tx_rrc.rolloff)
    
    # 绘制幅频响应
    plt.figure(figsize=(10, 4))
    plt.plot(freq_pos/1e6, mag_pos)
    plt.title("RRC滤波器幅频响应（滚降系数=0.35）")
    plt.xlabel("频率 (MHz)")
    plt.ylabel("幅度 (dB)")
    plt.xlim(0, 1.5)
    plt.ylim(-80, 10)
    # 标注关键频率
    plt.axvline(nyquist_freq/1e6, color='r', linestyle='--', label=f'奈奎斯特频率 ({nyquist_freq/1e6:.2f}MHz)')
    plt.axvline(cutoff_freq/1e6, color='g', linestyle='--', label=f'截止频率 ({cutoff_freq/1e6:.2f}MHz)')
    plt.legend()
    plt.grid(True)
    plt.savefig("single_rrc_freq_response.png", dpi=300)
    print("RRC幅频响应图已保存为 single_rrc_freq_response.png")

    # ========== 测试6：不同滚降系数RRC滤波器对比 ==========
    print("\n=== 测试6：不同滚降系数RRC滤波器对比 ===")
    rolloff_values = [0.1, 0.5, 1.0]
    plt.figure(figsize=(10, 6))
    
    for alpha in rolloff_values:
        test_params = {
            "symbol_rate": 1e6,
            "sample_rate": 4e6,
            "rolloff": alpha,
            "filter_type": "rrc",
            "filter_length": 63
        }
        tx_test = TxPulseShaper(test_params)
        freq_pos, mag_pos = tx_test.get_freq_response()
        # 绘制正频率部分
        plt.plot(freq_pos/1e6, mag_pos, label=f'滚降系数={alpha}')
        # 标注对应截止频率
        cutoff = (1 + alpha) * tx_test.symbol_rate / 2
        plt.axvline(cutoff/1e6, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("不同滚降系数的RRC滤波器幅频响应")
    plt.xlabel("频率 (MHz)")
    plt.ylabel("幅度 (dB)")
    plt.xlim(0, 1.5)
    plt.ylim(-80, 10)
    plt.legend()
    plt.grid(True)
    plt.savefig("rrc_rolloff_comparison.png", dpi=300)
    print("RRC滚降系数对比图已保存为 rrc_rolloff_comparison.png")

    # ========== 测试7：三种滤波器类型对比 ==========
    print("\n=== 测试7：RC/RRC/矩形滤波器幅频响应对比 ===")
    # 统一参数
    base_params = {
        "symbol_rate": 1e6,
        "sample_rate": 4e6,
        "rolloff": 0.35,
        "filter_length": 63
    }
    
    plt.figure(figsize=(10, 6))
    filter_types = ['rrc', 'rc', 'rect']
    labels = ['根升余弦(RRC)', '升余弦(RC)', '矩形(rect)']
    colors = ['blue', 'orange', 'green']
    
    for ftype, label, color in zip(filter_types, labels, colors):
        params = base_params.copy()
        params['filter_type'] = ftype
        tx = TxPulseShaper(params)
        freq_pos, mag_pos = tx.get_freq_response()
        plt.plot(freq_pos/1e6, mag_pos, color=color, label=label)
    
    plt.title("RC/RRC/矩形滤波器幅频响应对比（滚降系数=0.35）")
    plt.xlabel("频率 (MHz)")
    plt.ylabel("幅度 (dB)")
    plt.xlim(0, 1.5)
    plt.ylim(-80, 10)
    plt.legend()
    plt.grid(True)
    plt.savefig("filter_type_comparison.png", dpi=300)
    print("滤波器类型对比图已保存为 filter_type_comparison.png")

    print("\n=== 所有测试完成！ ===")