import numpy as np
import matplotlib.pyplot as plt
# 保留原有业务导入
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.GIremover import GIRemover
from receiver.DeModulator import THzDemodulator
from utils.GSDecoder import GSDecoder
import galois
from math import comb



class BGMDDecoder:
    def __init__(self, q, k, M_max=2):
        """
        :param q: 有限域GF(q)的阶，建议用2的幂（如2^8=256）
        :param k: RS码的信息符号数，对应加权次数的权重(k-1)
        :param M_max: 最大重数分配值
        """
        self.GF = galois.GF(q)  # 初始化有限域
        self.k = k  # RS码参数K
        self.alpha = self.GF.primitive_element  # 有限域本原元
        self.M_max = M_max
        self.weight = (1, k-1)  # (wx, wy)加权系数
        self.dtype = self.GF(1).dtype  # 有限域原生dtype，避免转换错误
        
        # 插值相关动态变量
        self.polys = []  # 存储Q^(l)(x,y)多项式
        self.w_degs = []  # 存储每个多项式的加权次数
        self.max_dim = 0  # 多项式统一最大维度
        self.multiplicity = [] # 符号对应重数列表
        self.hard_decision = [] # 硬判决结果
        self.erased_bit_indices = [] # 擦除比特全局索引
        
        # 全局缓存：避免重复计算（跨插值点复用）
        self.comb_cache = {}  # 组合数缓存：(n,k) → 模q后的值
        self.power_cache = {}  # 幂运算缓存：(base_int, exp) → 有限域元素

    # ======================== BGMD重数分配相关方法（向量化优化） ========================    
    def assign_multiplicity(self, llr_bits, bits_per_symbol, erase_num=1, erase_ratio=None):
        """
        完整流程（擦除LLR最低比特）：
        信道观测值 → LLR → 硬判决 → 擦除最不可靠比特 → 按符号重数分配
        【优化点】：全向量化处理，消除Python循环，提升大数组长处理效率
        """
        llr_bits = np.asarray(llr_bits, dtype=np.float64)
        total_bits = len(llr_bits)
        
        # 步骤1：比特级硬判决（向量化，替代for循环）
        hard_decision_bits = np.where(llr_bits >= 0, 0, 1).astype(np.uint8)

        # 步骤2：擦除LLR绝对值最低的比特（无阈值，按数量/比例）
        reliability = np.abs(llr_bits)
        # 确定要擦除的比特数
        if erase_num is not None:
            erase_count = min(int(erase_num), total_bits)
        elif erase_ratio is not None:
            erase_count = max(1, int(total_bits * erase_ratio))
        else:
            erase_count = 1
        # 按可靠性升序排序，取前erase_count个比特的索引（向量化排序）
        sorted_indices = np.argsort(reliability)[:erase_count]
        self.erased_bit_indices.append(sorted_indices)
        
        # 生成擦除标记数组（1=擦除，0=不擦除）
        erased_mask = np.zeros(total_bits, dtype=np.uint8)
        erased_mask[sorted_indices] = 1
        
        # 步骤3：按符号统计擦除位置（向量化，核心优化）
        num_symbols = total_bits // bits_per_symbol
        # 重塑为[符号数, 每符号比特数]，方便批量处理
        erased_mask_2d = erased_mask[:num_symbols*bits_per_symbol].reshape(num_symbols, bits_per_symbol)
        hard_bits_2d = hard_decision_bits[:num_symbols*bits_per_symbol].reshape(num_symbols, bits_per_symbol)
        
        # 向量化计算每个符号的硬判决值（二进制→十进制）
        bit_weights = 2 ** np.arange(bits_per_symbol-1, -1, -1, dtype=np.uint32)
        self.hard_decision = (hard_bits_2d @ bit_weights).astype(np.uint32)
        
        # 向量化计算每个符号内的擦除比特数
        erased_count_per_sym = np.sum(erased_mask_2d, axis=1)
        
        # 向量化重数分配（替代for循环+条件判断）
        self.multiplicity = np.where(
            erased_count_per_sym == 0, self.M_max,
            np.where(erased_count_per_sym == 1, self.M_max/2, 0)
        ).tolist()   




# ===================== 测试用例（保留原有逻辑，兼容优化后代码） =====================
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # === 进行匹配滤波并获得中间信号 ===
    rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)
    # 初始化GI移除器
    gi_remover = GIRemover(params)
    # === 进行GI移除 ===
    rx_data_no_gi_dict = gi_remover.remove_gi(rx_matched_dict)
    
    # 解调
    snr_db = params.get("SNRdB")    
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.00 / (2 * snr_linear))    
    demodulator = THzDemodulator(params)
    llr_dict = demodulator.demodulate(rx_data_no_gi_dict, sigma)
    NCBPS = params.get("NCBPS")
    rs_size_bits = (params.get("rs_nsym") + params.get("rs_packet_size")) * NCBPS
    llr_bits = llr_dict["signal_stream"][:rs_size_bits]

    # 初始化解码器（GF(256), RS(255,239), 最大重数2）
    BM = BGMDecoder(256, 239, 2)
    # 重数分配（向量化优化）
    BM.assign_multiplicity(llr_bits, NCBPS)
    # 插值（全流程优化）
    result_poly = BM.interpolate()
    # 提取多项式项并打印（调试，向量化优化）
    poly_terms = BM.extract_poly_terms(result_poly)
    print("\n=== 插值多项式非零项 ===")
    for r, s, coeff in poly_terms[:20]:  # 限制打印数量，避免刷屏
        print(f"x^{r} y^{s}: {coeff}")
    if len(poly_terms) > 20:
        print(f"... 共{len(poly_terms)}个非零项，省略后续打印")

