import numpy as np
import galois
import matplotlib.pyplot as plt
from itertools import product

GF = galois.GF(2**8)
n, k = 255, 192                      # 合法参数：15 | 255
rs = galois.ReedSolomon(n, k, field=GF)

Eb0_dB = np.arange(-6.5, -5, 0.2)     # -6~0 dB
num_messages = 1000                 # 每条消息 9 个符号（72 比特）
d_min = n - k + 1                   # = 7
# p = (d_min - 1) // 2              # = 3，最不可靠符号数量（实际为比特级）
p = 3                               # = 3，最不可靠符号数量（实际为比特级）


def symbols_to_bits(sym):
    return np.array([int(b) for s in sym for b in format(int(s), '08b')])

def bits_to_symbols(bits, field):
    assert len(bits) % 8 == 0
    sym = []
    for i in range(0, len(bits), 8):
        val = int(''.join(str(b) for b in bits[i:i+8]), 2)
        sym.append(val)
    return field(sym)

def chase_decoding(rx_signal, rx_bits, sigma):
    """
    比特级 Chase-II 译码（p=3，8个试探序列）
    """
    llr = 2 * rx_signal / (sigma**2)
    # 选择最不可靠的 p 个比特位置（LLR绝对值最小）
    unreliable_bits = np.argsort(np.abs(llr))[:p]
    candidate_messages = []
    candidate_distances = []

    for flip_pattern in product([0,1], repeat=p):
        test_bits = rx_bits.copy()
        for pos, flip in zip(unreliable_bits, flip_pattern):
            if flip:
                test_bits[pos] ^= 1
        test_symbols = bits_to_symbols(test_bits, GF)
        try:
            decoded_msg = rs.decode(test_symbols)
            decoded_codeword = rs.encode(decoded_msg)
            decoded_bits = symbols_to_bits(decoded_codeword)
            # 欧氏距离（与原始接收信号）
            dist = np.sum((rx_signal - (1 - 2*decoded_bits))**2)
            candidate_messages.append(decoded_msg)
            candidate_distances.append(dist)
        except galois.ReedSolomonError:
            continue

    if candidate_messages:
        best_idx = np.argmin(candidate_distances)
        return candidate_messages[best_idx]
    else:
        # 所有试探失败，返回硬判决消息（截断至前k个符号）
        return bits_to_symbols(rx_bits, GF)[:k]
    
def gmd_decoding(rx_signal, rx_bits, sigma):
    """
    GMD 译码（Generalized Minimum Distance，基于符号级可靠性排序）
    """
    llr = 2 * rx_signal / (sigma**2)
    # 获取符号级可靠性（每8个比特对应一个符号）
    symbol_llr = np.array([np.sum(np.abs(llr[i*8:(i+1)*8])) for i in range(0, len(llr), 8)])
    u = 0  # 初始化删除符号参数
    candidate_messages = []
    candidate_distances = []
    while True:
        # 选择最不可靠的 u 个符号位置（LLR绝对值最小）
        unreliable_symbols = np.argsort(np.abs(symbol_llr))[:u]
        # 使用one-hot编码将这些位置的符号标记为不可靠
        test_symbols = bits_to_symbols(rx_bits, GF)
        earase_mask = np.zeros(len(test_symbols), dtype=bool)
        earase_mask[unreliable_symbols] = True

        try:
            decoded_msg = rs.decode(test_symbols, erasures=earase_mask)
            decoded_codeword = rs.encode(decoded_msg)
            decoded_bits = symbols_to_bits(decoded_codeword)
            dist = np.sum((rx_signal - (1 - 2*decoded_bits))**2)
            candidate_messages.append(decoded_msg)
            candidate_distances.append(dist)
        except galois.ReedSolomonError:
            pass
        u += 2  # GMD通常以步长2增加删除符号数量
        if u > d_min - 1:  # 超过纠错能力则停止
            break
    if candidate_messages:
        best_idx = np.argmin(candidate_distances)
        return candidate_messages[best_idx]
    else:
        return bits_to_symbols(rx_bits, GF)[:k]    

# 存储结果
ber_hard = []
ber_chase = []
# ber_gmd = []

for Eb_dB in Eb0_dB:
    Eb = 1/8
    Eb_linear = 10**(Eb_dB/10)
    N0 = Eb / Eb_linear
    sigma = np.sqrt(N0/2)

    err_hard = 0
    err_chase = 0
    err_gmd = 0
    total_bits = 0

    for _ in range(num_messages):
        msg = GF.Random(k, low=1, high=255)
        codeword = rs.encode(msg)
        tx_bits = symbols_to_bits(codeword)
        tx_signal = 1 - 2 * tx_bits
        noise = np.random.normal(0, sigma, tx_signal.shape)
        rx_signal = tx_signal + noise
        rx_bits = (rx_signal < 0).astype(int)
        rx_symbols = bits_to_symbols(rx_bits, GF)

        # 硬判决译码
        try:
            dec_hard = rs.decode(rx_symbols)
        except galois.ReedSolomonError:
            dec_hard = GF.Zeros(k)

        # Chase译码
        dec_chase = chase_decoding(rx_signal, rx_bits, sigma)

        # # GMD译码
        # dec_gmd = gmd_decoding(rx_signal, rx_bits, sigma)

        orig_bits = symbols_to_bits(msg)
        hard_bits = symbols_to_bits(dec_hard)
        chase_bits = symbols_to_bits(dec_chase)
        # gmd_bits = symbols_to_bits(dec_gmd)

        err_hard += np.sum(orig_bits != hard_bits)
        err_chase += np.sum(orig_bits != chase_bits)
        # err_gmd += np.sum(orig_bits != gmd_bits)
        total_bits += len(orig_bits)

    ber_hard.append(err_hard / total_bits)
    ber_chase.append(err_chase / total_bits)
    # ber_gmd.append(err_gmd / total_bits)
    print(f"{Eb_dB} dB: Hard BER={ber_hard[-1]:.2e}, Chase BER={ber_chase[-1]:.2e}") #, GMD BER={ber_gmd[-1]:.2e}")

# 绘图
plt.figure(figsize=(8,5))
plt.semilogy(Eb0_dB, ber_hard, 'o-', label='Hard Decision')
plt.semilogy(Eb0_dB, ber_chase, 's-', label='Chase-II (bit-level, p=3)')
# plt.semilogy(Eb0_dB, ber_gmd, 'd-', label='GMD (symbol-level)')
plt.grid(True, which='both', linestyle='--')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('RS(255,192) Performance with Chase-II')
plt.legend()
plt.ylim(1e-6, 1)
plt.show()