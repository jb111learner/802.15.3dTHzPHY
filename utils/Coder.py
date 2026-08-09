from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from itertools import product
from utils.LDPCMatrix import (
    build_ieee802153d_1440_h,
    ieee802153d_1440_dimensions,
    ieee802153d_1440_encoder_matrices,
    ieee802153d_1440_matrix_metadata,
    ieee802153d_rate11_direction_note,
    ieee802153d_standard_confirmation_status,
    normalize_ieee802153d_rate11_direction,
    RATE11_PLUS_SYSTEMATIC_CANDIDATE,
)


class RSCoder:
    """
    高性能 RS 编译码器：比特流 ↔ GF域 直接转换
    已修复编码长度不匹配问题，严格保证输出比特长度正确
    """
    def __init__(self, params):
        # galois is required only by RS coding. Keep the import local so LDPC
        # users do not need to import or install the unrelated RS dependency.
        global galois
        import galois

        self.params = params
        self.nsym = params.get("rs_nsym")
        self.c_exp = params.get("rs_c_exp")
        self.packet_size = params.get("rs_packet_size")  # k, 单位：符号（字节）
        self.efficiency = self.packet_size / (self.packet_size + self.nsym)

        self.GF = galois.GF(2 ** self.c_exp)
        self.n = self.packet_size + self.nsym
        self.k = self.packet_size

        self.rs = galois.ReedSolomon(self.n, self.k, field=self.GF)

        self._pad_bits = 0
        self._original_bit_len = 0
        self._original_sym_len = 0
        self._num_packets = 0

    # =================================================================
    # 比特流 ↔ GF 符号（直接转换，超快）
    # =================================================================
    # def bits_to_gf(self, bits: np.ndarray) -> galois.FieldArray:
    #     bits = bits.astype(np.uint8)
    #     self._original_bit_len = len(bits)

    #     pad_bits = (self.c_exp - (len(bits) % self.c_exp)) % self.c_exp
    #     if pad_bits > 0:
    #         bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
    #     self._pad_bits = pad_bits
    #     bits_array = bits.reshape(-1, self.c_exp)

    #     # 转符号
    #     symbols = np.packbits(bits_array, bitorder='little')
    #     self._original_sym_len = len(symbols)
    #     return self.GF(symbols)
    def bits_to_gf(self, bits: np.ndarray) -> galois.FieldArray:
        bits = bits.astype(np.uint8)
        self._original_bit_len = len(bits)

        pad_bits = (self.c_exp - (len(bits) % self.c_exp)) % self.c_exp
        if pad_bits > 0:
            bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
        self._pad_bits = pad_bits
        bits_array = bits.reshape(-1, self.c_exp)

        # 将每行 c_exp 个比特转换为符号（LSB first）
        weights = 1 << np.arange(self.c_exp)          # [1, 2, 4, ...]
        symbols = bits_array.dot(weights)             # 结果范围 [0, 2^c_exp - 1]

        self._original_sym_len = len(symbols)
        return self.GF(symbols)    

    # def gf_to_bits(self, gf_arr: galois.FieldArray) -> np.ndarray:
    #     arr = gf_arr.view(np.ndarray).astype(np.uint8)
    #     bits = np.unpackbits(arr).ravel()
    #     return bits
    def gf_to_bits(self, gf_arr: galois.FieldArray) -> np.ndarray:
        # 获取整数值
        values = gf_arr.view(np.ndarray).astype(np.uint8)
        
        # 提取每个符号的低 c_exp 位，按 LSB first 排列
        # 构造形状 (N, c_exp) 的比特矩阵，每行是一个符号的二进制位（低位在前）
        bits_matrix = (values[:, None] >> np.arange(self.c_exp)) & 1
        bits = bits_matrix.ravel().astype(np.uint8)
        
        # # （可选）如果保存了原始比特长度，可裁剪掉填充的零
        # if hasattr(self, '_original_bit_len'):
        #     bits = bits[:self._original_bit_len]
        
        return bits    

    # =================================================================
    # 分包 / 组包（严格长度，保证编码后比特数正确）
    # =================================================================
    def split_to_packets(self, gf_arr: galois.FieldArray) -> List[galois.FieldArray]:
        packets = []
        total_sym = len(gf_arr)
        self._num_packets = (total_sym + self.k - 1) // self.k

        for i in range(self._num_packets):
            pkt = gf_arr[i*self.k : (i+1)*self.k]
            if len(pkt) < self.k:
                pkt = self.GF(np.concatenate([
                    pkt.view(np.ndarray),
                    np.zeros(self.k - len(pkt), dtype=np.uint8)
                ]))
            packets.append(pkt)
        return packets

    # =================================================================
    # 编码（已修复长度问题 ✅）
    # =================================================================
    def encode(self, bits: np.ndarray) -> np.ndarray:
        gf_msg = self.bits_to_gf(bits)
        packets = self.split_to_packets(gf_msg)

        # 逐包 RS 编码
        encoded_packets = [self.rs.encode(pkt) for pkt in packets]
        gf_encoded = self.GF(np.concatenate([p.view(np.ndarray) for p in encoded_packets]))

        # 转回比特 → 长度严格正确
        bits_encoded = self.gf_to_bits(gf_encoded)
        return bits_encoded

    # =================================================================
    # 解码
    # =================================================================
    def decode(
        self,
        llrs: np.ndarray,
        original_bit_len: int,
        erase_pos_bytes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, dict]:

        bits = (llrs < 0).astype(np.uint8)
        gf_rx = self.bits_to_gf(bits)

        if len(gf_rx) % self.n != 0:
            raise ValueError(f"编码符号长度必须是 {self.n} 的整数倍")

        # 拆包
        pkts = [gf_rx[i*self.n : (i+1)*self.n] for i in range(len(gf_rx) // self.n)]
        debug = {
            "packet_count": len(pkts),
            "success_packets": [],
            "partial_packets": [],
            "failed_packets": [],
            "original_bit_len": original_bit_len
        }

        decoded_pkts = []
        for idx, pkt in enumerate(pkts):
            erasures = None
            if erase_pos_bytes is not None:
                start = idx * self.n
                erasures = [p - start for p in erase_pos_bytes if start <= p < start + self.n]

            try:
                dec = self.rs.decode(pkt, erasures=erasures)
                decoded_pkts.append(dec)
                debug["success_packets"].append(idx)
            except galois.ReedSolomonError:
                decoded_pkts.append(pkt[:self.k])
                debug["partial_packets"].append(idx)
            except Exception:
                decoded_pkts.append(pkt[:self.k])
                debug["failed_packets"].append(idx)

        gf_dec = self.GF(np.concatenate([p.view(np.ndarray) for p in decoded_pkts]))
        decoded_bits = self.gf_to_bits(gf_dec)

        # 截断到原始有效比特长度
        decoded_bits = decoded_bits[:original_bit_len]
        return decoded_bits
    
    def decode_with_auto_erase(
        self,
        llrs: np.ndarray,
        original_bit_len: int,
        num_erasures_per_packet: int = 2
    ) -> Tuple[np.ndarray, dict]:
        """
        自动擦除译码：每个 RS 包选择最不可靠的 num_erasures_per_packet 个符号作为擦除。
        符号可靠性 = 对应比特 LLR 绝对值之和（越小越不可靠）。
        """
        bits_per_sym = self.c_exp
        # 1. 计算编码后的比特长度
        L = original_bit_len
        pad_to_8 = (bits_per_sym - (L % bits_per_sym)) % bits_per_sym
        L_padded = L + pad_to_8
        num_symbols = L_padded // bits_per_sym
        num_packets = (num_symbols + self.k - 1) // self.k
        total_coded_bits = num_packets * self.n * bits_per_sym

        if len(llrs) != total_coded_bits:
            if len(llrs) > total_coded_bits:
                llrs = llrs[:total_coded_bits]
            else:
                llrs = np.pad(llrs, (0, total_coded_bits - len(llrs)), constant_values=0)

        # 2. 计算每个符号的可靠性（LLR 绝对值之和）
        llrs_per_sym = llrs.reshape(-1, bits_per_sym)
        sym_reliability = np.sum(np.abs(llrs_per_sym), axis=1)  # 越小越不可靠

        # 3. 按包分组
        pkt_reliability = sym_reliability.reshape(num_packets, self.n)

        # 4. 为每个包生成擦除布尔掩码
        erasures_masks = []
        for pkt_rel in pkt_reliability:
            k = min(num_erasures_per_packet, self.n)
            idx = np.argpartition(pkt_rel, k-1)[:k]   # 最不可靠的 k 个索引
            mask = np.zeros(self.n, dtype=bool)
            mask[idx] = True
            erasures_masks.append(mask)

        # 5. 硬判决 + GF 符号
        rx_bits = (llrs < 0).astype(np.uint8)
        gf_rx = self.bits_to_gf(rx_bits)
        gf_packets = gf_rx.reshape(num_packets, self.n)

        # 6. 逐包译码（使用布尔掩码）
        decoded_pkts = []
        debug = {
            "packet_count": num_packets,
            "success_packets": [],
            "partial_packets": [],
            "failed_packets": [],
            "erasures_masks": erasures_masks
        }

        for idx, (pkt, mask) in enumerate(zip(gf_packets, erasures_masks)):
            try:
                dec = self.rs.decode(pkt, erasures=mask)   # 此处传入布尔掩码
                decoded_pkts.append(dec)
                debug["success_packets"].append(idx)
            except galois.ReedSolomonError:
                decoded_pkts.append(pkt[:self.k])
                debug["partial_packets"].append(idx)
            except Exception:
                decoded_pkts.append(pkt[:self.k])
                debug["failed_packets"].append(idx)

        # 7. 合并并转回比特
        gf_decoded = self.GF(np.concatenate([p.view(np.ndarray) for p in decoded_pkts]))
        decoded_bits = self.gf_to_bits(gf_decoded)
        decoded_bits = decoded_bits[:original_bit_len]
        return decoded_bits
    

    def decode_chase(
    self,
    llrs: np.ndarray,
    original_bit_len: int,
    num_chase_per_packet: int = 3
    ) -> Tuple[np.ndarray, dict]:
        """
        采用 Chase-II 软判决译码（p=3，8 个试探序列）的 RS 解码器。

        参数：
            llrs: 对数似然比，形状 (N_bits,)，正对应比特 0，负对应比特 1。
            original_bit_len: 原始信息比特长度（用于最终截断）。
            num_chase_per_packet: 每个包的 Chase 试探序列数量。

        返回：
            decoded_bits: 解码后的比特数组，长度 <= original_bit_len。
            debug: 调试信息字典。
        """
        # 每个 GF 符号对应的比特数
        bits_per_symbol = int(np.log2(self.GF.order))
        # 硬判决比特
        bits = (llrs < 0).astype(np.uint8)
        # 转换为 GF 符号
        gf_rx = self.bits_to_gf(bits)

        if len(gf_rx) % self.n != 0:
            raise ValueError(f"编码符号长度必须是 {self.n} 的整数倍")

        # 拆包
        pkts = [gf_rx[i*self.n : (i+1)*self.n] for i in range(len(gf_rx) // self.n)]
        debug = {
            "packet_count": len(pkts),
            "success_packets": [],
            "partial_packets": [],
            "failed_packets": [],
            "original_bit_len": original_bit_len
        }

        decoded_pkts = []  # 存放每个包解码出的消息符号（长度 k）

        # Chase 参数
        for idx, pkt in enumerate(pkts):
            # 计算当前包对应的比特区间
            bit_start = idx * self.n * bits_per_symbol
            bit_end = (idx + 1) * self.n * bits_per_symbol
            pkt_llrs = llrs[bit_start:bit_end]  # 长度 = n * bits_per_symbol

            # 硬判决比特
            hard_bits = (pkt_llrs < 0).astype(np.uint8)

            # 选择最不可靠的 p 个比特位置（LLR 绝对值最小）
            abs_llr = np.abs(pkt_llrs)
            unreliable_pos = np.argsort(abs_llr)[:num_chase_per_packet]

            best_msg = None
            best_score = -np.inf

            # 遍历所有翻转模式
            for flip_pattern in product([0, 1], repeat=num_chase_per_packet):
                test_bits = hard_bits.copy()
                for pos, flip in zip(unreliable_pos, flip_pattern):
                    if flip:
                        test_bits[pos] ^= 1

                # 转换为 GF 符号并尝试 RS 解码
                test_symbols = self.bits_to_gf(test_bits)  # 长度 n
                try:
                    decoded_msg = self.rs.decode(test_symbols)  # 长度 k
                    # 重新编码得到完整码字
                    codeword_sym = self.rs.encode(decoded_msg)   # 长度 n
                    codeword_bits = self.gf_to_bits(codeword_sym)  # 长度 n * bits_per_symbol
                    # 相关性得分：∑ llr * (1-2*bit) ，越大表示越可靠
                    score = np.sum(pkt_llrs * (1 - 2 * codeword_bits.astype(np.float32)))
                    if score > best_score:
                        best_score = score
                        best_msg = decoded_msg
                except galois.ReedSolomonError:
                    continue

            if best_msg is not None:
                decoded_pkts.append(best_msg)
                debug["success_packets"].append(idx)
            else:
                # 所有试探失败，使用硬判决消息（截取码字的前 k 个符号）
                fallback_msg = self.bits_to_gf(hard_bits)[:self.k]
                decoded_pkts.append(fallback_msg)
                debug["partial_packets"].append(idx)

        # 拼接所有消息符号，转换为比特并截断
        gf_dec = self.GF(np.concatenate([pkt.view(np.ndarray) for pkt in decoded_pkts]))
        decoded_bits = self.gf_to_bits(gf_dec)
        decoded_bits = decoded_bits[:original_bit_len]

        return 
    
class LDPCCoder:
    """
    LDPC coder with two matrix modes.

    Engineering mode keeps the legacy smoke-test code:
        c = [u, p]
        p = u A mod 2
        H = [A.T, I]

    IEEE 802.15.3d mode uses the standard THz-SC mandatory LDPC(1440,1344)
    or LDPC(1440,1056) H matrix and solves parity bits over GF(2).
    For 11/15, minus_literal follows Equation 13-1 and supports systematic
    c=[i,p] encoding. plus_systematic_candidate is retained only for backward
    compatibility with earlier experiments.
    """

    def __init__(self, params):
        self.params = params
        self.matrix_type = str(self._get_param("ldpc_matrix_type", "engineering") or "engineering")
        self.standard_rate = str(self._get_param("ldpc_standard_rate", "14/15") or "14/15")
        self.rate11_direction = normalize_ieee802153d_rate11_direction(
            self._get_param("ldpc_rate11_direction", "minus_literal")
        )
        self.rate11_direction_note = (
            ieee802153d_rate11_direction_note(self.rate11_direction)
            if self.standard_rate == "11/15"
            else None
        )
        self.rate11_systematic_candidate = (
            self.standard_rate == "11/15"
            and self.rate11_direction == RATE11_PLUS_SYSTEMATIC_CANDIDATE
        )
        self.is_systematic_candidate = self.rate11_systematic_candidate
        self.standard_confirmation_status = (
            ieee802153d_standard_confirmation_status(self.standard_rate, self.rate11_direction)
            if self.matrix_type == "ieee802153d_1440"
            else "not_applicable_engineering_ldpc"
        )
        if self.matrix_type == "ieee802153d_1440":
            self.n, self.k, self.r = ieee802153d_1440_dimensions(self.standard_rate)
        elif self.matrix_type == "engineering":
            self.n = int(self._get_param("ldpc_n", 672))
            self.k = int(self._get_param("ldpc_k", 336))
            self.r = self.n - self.k
        else:
            raise ValueError("ldpc_matrix_type must be 'engineering' or 'ieee802153d_1440'")
        self.max_iter = int(self._get_param("ldpc_max_iter", 30))
        self.col_weight = int(self._get_param("ldpc_col_weight", self._get_param("ldpc_dv", 3)))
        self.seed = int(self._get_param("ldpc_seed", 2024))
        self.decode_algorithm = str(self._get_param("ldpc_decode_algorithm", "min_sum"))
        self.llr_clip = float(self._get_param("ldpc_llr_clip", 20.0))
        self.hard_llr_value = float(self._get_param("ldpc_hard_llr_value", 12.0))
        self.min_sum_alpha = float(self._get_param("ldpc_min_sum_alpha", 0.8))
        self.efficiency = self.k / self.n

        if self.k <= 0 or self.n <= self.k:
            raise ValueError("LDPC requires ldpc_n > ldpc_k > 0")
        if self.matrix_type == "engineering" and (self.col_weight <= 0 or self.col_weight > self.r):
            raise ValueError("LDPC ldpc_col_weight must be in [1, ldpc_n - ldpc_k]")
        if self.decode_algorithm not in ("min_sum", "normalized_min_sum"):
            raise ValueError("LDPC decode_algorithm must be 'min_sum' or 'normalized_min_sum'")

        if self.matrix_type == "ieee802153d_1440":
            self.H = build_ieee802153d_1440_h(self.standard_rate, self.rate11_direction)
            self.matrix_validation = ieee802153d_1440_matrix_metadata(
                self.standard_rate,
                self.rate11_direction,
            )
            (
                self.info_positions,
                self.parity_positions,
                self.Hp_inv,
                self.parity_matrix,
                self.systematic_info_is_prefix,
            ) = ieee802153d_1440_encoder_matrices(
                self.standard_rate,
                self.rate11_direction,
            )
            self.A = None
        else:
            self.A = self._build_sparse_a_matrix()
            self.H = self._build_engineering_h_matrix()
            self.info_positions = np.arange(self.k, dtype=np.int32)
            self.parity_positions = np.arange(self.k, self.n, dtype=np.int32)
            self.parity_matrix = self.A
            self.Hp_inv = np.eye(self.r, dtype=np.uint8)
            self.systematic_info_is_prefix = True
            self.matrix_validation = {
                "shape": self.H.shape,
                "column_weight": int(self.col_weight + 1),
                "rank": int(self.r),
                "ones": int(np.sum(self.H)),
            }

        self.check_to_vars = self._build_check_to_vars()
        self.var_to_checks = self._build_var_to_checks()
        self.edge_var, self.edge_check, self.check_edges, self.var_edges = self._build_edges()

        self.last_padding_bits = 0
        self.last_num_blocks = 0
        self.last_info_length = 0
        self.last_encoded_length = 0

    def _get_param(self, key, default):
        try:
            return self.params.get(key)
        except KeyError:
            return default

    def _build_sparse_a_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        a = np.zeros((self.k, self.r), dtype=np.uint8)
        for row in range(self.k):
            checks = rng.choice(self.r, size=self.col_weight, replace=False)
            a[row, checks] = 1
        return a

    def _build_engineering_h_matrix(self) -> np.ndarray:
        h = np.zeros((self.r, self.n), dtype=np.uint8)
        h[:, : self.k] = self.A.T
        h[:, self.k :] = np.eye(self.r, dtype=np.uint8)
        return h

    def _build_check_to_vars(self) -> list[np.ndarray]:
        check_to_vars = []
        for check in range(self.r):
            check_to_vars.append(np.flatnonzero(self.H[check]).astype(np.int32))
        return check_to_vars

    def _build_var_to_checks(self) -> list[np.ndarray]:
        var_to_checks = [[] for _ in range(self.n)]
        rows, cols = np.nonzero(self.H)
        for check, var in zip(rows, cols):
            var_to_checks[int(var)].append(int(check))
        return [np.asarray(checks, dtype=np.int32) for checks in var_to_checks]

    def _build_edges(self):
        edge_var = []
        edge_check = []
        check_edges = []
        var_edges = [[] for _ in range(self.n)]
        for check, vars_for_check in enumerate(self.check_to_vars):
            edges = []
            for var in vars_for_check:
                edge_idx = len(edge_var)
                edge_var.append(int(var))
                edge_check.append(int(check))
                edges.append(edge_idx)
                var_edges[int(var)].append(edge_idx)
            check_edges.append(np.asarray(edges, dtype=np.int32))

        return (
            np.asarray(edge_var, dtype=np.int32),
            np.asarray(edge_check, dtype=np.int32),
            check_edges,
            [np.asarray(edges, dtype=np.int32) for edges in var_edges],
        )

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if not np.all(np.isin(bits, [0, 1])):
            raise ValueError("LDPC encode input must be binary bits")

        self.last_info_length = int(len(bits))
        self.last_padding_bits = int((-len(bits)) % self.k)
        if self.last_padding_bits:
            bits_padded = np.concatenate([bits, np.zeros(self.last_padding_bits, dtype=np.uint8)])
        else:
            bits_padded = bits

        info_blocks = bits_padded.reshape(-1, self.k)
        parity = (info_blocks.astype(np.uint16) @ self.parity_matrix.astype(np.uint16)) & 1
        parity = parity.astype(np.uint8, copy=False)
        if self.systematic_info_is_prefix:
            codewords = np.concatenate([info_blocks, parity], axis=1)
        else:
            codewords = np.zeros((info_blocks.shape[0], self.n), dtype=np.uint8)
            codewords[:, self.info_positions] = info_blocks
            codewords[:, self.parity_positions] = parity

        self.last_num_blocks = int(codewords.shape[0])
        self.last_encoded_length = int(codewords.size)
        return codewords.reshape(-1).astype(np.uint8)

    def syndrome(self, codeword: np.ndarray) -> np.ndarray:
        codeword = np.asarray(codeword, dtype=np.uint8).ravel()
        if len(codeword) != self.n:
            raise ValueError(f"LDPC syndrome expects one codeword of length {self.n}")
        syndrome = np.empty(self.r, dtype=np.uint8)
        for check, vars_for_check in enumerate(self.check_to_vars):
            syndrome[check] = np.bitwise_xor.reduce(codeword[vars_for_check])
        return syndrome

    def decode(
        self,
        llrs: np.ndarray,
        original_bit_len: Optional[int] = None,
        ldpc_padding_bits: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        llrs = np.asarray(llrs).ravel()
        used_hard_fallback = bool(np.all(np.isin(llrs, [0, 1])))
        if used_hard_fallback:
            llrs = np.where(llrs.astype(np.uint8) == 0, self.hard_llr_value, -self.hard_llr_value)
        else:
            llrs = llrs.astype(np.float64, copy=False)

        if self.llr_clip > 0:
            llrs = np.clip(llrs, -self.llr_clip, self.llr_clip)

        input_llr_length = int(len(llrs))
        llr_padding_bits = int((-len(llrs)) % self.n)
        if llr_padding_bits:
            llrs = np.pad(llrs, (0, llr_padding_bits), mode="constant", constant_values=0.0)

        blocks = llrs.reshape(-1, self.n)
        decoded_blocks = []
        iterations = []
        syndrome_weights = []
        success_blocks = []
        failed_blocks = []

        for block_idx, block_llrs in enumerate(blocks):
            hard_bits, posterior, used_iter, success = self._decode_block(block_llrs)
            syndrome_weight = int(np.sum(self.syndrome(hard_bits)))
            decoded_blocks.append(hard_bits[self.info_positions])
            iterations.append(int(used_iter))
            syndrome_weights.append(syndrome_weight)
            if success:
                success_blocks.append(block_idx)
            else:
                failed_blocks.append(block_idx)

        decoded_bits = np.concatenate(decoded_blocks).astype(np.uint8) if decoded_blocks else np.array([], dtype=np.uint8)
        padding = int(self.last_padding_bits if ldpc_padding_bits is None else ldpc_padding_bits)
        if padding > 0 and len(decoded_bits) >= padding:
            decoded_bits = decoded_bits[:-padding]
        if original_bit_len is not None:
            decoded_bits = decoded_bits[: int(original_bit_len)]

        debug = {
            "algorithm": self.decode_algorithm,
            "ldpc_matrix_type": self.matrix_type,
            "ldpc_standard_rate": self.standard_rate if self.matrix_type == "ieee802153d_1440" else None,
            "ldpc_rate11_direction": self.rate11_direction if self.standard_rate == "11/15" else None,
            "ldpc_rate11_direction_note": self.rate11_direction_note,
            "ldpc_rate11_systematic_candidate": self.rate11_systematic_candidate,
            "is_systematic_candidate": self.is_systematic_candidate,
            "standard_confirmation_status": self.standard_confirmation_status,
            "systematic_info_is_prefix": self.systematic_info_is_prefix,
            "ldpc_n": self.n,
            "ldpc_k": self.k,
            "ldpc_r": self.r,
            "num_blocks": int(len(blocks)),
            "input_llr_length": input_llr_length,
            "llr_padding_bits": llr_padding_bits,
            "ldpc_padding_bits": padding,
            "used_hard_fallback": used_hard_fallback,
            "iterations": iterations,
            "success_blocks": success_blocks,
            "failed_blocks": failed_blocks,
            "syndrome_weights": syndrome_weights,
            "decode_success": len(failed_blocks) == 0,
        }
        return decoded_bits, debug

    def _decode_block(self, llr_block: np.ndarray):
        llr = np.asarray(llr_block, dtype=np.float64).ravel()
        if len(llr) != self.n:
            raise ValueError(f"LDPC decode block expects length {self.n}")

        edge_count = len(self.edge_var)
        var_to_check = llr[self.edge_var].copy()
        check_to_var = np.zeros(edge_count, dtype=np.float64)
        posterior = llr.copy()
        min_sum_scale = self.min_sum_alpha if self.decode_algorithm == "normalized_min_sum" else 1.0
        hard_bits = (posterior < 0).astype(np.uint8)
        if np.all(self.syndrome(hard_bits) == 0):
            return hard_bits, posterior, 0, True

        eps = 1e-12
        for iteration in range(1, self.max_iter + 1):
            for edges in self.check_edges:
                messages = var_to_check[edges]
                signs = np.where(messages < 0, -1.0, 1.0)
                abs_messages = np.maximum(np.abs(messages), eps)
                total_sign = np.prod(signs)

                if len(abs_messages) == 1:
                    min_values = abs_messages
                else:
                    min_index = int(np.argmin(abs_messages))
                    min1 = abs_messages[min_index]
                    min2 = np.min(np.delete(abs_messages, min_index))
                    min_values = np.full(len(edges), min1, dtype=np.float64)
                    min_values[min_index] = min2

                check_to_var[edges] = min_sum_scale * total_sign * signs * min_values

            posterior = llr.copy()
            for var, edges in enumerate(self.var_edges):
                if len(edges) > 0:
                    posterior[var] += np.sum(check_to_var[edges])
                    var_to_check[edges] = posterior[var] - check_to_var[edges]

            if self.llr_clip > 0:
                posterior = np.clip(posterior, -self.llr_clip, self.llr_clip)
                var_to_check = np.clip(var_to_check, -self.llr_clip, self.llr_clip)

            hard_bits = (posterior < 0).astype(np.uint8)
            if np.all(self.syndrome(hard_bits) == 0):
                return hard_bits, posterior, iteration, True

        return hard_bits, posterior, self.max_iter, False
    

