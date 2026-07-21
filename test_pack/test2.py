import galois
from math import comb
import numpy as np
from collections import defaultdict
from transmitter.THzTransmitter import THzTransmitter
from params.PHYParams import PHYParams
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.GIremover import GIRemover
from receiver.DeModulator import THzDemodulator

class GSDecoder:
    """
    Koetter-Vardy 软判决 RS 译码器（基于你原有 GS 代码完整扩展）
    支持：输入比特 LLR → 输出译码消息/码字
    """
    class QPoly:
        """内部二元多项式 Q(x,y)，完全保留你的原逻辑"""
        def __init__(self, GF, ell):
            self.GF = GF
            self.ell = ell
            self.q = [galois.Poly.Zero(GF) for _ in range(ell+1)]

        def copy(self):
            r = GSDecoder.QPoly(self.GF, self.ell)
            r.q = [galois.Poly(poly.coefficients(), field=self.GF) for poly in self.q]
            return r

        def weighted_degree(self, w):
            degs = [q.degree + w*j if q != 0 else -1 for j,q in enumerate(self.q)]
            return max(degs) if degs else -1

        def lead_index(self, w):
            return max(range(self.ell+1), 
                       key=lambda j: self.q[j].degree + w*j if self.q[j] != 0 else -1)

        def scale(self, c):
            r = self.copy()
            r.q = [c*q for q in r.q]
            return r

        def add_scaled(self, other, c):
            for j in range(self.ell+1):
                self.q[j] += c*other.q[j]

        def mul_x_minus_xi(self, xi):
            for j in range(self.ell+1):
                self.q[j] *= galois.Poly([1, -xi], field=self.GF)

        def hasse(self, dx, dy, xi, yi):
            total = self.GF(0)
            for j in range(dy, self.ell+1):
                qj = self.q[j]
                if qj == 0 or qj.degree < dx:
                    continue
                if dx == 0:
                    deriv = qj(xi)
                else:
                    deriv = qj.derivative(dx)(xi)
                comb_val = self.GF(comb(j, dy))
                total += comb_val * deriv * (yi ** (j - dy))
            return total

        def evaluate(self, x_val, y_val):
            result = self.GF(0)
            for j, qj in enumerate(self.q):
                if qj == 0:
                    continue
                qj_x = qj(x_val)
                y_pow = y_val ** j
                result += qj_x * y_pow
            return result

        def to_univariate_y(self, x_fixed=None):
            if x_fixed is None:
                coeffs = self.q[::-1]
            else:
                coeffs = [qj(x_fixed) if qj != 0 else self.GF(0) for qj in self.q]
                coeffs = coeffs[::-1]
            
            while len(coeffs) > 1 and coeffs[-1] == 0:
                coeffs.pop()
            
            if all(c == 0 for c in coeffs):
                return galois.Poly([0], field=self.GF)
            
            return galois.Poly(coeffs, field=self.GF)

        def divide_by_y_minus_f(self, f):
            GF = self.GF
            ell = self.ell
            S = GSDecoder.QPoly(GF, ell-1)
            remainder = GF(0)
            coeffs = self.q.copy()
            for j in range(len(coeffs)-1, 0, -1):
                if coeffs[j] == 0:
                    S.q[j-1] = galois.Poly.Zero(GF)
                    continue
                S.q[j-1] = coeffs[j]
                coeffs[j-1] += S.q[j-1] * f
            remainder = coeffs[0] if len(coeffs) > 0 else GF(0)
            return S, remainder

        def divide_by_x_power(self, r):
            if r == 0:
                return self.copy()
            
            new_Q = GSDecoder.QPoly(self.GF, self.ell)
            for j in range(self.ell + 1):
                poly = self.q[j]
                if poly.degree == -1:
                    new_Q.q[j] = galois.Poly.Zero(self.GF)
                    continue
                
                if poly.degree < r:
                    new_Q.q[j] = galois.Poly.Zero(self.GF)
                    continue
                
                coeffs = poly.coefficients()
                if len(coeffs) <= r:
                    new_coeffs = self.GF([])
                else:
                    new_coeffs = coeffs[:-r]
                
                if len(new_coeffs) == 0:
                    new_Q.q[j] = galois.Poly.Zero(self.GF)
                else:
                    new_Q.q[j] = galois.Poly(new_coeffs, field=self.GF)
            
            return new_Q

        def max_x_divisor(self):
            max_r = 0
            while True:
                next_r = max_r + 1
                can_divide_next = True
                for j in range(self.ell + 1):
                    poly = self.q[j]
                    if poly.degree == -1:
                        continue
                    coeffs = poly.coefficients()
                    for i in range(1, next_r + 1):
                        if i > len(coeffs): continue
                        if coeffs[-i] != self.GF(0):
                            can_divide_next = False
                            break
                    if not can_divide_next: break
                if can_divide_next:
                    max_r = next_r
                else:
                    break
            return max_r

        def substitute_y(self, func):
            new_Q = GSDecoder.QPoly(self.GF, self.ell * 2)
            temp_q = [galois.Poly.Zero(self.GF) for _ in range(new_Q.ell + 1)]
            for j_old in range(self.ell + 1):
                if self.q[j_old] == 0: continue
                substitutions = func(j_old)
                for coeff, j_new in substitutions:
                    while j_new >= len(temp_q):
                        temp_q.append(galois.Poly.Zero(self.GF))
                    coeff_gf = coeff if isinstance(coeff, self.GF) else coeff
                    temp_q[j_new] += coeff_gf * self.q[j_old]
            new_Q.ell = len(temp_q) - 1
            new_Q.q = temp_q
            return new_Q
        
        def truncate(self, k):
            new_Q = GSDecoder.QPoly(self.GF, self.ell)
            for j in range(self.ell + 1):
                poly = self.q[j]
                if poly.degree == -1:
                    new_Q.q[j] = galois.Poly.Zero(self.GF)
                    continue
                coeffs = poly.coefficients()
                if len(coeffs) > k:
                    truncated_coeffs = coeffs[-k:]
                else:
                    truncated_coeffs = coeffs
                new_Q.q[j] = galois.Poly(truncated_coeffs, field=self.GF)
            return new_Q

        def substitute_half_root(self, hi, di):
            GF = self.GF
            def sub_func(j_old):
                substitutions = []
                for t in range(j_old + 1):
                    if GF.characteristic == 2 and (t & j_old) != t:
                        continue
                    c = GF(comb(j_old, t))
                    hi_pow = hi ** (j_old - t) if (j_old - t) > 0 else GF(1)
                    x_pow = galois.Poly([1] + [0]*(di*t), field=GF) if di*t > 0 else GF(1)
                    total_coeff = c * hi_pow * x_pow
                    substitutions.append((total_coeff, t))
                return substitutions
            return self.substitute_y(sub_func)    

        def __repr__(self):
            s = []
            for j,q in enumerate(self.q):
                if q != 0:
                    s.append(f"({q})y^{j}")
            return " + ".join(s) if s else "0"

    # -------------------------------------------------------------------------
    # 类初始化
    # -------------------------------------------------------------------------
    def __init__(self, n: int, k: int, eval_pts=None, GF=galois.GF(2**8)):
        self.n = n          # RS 码长
        self.k = k          # 信息长度
        self.GF = GF        # 有限域
        self.w = k - 1      # 加权次数权重
        
        # 默认评估点（1,2,3,...n）
        if eval_pts is None:
            self.eval_pts = [GF(i) for i in range(1, n+1)]
        else:
            self.eval_pts = [GF(x) for x in eval_pts]

    # -------------------------------------------------------------------------
    # ===================== ① 新增：LLR → 符号可靠性矩阵 =====================
    # 对接你的 THzDemodulator 输出
    # -------------------------------------------------------------------------
    def llr_to_reliability_matrix(self, llr_sequence, bits_per_symbol):
        """
        输入：解调器输出的一维 LLR 数组
        输出：q × n 可靠性矩阵 Pi，Pi[i,j] = 符号 j 为 i 的概率
        """
        n = self.n
        q = self.GF.order
        m = bits_per_symbol
        
        # 重塑 LLR：[n 符号, m 比特]
        llr_matrix = llr_sequence[:n*m].reshape(n, m)
        
        # 生成符号→比特映射表
        symbol_bits = np.zeros((q, m), dtype=np.uint8)
        for s in range(q):
            bits = [(s >> b) & 1 for b in range(m-1, -1, -1)]
            symbol_bits[s] = bits
        
        # 计算符号概率
        log_P = np.zeros((n, q))
        for j in range(n):
            llrs = llr_matrix[j]
            for s in range(q):
                bits = symbol_bits[s]
                log_P[j, s] = -np.sum(bits * llrs)
        
        # 数值稳定归一化
        P = np.exp(log_P - np.max(log_P, axis=1, keepdims=True))
        Pi = P / np.sum(P, axis=1, keepdims=True)
        return Pi.T  # [q, n]

    # -------------------------------------------------------------------------
    # ===================== ② 新增：可靠性 → 重数矩阵 =====================
    # -------------------------------------------------------------------------
    def get_multiplicity_matrix(self, Pi, total_multiplicity=20):
        """
        可靠性矩阵 → KV 重数矩阵 M (q × n)
        """
        q, n = Pi.shape
        M = np.zeros((q, n), dtype=int)
        remaining = total_multiplicity
        
        while remaining > 0:
            best_score = -1
            best_j = best_i = 0
            
            for j in range(n):
                col = Pi[:, j]
                i_max = np.argmax(col)
                score = col[i_max] / (M[i_max, j] + 1)
                
                if score > best_score:
                    best_score = score
                    best_i = i_max
                    best_j = j
            
            M[best_i, best_j] += 1
            remaining -= 1
        
        return M

    # -------------------------------------------------------------------------
    # ===================== ③ 核心：Koetter-Vardy 插值 =====================
    # 输入：重数矩阵 M → 输出：插值多项式 Q(x,y)
    # -------------------------------------------------------------------------
    def kv_interpolate(self, M):
        GF = self.GF
        n = self.n
        k = self.k
        w = self.w
        eval_pts = self.eval_pts
        
        # 1. 计算总约束数
        cost = 0
        max_m = 0
        constraints = []
        for j in range(n):
            for i in range(GF.order):
                r = int(M[i, j])
                if r <= 0: continue
                cost += r * (r + 1) // 2
                max_m = max(max_m, r)
                xi = eval_pts[j]
                yi = GF(i)
                for dy in range(r):
                    for dx in range(r - dy):
                        constraints.append((xi, yi, dx, dy))
        
        # 2. 确定单项式空间
        max_ell = int(np.sqrt(2 * cost / k) + 2)
        max_x_deg = int(np.sqrt(2 * cost * k) + k + 2)
        
        monomials = []
        for q_deg in range(max_ell + 1):
            max_p = max_x_deg - w * q_deg
            if max_p < 0: continue
            for x_deg in range(0, max_p + 1):
                monomials.append((x_deg, q_deg))
        
        if not monomials:
            raise ValueError("单项式空间为空，降低重数")
        
        # 3. 构建插值矩阵
        n_rows = len(constraints)
        n_cols = len(monomials)
        mat = np.zeros((n_rows, n_cols), dtype=int)
        
        for row, (xi, yi, dx, dy) in enumerate(constraints):
            for col, (p, q_deg) in enumerate(monomials):
                if q_deg < dy or p < dx:
                    mat[row, col] = 0
                    continue
                
                c_x = comb(p, dx)
                xi_pow = xi ** (p - dx)
                c_y = comb(q_deg, dy)
                yi_pow = yi ** (q_deg - dy)
                val = GF(c_x) * xi_pow * GF(c_y) * yi_pow
                mat[row, col] = int(val)
        
        # 4. 求零空间 → 得到 Q 系数
        gf_mat = GF(mat)
        ker = gf_mat.right_kernel()
        if ker.dimension() == 0:
            raise ValueError("插值失败：无非零解")
        
        # 5. 系数 → QPoly
        q_coeffs = ker.basis()[0]
        max_q_deg = max(d for _, d in monomials)
        Q = self.QPoly(GF, max_q_deg)
        
        for (p, q_deg), coeff in zip(monomials, q_coeffs):
            if coeff == 0: continue
            poly = galois.Poly([int(coeff)] + [0]*p, field=GF)
            Q.q[q_deg] += poly
        
        return Q

    # -------------------------------------------------------------------------
    # 根查找（你的原有代码）
    # -------------------------------------------------------------------------
    def _roth_ruckenstein(self, Q, precision=None, multiplicities=None):
        GF = self.GF
        ell = max(multiplicities) if multiplicities is not None else 1
        degree_bound = self.w * ell

        def rr_recursive(Q, lam, k_remaining, g):
            solutions = []
            if precision is not None and k_remaining <= 0:
                solutions.append((g, lam))
                return solutions

            val = Q.max_x_divisor()
            if precision is not None:
                k_remaining -= val
            M = Q.divide_by_x_power(val)
            M0y = M.to_univariate_y(x_fixed=GF(0))

            if M0y == 0 or (precision is not None and k_remaining <= 0):
                if precision is not None:
                    solutions.append((g, lam))
                else:
                    solutions.append(g)
                return solutions

            roots = M0y.roots(multiplicity=False)
            unique_roots = []
            seen = set()
            for r in roots:
                if int(r) not in seen:
                    seen.add(int(r))
                    unique_roots.append(r)

            for gamma in unique_roots:
                x_lam = galois.Poly([1] + [0]*lam, GF)
                g_new = g + gamma * x_lam

                if lam < degree_bound:
                    def sub_func(j_old):
                        subs = []
                        for t in range(j_old+1):
                            if GF.characteristic==2 and (t&j_old)!=t: continue
                            c = GF(comb(j_old,t)) * (gamma**(j_old-t))
                            xt = galois.Poly([1]+[0]*t, GF)
                            subs.append((c*xt, t))
                        return subs
                    M_bar = M.substitute_y(sub_func)
                    subsol = rr_recursive(M_bar, lam+1, k_remaining, g_new)
                    solutions.extend(subsol)
                else:
                    if precision is not None:
                        solutions.append((g_new, lam+1))
                    else:
                        if M.evaluate(GF(0), gamma) == 0:
                            solutions.append(g_new)
            return solutions

        raw = rr_recursive(Q, 0, precision, galois.Poly.Zero(GF))
        if precision is None:
            uniq = []
            seen = set()
            for p in raw:
                s = str(p.coefficients())
                if s not in seen:
                    seen.add(s)
                    uniq.append(p)
            return uniq
        else:
            uniq = []
            seen = set()
            for h,d in raw:
                key = f"{h}:{d}"
                if key not in seen:
                    seen.add(key)
                    uniq.append((h,d))
            return uniq

    # -------------------------------------------------------------------------
    # ===================== ④ 完整译码接口（对外使用） =====================
    # -------------------------------------------------------------------------
    def decode_llr(self, llr_sequence, bits_per_symbol, total_multiplicity=20):
        """
        最终使用接口！
        输入：解调器输出的 LLR 数组
        输出：候选消息多项式列表
        """
        # 1. LLR → 可靠性矩阵
        Pi = self.llr_to_reliability_matrix(llr_sequence, bits_per_symbol)
        
        # 2. 可靠性 → 重数矩阵
        M = self.get_multiplicity_matrix(Pi, total_multiplicity)
        
        # 3. KV 插值 → Q(x,y)
        Q = self.kv_interpolate(M)
        
        # 4. 根查找 → 候选多项式
        mult = np.max(M) if isinstance(M, np.ndarray) else 1
        polys = self._roth_ruckenstein(Q, multiplicities=[mult]*self.n)
        
        # 过滤有效多项式
        valid = []
        for f in polys:
            if f.degree <= self.k - 1:
                valid.append(f)
        
        return valid
    
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
    # 进行匹配滤波并获得中间信号
    rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)    