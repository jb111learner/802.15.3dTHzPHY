import galois
from math import comb
import numpy as np

class GSDecoder:
    """
    Guruswami-Sudan 译码器类
    初始化传入 RS 码核心参数 n, k
    提供统一 decode() 方法完成全流程译码
    """
    class QPoly:
        """内部二元多项式 Q(x,y)，完全保留原逻辑"""
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
    # 类初始化：传入 RS 码核心参数 n, k
    # -------------------------------------------------------------------------
    def __init__(self, n: int, k: int, GF=galois.GF(2**8)):
        self.n = n          # RS 码长
        self.GF = GF        # 有限域
        self.k = k          # RS 信息符号数
        self.w = k-1      # 加权次数权重
    # -------------------------------------------------------------------------
    # Roth-Ruckenstein 根查找（内部方法）
    # -------------------------------------------------------------------------
    def _roth_ruckenstein(self, Q, precision=None, multiplicities=None):
        GF = self.GF
        ell = max(multiplicities) if multiplicities is not None else 1
        degree_bound = self.w * ell  # 理论上最大加权次数为w*ell，实际可能更小

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
    # Alekhnovich 根查找（内部方法）
    # -------------------------------------------------------------------------
    def _alekhnovich(self, Q, precision=None, dc_threshold=3, multiplicities=None):
        GF = self.GF
        ell = max(multiplicities) if multiplicities is not None else 1
        degree_bound = self.w * ell

        def alekh_rec(Q, k, db, lvl):
            if k <= 0:
                return [(galois.Poly.Zero(GF), 0)]
            if db < 0:
                if Q.evaluate(GF(0),GF(0))==0 or Q.max_x_divisor()>=k:
                    return [(galois.Poly.Zero(GF),0)]
                else:
                    return []
            if k==1 or db==0:
                M0y = Q.to_univariate_y(x_fixed=GF(0))
                if M0y==0: return [(galois.Poly.Zero(GF),0)]
                roots = M0y.roots(multiplicity=False)
                return [(galois.Poly([r],GF),1) for r in roots]
            if dc_threshold is not None and k < dc_threshold:
                return self._roth_ruckenstein(Q, precision=k)

            Q_trunc = Q.truncate(k)
            half = alekh_rec(Q_trunc, k//2, db, lvl+1)
            full = []
            for hi,di in half:
                Qht = Q_trunc.substitute_half_root(hi,di)
                if all(p==0 for p in Qht.q):
                    full.append((hi,di))
                else:
                    val = Qht.max_x_divisor()
                    Qh = Qht.divide_by_x_power(val)
                    sec = alekh_rec(Qh, k-val, db-di, lvl+1)
                    for hij,dij in sec:
                        shift = hij * galois.Poly([1]+[0]*di,GF) if di>0 else hij
                        full.append( (hi+shift, di+dij) )

            uniq = []
            seen=set()
            for h,d in full:
                key=f"{h}:{d}"
                if key not in seen:
                    seen.add(key)
                    uniq.append((h,d))
            return uniq

        if precision is None:
            max_wd = 0
            for j in range(Q.ell+1):
                if Q.q[j]!=0:
                    max_wd = max(max_wd, Q.q[j].degree + degree_bound*j)
            k = 1 + max_wd
        else:
            k = precision

        mod_roots = alekh_rec(Q, k, degree_bound, 0)

        if precision is None:
            roots = []
            seen=set()
            for hi,_ in mod_roots:
                if hi.degree>degree_bound: continue
                ok=True
                for x in [GF(1),GF(2),GF(3)]:
                    if Q.evaluate(x, hi(x))!=0:
                        ok=False
                        break
                if ok:
                    s=str(hi.coefficients())
                    if s not in seen:
                        seen.add(s)
                        roots.append(hi)
            return roots
        else:
            uniq=[]
            seen=set()
            for h,d in mod_roots:
                key=f"{h}:{d}"
                if key not in seen:
                    seen.add(key)
                    uniq.append((h,d))
            return uniq
        