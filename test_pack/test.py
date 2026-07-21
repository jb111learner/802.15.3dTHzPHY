import galois
import numpy as np
from math import comb

# ========================
# 你的 QPoly 类（完全不变）
# ========================
class QPoly:
    def __init__(self, GF, ell):
        self.GF = GF
        self.ell = ell
        self.q = [galois.Poly.Zero(GF) for _ in range(ell+1)]

    def copy(self):
        r = QPoly(self.GF, self.ell)
        r.q = [galois.Poly(p.coefficients(), field=self.GF) for p in self.q]
        return r

    def weighted_degree(self, w):
        degs = []
        for j, q in enumerate(self.q):
            if q == 0:
                degs.append(-1)
            else:
                degs.append(q.degree + w * j)
        return max(degs) if degs else -1

    def lead_index(self, w):
        best_j = 0
        best_wd = -1
        for j, q in enumerate(self.q):
            if q == 0:
                continue
            wd = q.degree + w * j
            if wd > best_wd:
                best_wd = wd
                best_j = j
        return best_j

    def scale(self, c):
        r = self.copy()
        r.q = [c * p for p in r.q]
        return r

    def add_scaled(self, other, c):
        for j in range(self.ell + 1):
            self.q[j] += c * other.q[j]

    def mul_x_minus_xi(self, xi):
        x_poly = galois.Poly([1, -xi], field=self.GF)
        for j in range(self.ell + 1):
            self.q[j] *= x_poly

    def hasse(self, dx, dy, xi, yi):
        total = self.GF(0)
        for j in range(dy, self.ell + 1):
            qj = self.q[j]
            if qj == 0 or qj.degree < dx:
                continue
            if dx == 0:
                d_val = qj(xi)
            else:
                d_val = qj.derivative(dx)(xi)
            c = self.GF(comb(j, dy))
            total += c * d_val * (yi ** (j - dy))
        return total

    def evaluate(self, x, y):
        res = self.GF(0)
        for j, qj in enumerate(self.q):
            if qj == 0:
                continue
            res += qj(x) * (y ** j)
        return res

    def __repr__(self):
        terms = []
        for j, q in enumerate(self.q):
            if q != 0:
                terms.append(f"({q})y^{j}")
        return " + ".join(terms) if terms else "0"

# -------------------------------------------------------------------------
# ✅ 严格对齐伪代码的 Kötter 插值（重数 2 完全正常）
# -------------------------------------------------------------------------
def kotter_interpolate(points, multiplicities, GF, n, k, w):
    GF = GF
    L = max(multiplicities)
    w = w

    # ======================
    # 伪代码行 1–2：初始化 g_j = y^j
    # ======================
    g = [QPoly(GF, L) for _ in range(L + 1)]
    for j in range(L + 1):
        g[j].q[j] = galois.Poly.One(GF)

    # ======================
    # 伪代码行 3：遍历每个点 i = 1..n
    # ======================
    for (xi, yi), m in zip(points, multiplicities):
        # ======================
        # 伪代码行 4：严格！(r,s) 从 (0,0) → (m_i-1, 0)
        # 🔥🔥🔥 这是你之前错误的核心！
        # ======================
        for r in range(m):
            dx, dy = r, 0

            # ======================
            # 伪代码行 5：计算 Δ_j = D(g_j)
            # ======================
            Delta = [p.hasse(dx, dy, xi, yi) for p in g]

            # ======================
            # 伪代码行 6：J = {j | Δ_j ≠ 0}
            # ======================
            J = [j for j in range(L + 1) if Delta[j] != 0]
            if not J:
                continue

            # ======================
            # 伪代码行 8–9：选择最小多项式 j*
            # ======================
            j_star = min(J, key=lambda j: g[j].weighted_degree(w))
            f = g[j_star]
            delta_f = Delta[j_star]

            # ======================
            # 伪代码行 10–14：更新（必须先备份再修改）
            # ======================
            new_g = [p.copy() for p in g]
            for j in J:
                if j != j_star:
                    # 伪代码：g_j = delta_f * g_j - Δ_j * f
                    lam = Delta[j] / delta_f
                    new_g[j].add_scaled(f, -lam)
                else:
                    # 伪代码：g_j = delta_f * (x - xi) * f
                    new_g[j] = f.copy()
                    new_g[j].mul_x_minus_xi(xi)
                    new_g[j].scale(delta_f)
            g = new_g

    # ======================
    # 伪代码行 15：输出最小多项式
    # ======================
    return min(g, key=lambda poly: poly.weighted_degree(w))

# # -------------------------------------------------------------------------
# # Kötter 插值（严格伪代码）
# # -------------------------------------------------------------------------
# def kotter_interpolate(points, multiplicities, GF, n, k, w):
#     GF = GF
#     L = max(multiplicities)
#     g = [QPoly(GF, L) for _ in range(L+1)]
#     for j in range(L+1):
#         g[j].q[j] = galois.Poly.One(GF)

#     for (xi, yi), m in zip(points, multiplicities):
#         for r in range(m):
#             Delta = [p.hasse(r, 0, xi, yi) for p in g]
#             J = [j for j in range(L+1) if Delta[j] != 0]
#             if not J:
#                 continue
#             js = min(J, key=lambda j: g[j].weighted_degree(w))
#             fj = g[js]
#             df = Delta[js]
#             newg = [p.copy() for p in g]
#             for j in J:
#                 if j != js:
#                     newg[j].add_scaled(fj, -Delta[j]/df)
#                 else:
#                     n = fj.copy()
#                     n.mul_x_minus_xi(xi)
#                     newg[js] = n.scale(df)
#             g = newg
#     return min(g, key=lambda q: q.weighted_degree(w))

# ========================
# 🧪 自动测试验证函数
# ========================
def test_kotter_interpolation():
    print("="*60)
    print(" Starting Kötter Interpolation Test ")
    print("="*60)

    # 1. 设置参数
    GF = galois.GF(2**8)
    n = 8
    k = 8
    w = k - 1  # 伪代码指定权重
    m = 2      # 重数

    # 2. 随机生成正确多项式 f(x)
    f = galois.Poly.Random(k-1, field=GF)
    print(f"✅ 原始信息多项式 f(x): {f}")

    # 3. 生成码点与无噪声点
    alpha = GF.primitive_element
    xs = [alpha**i for i in range(n)]
    ys = [f(x) for x in xs]
    points = list(zip(xs, ys))
    multiplicities = [m]*n
    # multiplicities[0] = 1  # 给第一个点重数1，测试重数分配

    # 4. 运行 Kötter 插值
    print("\n⏳ Running Kötter interpolation...")
    Q = kotter_interpolate(points, multiplicities, GF, n, k, w)
    print(f"\n✅ 插值结果 Q(x,y):")
    print(Q)

    # 5. 验证 1：所有 Hasse 约束都满足
    print("\n" + "="*50)
    print("验证 1：所有 Hasse 导数约束 D_{r,0} Q(α_i, y_i) = 0")
    print("="*50)
    all_ok = True
    for (xi, yi), m_val in zip(points, multiplicities):
        for r in range(m_val):
            val = Q.hasse(r, 0, xi, yi)
            if val != 0:
                print(f"❌ 约束失败 at (x={xi}, y={yi}), r={r}: {val}")
                all_ok = False
    if all_ok:
        print("✅ 所有 Hasse 约束均满足！")

    # 6. 验证 2：Q(x, f(x)) = 0（核心GS条件）
    print("\n" + "="*50)
    print("验证 2：Q(x, f(x)) = 0（必须成立）")
    print("="*50)
    test_x = GF.Random()
    q_val = Q.evaluate(test_x, f(test_x))
    print(f"测试点 x = {test_x}, Q(x,f(x)) = {q_val}")
    if q_val == 0:
        print("✅ 完美满足 Q(x,f(x))=0！")
    else:
        print("❌ 不满足 Q(x,f(x))=0，插值错误！")

    # 7. 验证 3：输出不固定（两次运行结果不同）
    print("\n" + "="*50)
    print("验证 3：插值输出随输入变化")
    print("="*50)
    f2 = galois.Poly.Random(k-1, field=GF)
    ys2 = [f2(x) for x in xs]
    points2 = list(zip(xs, ys2))
    Q2 = kotter_interpolate(points2, multiplicities, GF, n, k, w)

    if str(Q) != str(Q2):
        print("✅ 输出随输入变化，算法正常！")
    else:
        print("❌ 输出固定不变，算法异常！")

    print("\n" + "="*60)
    print("✅ 测试全部完成！")
    print("="*60)

if __name__ == "__main__":
    test_kotter_interpolation()






def gs_knh_interpolate(points, multiplicities, GF, n, k, w):
    GF = GF
    ell = max(multiplicities)
    B = []
    for j in range(ell+1):
        p = QPoly(GF, ell)
        p.q[j] = galois.Poly.One(GF)
        B.append(p)

    for (xi, yi), m in zip(points, multiplicities):
        Ds = [(dx,dy) for dx in range(m) for dy in range(m) if dx+dy < m]
        for dx, dy in Ds:
            best = None
            best_deg = float("inf")
            for b in B:
                b_val = b.hasse(dx, dy, xi, yi)
                if b_val != 0:
                    d = b.weighted_degree(w)
                    if d < best_deg:
                        best_deg = d
                        best = b
            if best is None: continue
            bt = best
            bt_val = bt.hasse(dx, dy, xi, yi)
            newB = []
            for bj in B:
                if bj is bt: continue
                val = bj.hasse(dx, dy, xi, yi)
                if val == 0:
                    newB.append(bj)
                else:
                    lam = val / bt_val
                    bj2 = bj.copy()
                    bj2.add_scaled(bt, -lam)
                    newB.append(bj2)
            bt2 = bt.copy()
            bt2.mul_x_minus_xi(xi)
            newB.append(bt2)
            B = newB
    return min(B, key=lambda b: b.weighted_degree(w))
