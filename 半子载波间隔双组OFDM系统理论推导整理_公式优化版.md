# 半子载波间隔双组 OFDM 系统理论推导整理

> 说明：本版本采用标准 Markdown 数学公式格式： - 行间公式使用
> `$$ ... $$` - 行内公式使用 `$ ... $` 适用于 Typora、Obsidian、VS Code
> Markdown、GitHub MathJax 等环境。

------------------------------------------------------------------------

## 1. 系统模型概述

传统 OFDM 子载波间隔：

$$
\Delta f=\frac{1}{T}
$$

本文采用半子载波间隔：

$$
\Delta f=\frac{1}{2T}
$$

将子载波划分为两组：

第一组：

$$
f_0(k)=\frac{k}{T}
$$

第二组：

$$
f_1(k)=\frac{k+\frac{1}{2}}{T}
$$

两组内部满足正交性，但两组之间存在非正交耦合。

------------------------------------------------------------------------

# 2. 发射信号模型

两组发送符号：

$$
\mathbf a_0=
[a_0^{(0)},a_1^{(0)},...,a_{K-1}^{(0)}]^T
$$

$$
\mathbf a_1=
[a_0^{(1)},a_1^{(1)},...,a_{K-1}^{(1)}]^T
$$

第一组：

$$
x_0(t)=
\sum_{k=0}^{K-1}
a_k^{(0)}
e^{j2\pi kt/T}
$$

第二组：

$$
x_1(t)=
\sum_{k=0}^{K-1}
a_k^{(1)}
e^{j2\pi kt/T}
$$

半子载波偏移：

$$
\tilde{x}_1(t)=x_1(t)e^{j\pi t/T}
$$

因此：

$$
\tilde{x}_1(t)
=
\sum_{k=0}^{K-1}
a_k^{(1)}
e^{j2\pi(k+\frac12)t/T}
$$

最终：

$$
s(t)=
\sqrt{\frac{P}{2T}}
\left(
x_0(t)+\tilde{x}_1(t)
\right)
$$

------------------------------------------------------------------------

# 3. 信道模型

信道冲激响应：

$$
h(\tau)
$$

系统函数：

$$
H(f)=
\int_{-\infty}^{+\infty}
h(\tau)e^{-j2\pi f\tau}d\tau
$$

两组子载波信道：

$$
H_0(k)=H(k/T)
$$

$$
H_1(k)=H((k+\frac12)/T)
$$

定义：

$$
\mathbf H_0=
diag(H_0(0),...,H_0(K-1))
$$

$$
\mathbf H_1=
diag(H_1(0),...,H_1(K-1))
$$

------------------------------------------------------------------------

# 4. 组间耦合矩阵

两组子载波之间：

$$
\Delta f=
\frac{k-m+\frac12}{T}
$$

对应积分：

$$
\int_0^T
e^{j2\pi(k-m+\frac12)t/T}dt
$$

计算得到：

$$
\frac{jT}
{\pi(k-m+\frac12)}
$$

定义归一化耦合矩阵：

$$
P_{m,k}
=
\frac{j}
{\pi(k-m+\frac12)}
$$

由于：

$$
P_{m,k}=f(k-m)
$$

所以：

$$
\mathbf P
$$

为 Toeplitz 矩阵。

------------------------------------------------------------------------

# 5. 完整系统模型

定义：

$$
\mathbf C=
\begin{bmatrix}
I&P\\
P^H&I
\end{bmatrix}
$$

$$
\mathbf D=
\begin{bmatrix}
H_0&0\\
0&H_1
\end{bmatrix}
$$

发送符号：

$$
\mathbf a=
\begin{bmatrix}
\mathbf a_0\\
\mathbf a_1
\end{bmatrix}
$$

接收模型：

$$
\mathbf y=
\mu T\mathbf C\mathbf D\mathbf a+\mathbf n
$$

------------------------------------------------------------------------

# 6. 噪声协方差

连续时间复白噪声：

$$
E[n(t)n^*(\tau)]
=
N_0\delta(t-\tau)
$$

偶数组：

$$
n_0(k)=
\int_0^T
n(t)e^{-j2\pi kt/T}dt
$$

得到：

$$
E[n_0(m)n_0(k)^*]
=
N_0T\delta(m-k)
$$

因此：

$$
R_{n0}=N_0TI
$$

同理：

$$
R_{n1}=N_0TI
$$

两组之间：

$$
E[n_0(m)n_1(k)^*]
=
N_0T
\frac{j}
{\pi(k-m+\frac12)}
$$

因此：

$$
E[\mathbf n_0\mathbf n_1^H]
=
N_0T\mathbf P
$$

最终：

$$
\mathbf R_n
=
N_0T
\begin{bmatrix}
I&P\\
P^H&I
\end{bmatrix}
$$

即：

$$
\mathbf R_n=N_0T\mathbf C
$$

------------------------------------------------------------------------

# 7. LMMSE 检测

由于接收基函数非正交，噪声为有色噪声。

广义 LMMSE：

$$
\hat{\mathbf a}
=
(
G^HR_n^{-1}G+R_a^{-1}
)^{-1}
G^HR_n^{-1}\mathbf y
$$

其中：

$$
G=\mu T\mathbf C\mathbf D
$$

最终需要求解：

$$
\begin{bmatrix}
H_0^HH_0+\lambda I
&
H_0^HPH_1
\\
H_1^HP^HH_0
&
H_1^HH_1+\lambda I
\end{bmatrix}
\begin{bmatrix}
\hat a_0\\
\hat a_1
\end{bmatrix}
=
\mathbf b
$$

其中：

-   对角块为对角矩阵；
-   复杂度主要集中在：

$$
H_0^HPH_1
$$

------------------------------------------------------------------------

# 8. 低复杂度检测方向

由于：

$$
P_{m,k}
=
\frac{j}
{\pi(k-m+\frac12)}
$$

仅与：

$$
k-m
$$

有关，因此：

$$
\mathbf P
$$

具有 Toeplitz 结构。

可进一步研究：

## (1) Toeplitz-FFT

利用 FFT 实现：

$$
\mathbf Pv
$$

复杂度：

$$
O(N^2)
\rightarrow
O(N\log N)
$$

## (2) Schur 补降维

利用块矩阵结构降低 MMSE 求解维度。

## (3) PCG 迭代求解

避免显式矩阵求逆。

目标：

$$
O(N^3)
\rightarrow
O(K_{iter}N\log N)
$$
