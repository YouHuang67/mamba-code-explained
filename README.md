[English Version](README.en.md)

# Mamba CUDA 代码解析

## SSM基本形式

这篇文章将对《**Mamba: Linear-time sequence modeling with selective state spaces**》中Mamba模型[1]的CUDA代码进行分析和推导，尝试解释下Mamba为何计算高效。（ **虽然Mamba2已经中了ICML且通过高效的重构而让其实现主要依赖于Triton库（避免CUDA的优化问题），但最初的Mamba仍然具有很高的参考价值** ）

在了解代码之前，可以参考苏剑林博客[《重温SSM（一）：线性系统和HiPPO矩阵》](https://spaces.ac.cn/archives/10114)对Mamba理论的介绍，最终导出如下SSM（State Space Model）的基本形式：
$$
\begin{equation}
\begin{aligned}
	x'(t) &= Ax(t) + Bu(t) \\
	y(t) &= Cx(t) + Du(t)
\end{aligned}
\tag{1}
\end{equation}
$$

物理含义： $u(t) \in \mathbb{R}^{D}$ 是用于记录一个时间段内信息的函数， $t$ 为连续的时间变量，即在任意时刻 $t=t_0$ ， $u(t_0)$ 描述了该时刻的信号，基于此构建上述微分方程（1）的第一行：引入隐藏状态变量 $x(t) \in \mathbb{R}^{N}$ ，并通过预定义矩阵的 $A \in \mathbb{R}^{N\times N},B \in \mathbb{R}^{N \times D}$ 以及方程 $x'(t) = Ax(t) + Bu(t)$ 来建立 $x(t), u(t)$ 函数的联系（注意 $x^\prime(t)$ 是对 $t$ 求导）。公式（1）的第二行在确定 $x(t), u(t)$ 在相应时刻数值后由基本的线性计算给出 $y(t)$ ，因此不需要进一步讨论（**下文也只讨论对第一行的处理**）。总结：输入 $u(t)$ ，经过隐藏变量 $x(t)$ ，输出 $y(t)$ 。



## SSM离散化

上述SSM形式针对的是连续变量 $t$ ，但无法应用于实际计算，因此需要离散化，这部分参考 [《SSM离散化推导》](https://zhuanlan.zhihu.com/p/680534665)，并最终获得以下可实际计算的迭代形式：

$$
\begin{equation}
\begin{aligned}
	x_k&=\bar{A}x_{k-1}+\bar{B}u_k\\
	y_k&=Cx_k + Du_k \\
	\bar{A}&=e^{\Delta A}\\
	\bar{B}&=A^{-1}(e^{\Delta A}-I)B\\
\end{aligned}
\tag{2}
\end{equation}
$$

这里引入了时间步长 $\Delta = t_k - t_{k - 1} \in \mathbb{R}^{1}$ ，其中 $t_k, t_{k - 1}$ 为离散化时用于采样的时刻， $A \in \mathbb{R}^{N\times N},B \in \mathbb{R}^{N \times D}$ 同上， $I$ 为单位矩阵

直观理解： $u_k$ 可以对应于自然语言中的token，即给定一个长度为 $L$ 的token序列， $ \{ u_k \}_{k=1}^L$ ，SSM先通过公式（2）的迭代形式将这个序列映射为对应每个token的隐藏状态 $x_k$ ，后再线性映射为输出 $y_k$ 。（ **为便于后续分析，下文忽略 $y_k$ 的部分，其只是 $x_k, u_k$ 的简单线性组合** ）。



## Mamba SSM形式

公式（2）为参数固定的SSM，但对于Mamba则采用依赖于输入变化的参数，即有以下形式：

$$
\begin{equation}
\begin{aligned}
	x_k&=\bar{A_k} x_{k-1}+\bar{B_k}u_k\\
	\bar{A_k}&=e^{\Delta_k A}\\
	\bar{B_k}&=A^{-1}(e^{\Delta_k A}-I)B_k\\
\end{aligned}
\tag{3}
\end{equation}
$$

其中 $\Delta_k = \Delta_k(u_k) \in \mathbb{R}^{1}, B_k = B_k(u_k) \in \mathbb{R}^{N\times D}$ 根据输入 $u_k$ 决定，例如可以由简单线性映射 $\text{Linear}(u_k)$ 生成[1]。











$$
\begin{aligned}
	v_k&=a_kv_{k-1}+b_ku_{k}^{i}\\
	&=a_k(a_{k-1}v_{k-2}+b_{k-1}u_{k-1}^{i})+b_ku_{k}^{i}\\
	&=a_ka_{k-1}v_{k-2}+a_kb_{k-1}u_{k-1}^{i}+b_ku_{k}^{i}\\
	&\vdots\\
	&=a_ka_{k-1}\cdots a_1v_0+\sum_{j=1}^k{\left( \prod_{m=j+1}^k{a_m} \right)}b_ju_{j}^{i}\\
\end{aligned}
$$

## 参考文献

**[1] Gu, Albert, and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.**







