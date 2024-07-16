[English Version](README.en.md)

# Mamba CUDA 代码解析

## 概述

这篇文章将对**《Mamba: Linear-time sequence modeling with selective state spaces》**中Mamba模型[1]的CUDA代码进行分析和推导，尝试解释下Mamba为何计算高效。在了解代码之前，可以参考苏剑林博客[《重温SSM（一）：线性系统和HiPPO矩阵》](https://spaces.ac.cn/archives/10114)对Mamba理论的介绍，最终导出如下SSM（State Space Model）的基本形式：

$$
\begin{equation}
\begin{aligned}
	x'(t) &= Ax(t) + Bu(t) \\
	y(t) &= Cx(t) + Du(t)
\end{aligned}
\tag{1}
\end{equation}
$$

物理含义：$u(t)$是用于记录一个时间段内信息的函数，$t$为连续的时间变量，即在任意时刻$t=t_0$，$u(t_0)$描述了该时刻的信号，基于此构建上述微分方程（1）的第一行：引入隐藏状态变量$x(t)$，并通过$x'(t) = Ax(t) + Bu(t)$来建立$x, u$函数的联系（注意$x^\prime(t)$是对$t$求导）。公式（1）的第二行在确定$x(t), u(t)$在相应时刻数值后由基本的线性计算给出$y(t)$，因此不需要进一步讨论（**下文也只讨论对第一行的处理**）。




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

**[1] Gu, Albert, and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. **







