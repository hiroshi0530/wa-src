#!/usr/bin/env python
# coding: utf-8

# ## HHLアルゴリズム
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-applications/hhl_tutorial.html
# 
# 私の拙いブログでqiskitがRec（推薦システム）のカテゴライズしいたのは、すべてHHLを理解するためでした。現在、推薦システムに興味があり、開発などを行っていますが、そこで重要なのが連立一次方程式を解く事です。連立一次方程式は、数理モデルをコンピュータを利用して解く場合に高い確率で利用されますが、推薦システムもUser-Item行列から如何にしてユーザーエンゲージメントの高い特徴量を抽出出来るかという事が重要になってきます。
# 
# よって、量子コンピュータを利用して高速に連立一次方程式を解く事を目標に量子アルゴリズムの復習を開始したわけですが、ようやく目的までたどり着きました。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### 筆者の環境

# In[2]:


get_ipython().system('sw_vers')


# In[3]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import qiskit
import json

import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.visualization import plot_histogram

dict(qiskit.__qiskit_version__)


# ## 共役勾配法
# 
# 復習の意味を込めて、古典アルゴリズムである共役勾配法の復習をします。
# 正定値行列である$A$を係数とする連立一次方程式、
# 
# $$
# A \boldsymbol{x}=\boldsymbol{b}
# $$
# 
# の解$x$を反復法を用いて数値計算的に解く方法になります。反復法ですので、計算の終了を判定する誤差$(\epsilon)$が必要になります。

# $A$,$x$,$b$は以下の様な行列になります。
# 
# $$
# A = \left(\begin{array}{cccc}
# a_{11} & a_{12} & \cdots & a_{1 n} \\
# a_{21} & a_{22} & \cdots & a_{2 n} \\
# \vdots & \vdots & \ddots & \vdots \\
# a_{n 1} & a_{n 2} & \cdots & a_{n n}
# \end{array}\right),\quad x=\left(\begin{array}{c}
# x_{1} \\
# x_{2} \\
# \vdots \\
# x_{n}
# \end{array}\right), \quad b=\left(\begin{array}{c}
# b_{1} \\
# b_{2} \\
# \vdots \\
# b_{n}
# \end{array}\right)
# $$

# 行列の表式で書くと以下の通りです。
# 
# $$
# \left(\begin{array}{cccc}
# a_{11} & a_{12} & \cdots & a_{1 n} \\
# a_{21} & a_{22} & \cdots & a_{2 n} \\
# \vdots & \vdots & \ddots & \vdots \\
# a_{n 1} & a_{n 2} & \cdots & a_{n n}
# \end{array}\right)\left(\begin{array}{c}
# x_{1} \\
# x_{2} \\
# \vdots \\
# x_{n}
# \end{array}\right)=\left(\begin{array}{c}
# b_{1} \\
# b_{2} \\
# \vdots \\
# b_{n}
# \end{array}\right)
# $$

# 次に次のように定義される関数$f(x)$を考えます。
# 
# $$
# f(\boldsymbol{x})=\frac{1}{2}(\boldsymbol{x}, A \boldsymbol{x})-(\boldsymbol{b}, \boldsymbol{x})
# $$
# 
# $(-,-)$は、ベクトルの内積を計算する演算子です。
# 
# $$
# (\boldsymbol{x}, \boldsymbol{y})=\boldsymbol{x}^{T} \boldsymbol{y}=\sum_{i=1}^{n} \boldsymbol{x}_{i} \boldsymbol{y}_{i}
# $$
# 
# 成分で表示すると以下の様になります。
# 
# $$
# f(x)=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i j} x_{i} x_{j}-\sum_{i=1}^{n} b_{i} x_{i}
# $$

# ここで、　$x_k$で微分すると、
# 
# $$
# \frac{\partial f(x)}{\partial x_{k}}=\frac{1}{2} \sum_{i=1}^{n} a_{i k} x_{i}+\frac{1}{2} \sum_{j=1}^{n} a_{k j} x_{j}-b_{k}
# $$
# 
# となります。

# $A$はエルミート行列なので、
# 
# $$
# \frac{\partial f(x)}{\partial x_{i}}=\sum_{j=1}^{n} a_{i j} x_{j}-b_{i}=0
# $$
# 
# となります。

# これを一般化すると、
# 
# $$
# \nabla f(x)=\left(\begin{array}{c}
# \frac{\partial f}{\partial x_{1}} \\
# \vdots \\
# \frac{\partial f}{\partial x_{n}}
# \end{array}\right)=A\boldsymbol{x}-b = 0
# $$
# 
# となり、関数$f(x)$の最小値となる$x$を求める事が、$A\boldsymbol{x}-b = 0$を解く事と同じである事が分かります。

# ### アルゴリズム
# 
# 上記の通り、共役勾配法(CG法)は、関数$f(x)$を最小化することに帰着されます。
# そのために、ある$x^{(0)}$を出発点に、以下の漸化式に従って最小値とする$x$を求めます。
# 
# $$
# x^{(k+1)}=x^{(k)}+\alpha_{k} p^{(k)}
# $$
# 
# ここで、$p^{k}$は解を探索する方向ベクトルです。
# 
# 
# $$
# f\left(x^{(k)}+\alpha_{k} p^{(k)}\right)=\frac{1}{2} \alpha_{k}{ }^{2}\left(p^{(k)}, A p^{(k)}\right)-\alpha_{k}\left(p^{(k)}, b-A x^{(k)}\right)+f\left(x^{(k)}\right)
# $$

# $$
# \frac{\partial f\left(x^{(k)}+\alpha_{k} p^{(k)}\right)}{\partial \alpha_{k}}=0 \Rightarrow \alpha_{k}=\frac{\left(p^{(k)}, b-A x^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}=\frac{\left(p^{(k)}, r^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
# $$

# $$
# \begin{aligned}
# &r^{(k+1)}=b-A x^{(k+1)} \\
# &r^{(k)}=b-A x^{(k)} \\
# &r^{(k+1)}-r^{(k)}=A x^{(k+1)}-A x^{(k)}=\alpha_{k} A p^{(k)}
# \end{aligned}
# $$

# $$
# \alpha_{k}=\frac{\left(p^{(k)}, r^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
# $$

# $$
# \left\|\boldsymbol{r}^{(k+1)}\right\|<\varepsilon
# $$

# $$
# \boldsymbol{p}^{(k+1)}=\boldsymbol{r}^{(k+1)}+\beta^{(k)} \boldsymbol{p}^{(k)}
# $$

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# $$
# p^{k}=-\nabla f\left(x^{k}\right)=\left(\frac{\partial f\left(x^{k}\right)}{\partial x^{1}}, \ldots, \frac{\partial f\left(x^{k}\right)}{\partial x^{n}}\right)^{T}
# $$

# 

# ### 計算量
# $s$は行列$A$の非0要素の割合、$\kappa$は行列$A$の最大固有値と最小固有値の比$\displaystyle \left| \frac{\lambda_{max}}{\lambda_{min}}\right|$、$\epsilon$  は精度です．
# 
# $$
# O(N s \kappa \log (1 / \varepsilon))
# $$
# 
# これは$N$に比例する形になっているため、$s\sim \log N$の場合、$N\log N$に比例する事になります。

# ## HHLアルゴリズム
# 
# ### HHLの仮定
# 
# HHLはある程度の仮定の下に成立するアルゴリズムです。
# 
# - ローディングを実施する効果的なオラクルが存在
# - ハミルトニアンシミュレーションと解の関数の計算が可能
# - $A$がエルミート行列
# - $\mathcal{O}\left(\log (N) s^{2} \kappa^{2} / \epsilon\right)$で計算可能
# - 古典アルゴリズムが完全解を返すが、HHLは解となるベクトルを与える関数を近似するだけ

# ### 量子回路へのマッピング
# 
# 連立一次方程式を量子アルゴリズムで解くには、$Ax=b$を量子回路にマッピングする必要があります。それは、$b$の$i$番目の成分は量子状態$|b\rangle$の$i$番目の基底状態の振幅に対応させるという方法です。また、当然ですが、その際は$\displaystyle \sum_i |b_i|^2 = 1$という規格化が必要です。
# 
# $$
# Ax=b \rightarrow A|x\rangle = |b\rangle
# $$
# 
# となります。

# ### スペクトル分解
# 
# $A$はエルミート行列なので、スペクトル分解が可能です。$A$のエルミート性を暗幕的に仮定していましたが、
# $$
# A'=\left(\begin{array}{ll}
# 0 & A \\
# A & 0
# \end{array}\right)
# $$
# 
# とすれば、$A'$はエルミート行列となるため、問題ありません。よって、$A$は固有ベクトル$|u_i\rangle$とその固有値$\lambda_i$を利用して、以下の様に展開できます。
# 
# $$
# A=\sum_{j=0}^{N-1} \lambda_{j}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# $$
# 
# よって、逆行列は以下の様になります。
# 
# $$
# A^{-1}=\sum_{j=0}^{N-1} \lambda_{j}^{-1}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# $$

# $u_i$は$A$の固有ベクトルなので、$|b\rangle$はその重ね合わせで表現できます。おそらくこれが量子コンピュータを利用する強い動機になっていると思います。
# 
# $$
# |b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
# $$
# 
# 本来であれば、$A$の固有ベクトルを計算できなければこの形で$|b\rangle$を用意することが出来ませんが、量子コンピュータでは$|b\rangle$を読み込むことで、自動的にこの状態を用意することが出来ます。
# 
# 最終的には以下の式の形を量子コンピュータを利用して求める事になります。
# 
# $$
# |x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1} \lambda_{j}^{-1} b_{j}\left|u_{j}\right\rangle
# $$

# ### 1. データのロード
# 
# 対象となる$|b\rangle$の各データの振幅を量子ビット$n_b$にロードします。
# 
# $$
# |0\rangle_{n_{b}} \mapsto|b\rangle_{n_{b}}
# $$

# ### 2. QPEの適用
# 
# 量子位相推定を利用して、ユニタリ演算子$U=e^{i A t}$ の$|b\rangle$の位相を推定します。
# 
# $|b\rangle$は上述の様に以下になります。
# 
# $$
# |b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
# $$

# $U$は展開して整理すると、以下の様になります。
# 
# $$
# \begin{aligned}
# &U=e^{i A t}=\sum_{k=0}^{\infty} \frac{\left(i A t\right)^{k}}{k !}\\
# &=\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k !}\left(\sum_{j=0}^{N-1} \lambda_{j}|u_{j}\rangle\langle u_{j} |\right)^{k}\\
# &=\sum_{k=0}^{\infty} \frac{(i t)^{k}}{k !} \sum_{j=0}^{N-1} \lambda_{j}^{k}\left|u_{j}\right\rangle \langle u_{j} |\\
# &=\sum_{j=0}^{N-1}\left(\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k_{i}} \lambda_{j}^{k}\right)|u_{j}\rangle \langle u_{j}| \\
# &=\sum_{r=0}^{N-1} e^{i \lambda_{j} t}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# \end{aligned}
# $$

# $U$を$|b\rangle$に作用させると、
# 
# $$
# \begin{aligned}
# U|b\rangle &=U\left(\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle\right) \\
# &=\sum_{j^{\prime}=0}^{N-1} e^{i \lambda j^{\prime} t}\left|h_{j^{\prime}}\right\rangle\left\langle h_{j^{\prime}}\right| \cdot\left(\sum_{j=0}^{N-1} b_{j}\left|h_{j}\right\rangle\right) \\
# &=\sum_{j=0}^{N-1} b_{j} e^{i \lambda_{j} t}\left|u_{j}\right\rangle
# \end{aligned}
# $$
# 
# となり、量子位相推定を利用して、$\lambda_j$の量子状態$|\tilde{\lambda}_j\rangle_{n_l}$を求める事が出来ます。
# 
# $\tilde{\lambda_{j}}$は、$\displaystyle 2^{n_{l}} \frac{\lambda_{j} t}{2 \pi}$に対する$n_l$-bitバイナリ近似となります。

# $t=2\pi$とし、$\lambda_l$が、$n_l$ビットで正確に表現できるとすると、量子位相推定は以下の様に表現できます。
# 
# $$
# \operatorname{QPE}\left(e^{i A 2 \pi}, \sum_{j=0}^{N-1} b_{j}|0\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}\right)=\sum_{j=0}^{N-1} b_{j}\left|\lambda_{j}\right\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}
# $$

# In[ ]:





# $b(\cdots)$は二進数表記である事を示し、$b_k(\cdots)$は、二進数表記の$k$ビット目の値を表します。まとめると、以下の様になります。
# 
# $$
# \begin{aligned}
# &b\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_{0} d_{1} d_2 \cdot \cdots d_{m-1} \\
# &b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_k \\
# &\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)=\frac{d_{0}}{2}+\frac{d_{1}}{4}+\frac{d_{2}}{8} \cdots \frac{d_{m-1}}{2^{m}} \quad \left(0 \leqq \frac{2}{\pi} \cos ^{-1}\left(v_{i}\right) \leqq 1\right)
# \end{aligned}
# $$
# 
# これから、
# 
# 
# $$
# \begin{aligned}
# \frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)&=\frac{1}{2} b_{0}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{1}\right)\right)+\frac{1}{4} b_{1}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)+ \cdots \\
# &=\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{n}\right)\right) 2^{-k-1} \cdots (1)
# \end{aligned}
# $$

# $\displaystyle b_{k}\left(\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\right)$をターゲットビットとして、$R_{y}\left(2^{-k-1}\pi\right)$である制御回転ゲートをかけることを考えます。
# 
# $$
# \begin{aligned}
# &\prod_{k=0}^{m-1} R_{y}\left(b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
# &=R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
# \end{aligned}
# $$

# 回転ゲートは、以下の様にになります。
# 
# $$
# R_{y}(\theta)|0\rangle=\cos \frac{\theta}{2}|0\rangle+\sin \frac{\theta}{2}|1\rangle
# $$

# この式と(1)を利用して、
# 
# $$
# \begin{aligned}
# &\cos\frac{1}{2}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right) \\
# &=\cos\left(\frac{1}{2}\times\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\times \pi\right) = v_i
# \end{aligned}
# $$

# となり、こちらを利用して、
# 
# $$
# R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle=v_{i}|0\rangle+\sqrt{1-v_{i}^{2}}|1\rangle
# $$
# 
# を得る。ここで、$\displaystyle v_i = \frac{1}{\lambda_i}$とすることで、補助ビットを、
# 
# $$
# \sqrt{1-\frac{C^{2}}{\lambda_{j}^{2}}}|0\rangle+\frac{C}{\lambda_{j}}|1\rangle
# $$
# 
# と計算することが出来る。

# 

# In[ ]:





# $$
# \operatorname{QPE}\left(U,|0\rangle_{n}|\psi\rangle_{m}\right)=|\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
# $$

# In[ ]:





# ## 計算量の比較
# 
# ## 量子
# 
# 
# $$
# O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon)))
# $$
# 
# 行列 $A$ がスパース $(s \sim O(\operatorname{poly} \log N))$なら、
# 
# $$
# O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon))) \sim O(s \kappa \operatorname{poly} \log N \operatorname{poly} \log (s \kappa / \varepsilon))
# $$
# 
# 
# ### 古典
# $$
# O(N s \kappa \log (1 / \varepsilon))
# $$

# ## qiskitで実装
# 

# In[84]:


from qiskit import Aer
from qiskit.circuit.library import QFT
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.initial_states import Custom
import numpy as np


# In[85]:


def create_eigs(matrix, num_ancillae, num_time_slices, negative_evals):
    ne_qfts = [None, None]
    if negative_evals:
        num_ancillae += 1
        ne_qfts = [QFT(num_ancillae - 1), QFT(num_ancillae - 1).inverse()]

    return EigsQPE(MatrixOperator(matrix=matrix),
                   QFT(num_ancillae).inverse(),
                   num_time_slices=num_time_slices,
                   num_ancillae=num_ancillae,
                   expansion_mode='suzuki',
                   expansion_order=2,
                   evo_time=None, # np.pi*3/4, #None,  # This is t, can set to: np.pi*3/4
                   negative_evals=negative_evals,
                   ne_qfts=ne_qfts)


# In[86]:


def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("Fidelity:\t\t %f" % fidelity)


# In[87]:


matrix = [[1, -1/3], [-1/3, 1]]
vector = [1, 0]


# In[95]:


orig_size = len(vector)
matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

# Initialize eigenvalue finding module
eigs = create_eigs(matrix, 3, 100, False)
num_q, num_a = eigs.get_register_sizes()

# Initialize initial state module
init_state = Custom(num_q, state_vector=vector)
# init_state = QuantumCircuit(num_q).initialize(vector/np.linalg.norm(vector))

# Initialize reciprocal rotation module
reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
           init_state, reci, num_q, num_a, orig_size)


# In[96]:


result = algo.run(QuantumInstance(Aer.get_backend('statevector_simulator')))
print("Solution:\t\t", np.round(result['solution'], 5))

result_ref = NumPyLSsolver(matrix, vector).run()
print("Classical Solution:\t", np.round(result_ref['solution'], 5))

print("Probability:\t\t %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])


# In[92]:


print("circuit_width:\t", result['circuit_info']['width'])
print("circuit_depth:\t", result['circuit_info']['depth'])
print("CNOT gates:\t", result['circuit_info']['operations']['cx'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


from qiskit import aqua
aqua.__version__


# In[ ]:





# In[65]:


get_ipython().system('pip install --upgrade qiskit')


# In[13]:


get_ipython().system('pip install qiskit-terra')


# In[ ]:





# In[42]:


get_ipython().run_line_magic('pinfo', 'Custom')

init signature:
Custom(
    num_qubits: int,
    state: str = 'zero',
    state_vector: Union[numpy.ndarray, qiskit.aqua.operators.state_fns.state_fn.StateFn, NoneType] = None,
    circuit: Union[qiskit.circuit.quantumcircuit.QuantumCircuit, NoneType] = None,
) -> None


# In[48]:


get_ipython().run_line_magic('pinfo', 'QuantumCircuit.initialize')

Args:
    params (str or list or int):
        * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
            :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.
            Notice the order of the labels is reversed with respect to the qubit index to
            be applied to. Example label '01' initializes the qubit zero to `|1>` and the
            qubit one to `|0>`.
        * list: vector of complex amplitudes to initialize to.
        * int: an integer that is used as a bitmap indicating which qubits to initialize
           to `|1>`. Example: setting params to 5 would initialize qubit 0 and qubit 2
           to `|1>` and qubit 1 to `|0>`.
    qubits (QuantumRegister or int):
        * QuantumRegister: A list of qubits to be initialized [Default: None].
        * int: Index of qubit to initialized [Default: None].

Returns:
    qiskit.circuit.Instruction: a handle to the instruction that was just initialized


# In[ ]:





# In[ ]:





# In[103]:


import math

qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
# Add Controlled-T
qc.cp(math.pi/4, 0, 1)
display(qc.draw())


# In[104]:


from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

qc.save_unitary()
usim = Aer.get_backend('aer_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, prefix="\\text{Circuit = }\n")


# In[ ]:





# In[ ]:





# In[ ]:





# ## 参考文献
# - https://www2.yukawa.kyoto-u.ac.  jp/~qischool2019/mitaraiCTO.pdf
