#!/usr/bin/env python
# coding: utf-8

# ## サイモンのアルゴリズム
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-algorithms/simon.html
# 
# 今回は、サイモンのアルゴリズムを数式を追って理解を深めようと思います。
# 
# ドイチェ-ジョサの問題設定は、関数$f(x)$が、定数型か分布型のどちらか判別するいう事でしたが、サイモンのアルゴリズムの問題設定は、1:1関数か、2:1の関数かのどちらかを判定するという違いです。その違いを判別し、さらに、2:1の関数の周期を求める事になります。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)
# 
# ### 筆者の環境

# In[36]:


get_ipython().system('sw_vers')


# In[37]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)


# In[4]:


import qiskit
import json

dict(qiskit.__qiskit_version__)


# In[5]:


from qiskit import IBMQ, Aer, execute
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import array_to_latex


# ## 問題設定
# 
# 問題としては、関数$f(x)$が、1：1の関数なのか、2：1の関数なのかを判定する事です。1：1の関数とは、$y=x$のような、単純な全単射関数を考えれば良いと思います。
# 
# $$
# \begin{aligned}
# &|00\rangle \stackrel{f}{\longrightarrow}| 00\rangle \\
# &|01\rangle \stackrel{f}{\longrightarrow}| 01\rangle \\
# &|10\rangle \stackrel{f}{\longrightarrow}| 10\rangle \\
# &|11\rangle \stackrel{f}{\longrightarrow}| 11\rangle 
# \end{aligned}
# $$
# 
# 2：1の関数というのは、以下の様に、NビットからN-1ビットへの関数になります。二つの入力値が一つの出力値に相当していて、2：1なので、ビット数が1つ減少することになります。
# 
# $$
# f:\lbrace 0,1 \rbrace^{n} \rightarrow \lbrace 0,1 \rbrace^{n-1}
# $$
# $$
# x \in\{0,1\}^{n}
# $$
# 
# 2ビットでの具体的例は以下の通りです。
# 
# $$
# \begin{aligned}
# &|00>\stackrel{f}{\longrightarrow}| 0\rangle \\
# &|01>\stackrel{f}{\longrightarrow}| 1\rangle \\
# &|10>\stackrel{f}{\longrightarrow}| 1\rangle \\
# &|11>\stackrel{f}{\longrightarrow}| 0\rangle 
# \end{aligned}
# $$
# 
# ![svg](base_nb_files_local/qiskit-2_1.svg)
# 
# 2：1の関数なので、あるNビット配列$a (a\ne |00\cdots\rangle)$が存在して、
# 
# $$
# f(x \oplus a)=f(x)
# $$
# 
# が成立します。

# どちらの関数を見極めるには、最大で、$2^{n-1}+1$回の関数の実行が必要です。運良く、異なる入力に対して、同じ出力が2回連続で出た場合は、2対1型の関数だと分かります。
# 
# 古典コンピューター上で、回数の下限が$\Omega\left(2^{n / 2}\right)$となるアルゴリズムが知られているようですが、それでも$n$に対して指数関数的に増加します。

# ## 1. 二つのn量子ビットの入力レジスタンスを0に初期化
# 
# $$
# \left|\psi_{1}\right\rangle=|0\rangle^{\otimes n}|0\rangle^{\otimes n}
# $$
# 
# ## 2. 一つ目もレジスタにアダマールゲートを適用
# 
# $$
# \left|\psi_{2}\right\rangle=\frac{1}{\sqrt{2^{n}}} \sum_{x \in\{0,1\}^{n}}|x\rangle|0\rangle^{\otimes n}
# $$
# 
# $|0\rangle^{\otimes n} $の量子ビットへのアダマールゲートの適用は以下の様になりますが、上記と同等です。
# 
# $$
# |0\rangle^{\otimes n} \longmapsto \frac{1}{\sqrt{2}^{n}} \sum_{k=0}^{2^n-1}|k\rangle
# $$
# 
# ## 3. オラクルへの問い合わせ関数を実行
# 
# $$
# \left|\psi_{3}\right\rangle=\frac{1}{\sqrt{2^{n}}} \sum_{x \in\{0,1\}^{n}}|x\rangle|f(x)\rangle
# $$
# 
# 量子オラクルは、二つ目のレジスタに関数$f(x)$の結果を格納します。上記のオラクルは以下のオラクルと同等です。
# 
# $$
# \begin{aligned}
# \frac{1}{\sqrt{2}^{n}} \sum_{k=0}^{2^n-1}|k\rangle \otimes|0\rangle \longmapsto \frac{1}{\sqrt{2^{n}}} \displaystyle \sum_{k=0}^{2^n-1}|k\rangle \otimes|f(k)\rangle 
# \end{aligned}
# $$
# 
# ## 4. 二つ目の量子レジスタを測定
# 
# 問題の設定から、関数$f(x)$の入力$x$は二つの量子ビットに対応づけることが出来ます。
# 
# ある$x$と、その$x$と量子ビット$b$のXOR、$y=x \oplus b$になります。$b=|00\cdots 0\rangle$であれば、1：1の関数で、0以外であれば、2：1の関数になります。この$x$と$y$を利用する事で、一つ目のレジスタの値は以下の様になります。
# 
# $$
# \left|\psi_{4}\right\rangle=\frac{1}{\sqrt{2}}(|x\rangle+|y\rangle)
# $$

# ### 対応表
# 
# $x$と$y=x \oplus b$の対応表は以下の様になります。$f(x)$は2：1の関数型となります。
# 
# - $b=|11\rangle$で$n=2$の場合
# 
# $$
# \begin{array}{|r|r|r|r|}
# \hline \mathrm{x} & \mathrm{f}(\mathrm{x}) & \mathrm{y}(=\mathrm{x} \oplus \mathrm{b}) & \mathrm{x} \cdot \mathrm{b} (\text{mod 2})\\
# \hline 00 & 00 & 11 & 0 \\
# 01 & 10 & 10 & 1 \\
# 10 & 10 & 01 & 1 \\
# 11 & 00 & 00 & 0 \\
# \hline
# \end{array}
# $$

# - $b=|110\rangle$で$n=3$の場合
# 
# $$
# \begin{array}{|r|r|r|r|}
# \hline \mathrm{x} & \mathrm{f}(\mathrm{x}) & \mathrm{y}(=\mathrm{x} \oplus \mathrm{b}) & \mathrm{x} \cdot \mathrm{b}(\text{mod 2}) \\
# \hline 000 & 000 & 110 & 0 \\
# 001 & 001 & 111 & 0 \\
# 010 & 100 & 100 & 1 \\
# 011 & 101 & 101 & 1 \\
# 100 & 100 & 010 & 1 \\
# 101 & 101 & 011 & 1 \\
# 110 & 000 & 000 & 0 \\
# 111 & 001 & 001 & 0 \\
# \hline
# \end{array}
# $$
# 

# ## 5. 一つ目のレジスタにアダマールゲートを適用
# 
# 4. で得られた$|\psi_4\rangle$にアダマールゲートを適用すると以下の様になります。
# 
# $$
# \left|\psi_{5}\right\rangle=\frac{1}{\sqrt{2^{n+1}}} \sum_{z \in\{0,1\}^{n}}\left[(-1)^{x \cdot z}+(-1)^{y-z}\right]|z\rangle
# $$
# 
# ### アダマールゲートに関する公式
# 
# STEP5でさらっとアダマール変換後の式が出てきたので、復習を込めてなぜそうなるのか計算してみます。
# 
# $$
# \left|\psi_{5}\right\rangle=\frac{1}{\sqrt{2^{n+1}}} \sum_{z \in\lbrace0,1\rbrace^{n}}\left[(-1)^{x \cdot z}+(-1)^{y \cdot z}\right]|z\rangle
# $$
# 
# $|b\rangle$を二進数表記で以下の様に表現します。$b_{k} \in\lbrace0,1\rbrace^{n}$です。
# 
# $$
# |b\rangle=\left|b_{n} b_{n-1} \cdots b_{1} b_{0}\right\rangle
# $$
# 
# この$|b\rangle$に対して、アダマールゲートを適用します。

# $$
# \begin{aligned}
# H^{\otimes n}|b\rangle&=H^{\otimes n}\left|b_{n} b_{n-1} \cdots b_1 \right\rangle \\
# &=\frac{1}{\sqrt{2^{n}}}\left(|0\rangle+(-1)^{b_n}|1\rangle\right)\otimes\left(|0\rangle+(-1)^{b_{n-1}}|1\rangle\right)\otimes \cdots \\
# &=\frac{1}{\sqrt{2^{n}}}(\mid 00 \ldots 0\rangle+(-1)^{b_{1}}|00 \cdots 01\rangle \\
# &\qquad \left.+(-1)^{b_{1}}|00 \cdots 01 0\right\rangle \left.+(-1)^{b_{2}+b_{1}}|00 \cdots 011\right\rangle \\
# &\qquad \qquad \cdots \left.+(-1)^{b_{n}+b_{n-1}+\cdots+b_{2}+b_{1}}|11 \cdots 1)\right\rangle \\
# &=\frac{1}{\sqrt{2^{n}}} \sum_{z \in\lbrace 0,1 \rbrace^n }(-1)^{b_{n} z_{n}+b_{n-1} z_{n-1}+\cdots+b_{1} z_{1}}|z\rangle \\
# &=\frac{1}{\sqrt{2^{n}}} \sum_{z \in\lbrace0,1\rbrace^n}(-1)^{b \cdot z}|z\rangle
# \end{aligned}
# $$
# 
# この変化はqiskitの説明でもよく出てきますが、式を追いかけていく上で、慣れないと躓いてしまいます。

# ## 6. 測定
# 
# 上記の$|\psi_5\rangle$を測定すると、$(-1)^{x \cdot z}=(-1)^{y \cdot z}$を満たす$z$のみが測定されます。それ以外の要素はすべて0です。
# 
# $$
# \begin{aligned}
# & x \cdot z=y \cdot z \\
# & x \cdot z=(x \oplus b) \cdot z \\
# & x \cdot z=x \cdot z \oplus b \cdot z \\
# & b \cdot z=0(\bmod 2)
# \end{aligned}
# $$
# 
# よって$b$との内積が0となる$z$が測定されます。測定を複数回行うことで、以下の様な連立一次方程式を得ることができ、これを古典コンピューターを利用して解く事で量子ビット$b$を得ることが出来ます。

# $$
# \begin{array}{c}
# b \cdot z_{1}=0 \\
# b \cdot z_{2}=0 \\
# \vdots \\
# b \cdot z_{n}=0
# \end{array}
# $$

# この連立一次方程式を解けば、$b$が特定され、もし、$b=|00 \cdots 0\rangle$であれば、1：1の関数であり、そうでなければ、2：1の関数という事になります。
# 
# 本来ならば、指数関数的な計算量が必要なのですが、おおよそ$n$回の測定と連立一次方程式を解く事によって問題の答えを見つけることが出来ます。

# ## 位相キックバックの復習
# 
# 位相キックバックのメモ。ターゲットビットの作用させるユニタリ行列を$U$、その固有値を$e^{i \phi}$とすると、コントールビットの$|1\rangle$の係数にその固有値が出現します。ターゲットビットにではなく、制御ビットの方にキックバックされると事だと思います。

# $$
# \begin{aligned}
# &\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle) \otimes|\psi\rangle  \\
# & \longrightarrow \frac{1}{\sqrt{2}}(|0\rangle \otimes|\psi\rangle+|1\rangle \otimes U|\psi\rangle) \\
# &=\frac{1}{\sqrt{2}}\left(|0\rangle \otimes|\psi\rangle+e^{i \phi}|1\rangle \otimes|\psi\rangle\right) \\
# &=\frac{1}{\sqrt{2}}\left(|0\rangle+e^{i \phi}|1\rangle\right) \otimes|\psi\rangle
# \end{aligned}
# $$
# 
# ###  qiskitでのユニタリ演算子をCNOTゲートにした例

# In[9]:


qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.draw('mpl')


# In[10]:


backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")


# ## CNOTゲートとXOR
# 
# こちらもメモ程度ですが、CNOTゲートをはターゲットビットを制御ビットとターゲットビットのXORに置き換える事に相当します。忘れないようにしないと。
# 
# $$
# |i j\rangle \stackrel{\mathrm{CX}}{\longrightarrow}|i(i \mathrm{XOR} j)\rangle
# $$
