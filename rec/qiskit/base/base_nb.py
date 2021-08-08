#!/usr/bin/env python
# coding: utf-8

# ## qiskit
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### 筆者の環境

# In[8]:


get_ipython().system('sw_vers')


# In[9]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[30]:


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


# In[31]:


import qiskit
qiskit.__qiskit_version__


# In[32]:


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram


# In[ ]:





# ## 単一の量子ビット
# 
# 単一の量子ビットは以下の様に表現できます。$\theta$、$\phi$は実数です。
# 
# $$
# |q\rangle=\cos \left(\frac{\theta}{2}\right)|0\rangle+e^{i \phi} \sin \left(\frac{\theta}{2}\right)|1\rangle
# $$
# 
# 
# 
# 
# ## 古典コンピュータのゲート
# 通常のロジック半導体は、
# 
# ![png](base_nb_files_local/logic_circuit.png)
# 
# $$
# \begin{array}{|c|c||c|}
# \hline A & B & Z \\
# \hline 0 & 0 & 0 \\
# \hline 0 & 1 & 0 \\
# \hline 1 & 0 & 0 \\
# \hline 1 & 1 & 1 \\
# \hline
# \end{array}
# $$
# 
# $$
# \begin{array}{|c|c||c|}
# \hline A & B & Z \\
# \hline 0 & 0 & 0 \\
# \hline 0 & 1 & 1 \\
# \hline 1 & 0 & 1 \\
# \hline 1 & 1 & 1 \\
# \hline
# \end{array}
# $$
# 
# $$
# \begin{array}{|c||c|}
# \hline A & Z \\
# \hline 0 & 1 \\
# \hline 1 & 0 \\
# \hline
# \end{array}
# $$
# 
# $$
# \begin{array}{|c|c||c|}
# \hline A & B & Z \\
# \hline 0 & 0 & 1 \\
# \hline 0 & 1 & 1 \\
# \hline 1 & 0 & 1 \\
# \hline 1 & 1 & 0 \\
# \hline
# \end{array}
# $$
# 
# $$
# \begin{array}{|c|c||c|}
# \hline A & B & Z \\
# \hline 0 & 0 & 1 \\
# \hline 0 & 1 & 0 \\
# \hline 1 & 0 & 0 \\
# \hline 1 & 1 & 0 \\
# \hline
# \end{array}
# $$
# 
# $$
# \begin{array}{|c|c||c|}
# \hline A & B & Z \\
# \hline 0 & 0 & 0 \\
# \hline 0 & 1 & 1 \\
# \hline 1 & 0 & 1 \\
# \hline 1 & 1 & 0 \\
# \hline
# \end{array}
# $$
# 
# ## パウリ行列
# 
# $$
# X=\left(\begin{array}{ll}
# 0 & 1 \\
# 1 & 0
# \end{array}\right)=|0\rangle\langle 1|+| 1\rangle\langle 0|
# $$

# $$
# X|0\rangle=\left(\begin{array}{ll}
# 0 & 1 \\
# 1 & 0
# \end{array}\right)\left(\begin{array}{l}
# 1 \\
# 0
# \end{array}\right)=\left(\begin{array}{l}
# 0 \\
# 1
# \end{array}\right)=|1\rangle
# $$
# 
# $$
# X|1\rangle=\left(\begin{array}{ll}
# 0 & 1 \\
# 1 & 0
# \end{array}\right)\left(\begin{array}{l}
# 0 \\
# 1
# \end{array}\right)=\left(\begin{array}{l}
# 1 \\
# 0
# \end{array}\right)=|0\rangle
# $$
# 

# In[ ]:





# $$
# |00\rangle=|0\rangle|0\rangle=|0\rangle \otimes|0\rangle=\left(\begin{array}{l}
# 1 \times\left(\begin{array}{l}
# 1 \\
# 0
# \end{array}\right) \\
# 0 \times\left(\begin{array}{l}
# 1 \\
# 0
# \end{array}\right)
# \end{array}\right)=\left(\begin{array}{l}
# 1 \\
# 0 \\
# 0 \\
# 0
# \end{array}\right)
# $$

# In[ ]:





# In[49]:


# |0> 量子ビットに対してゲート作用させてみましょう。
qc = QuantumCircuit(1)
qc.x(0)
qc.draw('mpl')


# In[50]:


# 結果を見てみましょう
backend = Aer.get_backend('statevector_simulator')
out = execute(qc,backend).result().get_statevector()
plot_bloch_multivector(out)


# In[ ]:





# In[ ]:





# In[ ]:



cr = ClassicalRegister(2)


# In[ ]:





# In[47]:


# 量子ビット、古典ビットの準備
qr = QuantumRegister(2)
cr = ClassicalRegister(2)

# 量子回路初期化
qc = QuantumCircuit(qr,cr)

# オラクル(|11>を反転)
qc.h(qr)
qc.cz(qr[0],qr[1])
qc.h(qr)

# 振幅増幅
qc.x(qr)
qc.cz(qr[0],qr[1])
qc.x(qr)
qc.h(qr)

# 測定
qc.measure(qr,cr)


# In[48]:


qc.draw(output='mpl')


# $ |q\rangle = \cos{\tfrac{\theta}{2}}|0\rangle + e^{i\phi}\sin{\tfrac{\theta}{2}}|1\rangle $

# $$
# |+++\rangle=\frac{1}{\sqrt{8}}\left[\begin{array}{l}
# 1 \\
# 1 \\
# 1 \\
# 1 \\
# 1 \\
# 1 \\
# 1 \\
# 1
# \end{array}\right]
# $$

# In[ ]:





# In[ ]:





# $$
# |b a\rangle=|b\rangle \otimes|a\rangle=\left[\begin{array}{l}
# b_{0} \times\left[\begin{array}{l}
# a_{0} \\
# a_{1}
# \end{array}\right] \\
# b_{1} \times\left[\begin{array}{l}
# a_{0} \\
# a_{1}
# \end{array}\right]
# \end{array}\right]=\left[\begin{array}{l}
# b_{0} a_{0} \\
# b_{0} a_{1} \\
# b_{1} a_{0} \\
# b_{1} a_{1}
# \end{array}\right]
# $$

# In[ ]:





# In[35]:


from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram


# In[36]:


from qiskit_textbook.widgets import binary_widget
binary_widget(nbits=5)


# In[37]:


n = 8
n_q = n
n_b = n
qc_output = QuantumCircuit(n_q,n_b)


# In[38]:


for j in range(n):
    qc_output.measure(j,j)


# In[39]:


qc_output.draw()


# In[40]:


counts = execute(qc_output,Aer.get_backend('qasm_simulator')).result().get_counts()
plot_histogram(counts)


# In[41]:


from qiskit_textbook.widgets import bloch_calc
bloch_calc()


# In[ ]:





# In[42]:


qc = QuantumCircuit(1)
qc.x(0)
qc.draw('mpl')


# In[43]:


from qiskit import *
from math import pi
from qiskit.visualization import plot_bloch_multivector

# 結果を見てみましょう
backend = Aer.get_backend('statevector_simulator')
out = execute(qc,backend).result().get_statevector()
plot_bloch_multivector(out)


# In[ ]:





# $|ba\rangle = |b\rangle \otimes |a\rangle = \begin{bmatrix} b_0 \times \begin{bmatrix} a_0 \\ a_1 \end{bmatrix} \\ b_1 \times \begin{bmatrix} a_0 \\ a_1 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} b_0 a_0 \\ b_0 a_1 \\ b_1 a_0 \\ b_1 a_1 \end{bmatrix}$

# In[ ]:





# In[26]:


# このセルのコードを実行してウィジェットを表示します。
from qiskit_textbook.widgets import gate_demo
gate_demo(gates='pauli+h')


# In[27]:


from qiskit.extensions import Initialize # Inititialize機能をインポートします。
# X測定関数を作成します。
def x_measurement(qc,qubit,cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)
    qc.h(qubit)
    return qc

# 量子ビットを初期化して測定します。
qc = QuantumCircuit(1,1)
initial_state = [0,1]
initialize_qubit = Initialize(initial_state)
qc.append(initialize_qubit, [0])
x_measurement(qc, 0, 0)
qc.draw()


# In[ ]:





# $$
# \|\mathbf{x}\|_{p}=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{\frac{1}{p}}
# $$

# $ X\otimes H = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \otimes \tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \frac{1}{\sqrt{2}}
# \begin{bmatrix} 0 \times \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
#               & 1 \times \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
#                 \\ 
#                 1 \times \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
#               & 0 \times \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
# \end{bmatrix} = \frac{1}{\sqrt{2}}
# \begin{bmatrix} 0 & 0 & 1 & 1 \\
#                 0 & 0 & 1 & -1 \\
#                 1 & 1 & 0 & 0 \\
#                 1 & -1 & 0 & 0 \\
# \end{bmatrix} $

# In[45]:


from qiskit_textbook.tools import array_to_latex


# In[46]:


qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0,1)
display(qc.draw())

# Let's see the result
statevector_backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,statevector_backend).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)


# In[ ]:





# $$ \int f(x) dx$$

# $$
# \lambda=\frac{(2 \pi) \times 1.97 \times 10^{3} \mathrm{eV} \cdot \angstrom}{\sqrt{2 \times 0.511 \times 10^{6} \mathrm{eV} \times V \cdot \mathrm{eV}}}=\frac{12.3}{\sqrt{V}} \AA
# $$

# $$
# \begin{aligned}
# \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n}-1} \exp \left(i \frac{2 \pi k j}{2^{n}}\right)|k\rangle &=\frac{1}{\sqrt{2^{n}}} \sum_{k_{0}=0,1} \sum_{k_{1}=0,1} \cdots \sum_{k_{n-1}=0,1} \exp \left(i \frac{2 \pi j}{2^{n}} \sum_{l=0}^{n-1} k_{l} 2^{l}\right)\left|k_{n-1} \cdots k_{0}\right\rangle \\
# &=\frac{1}{\sqrt{2^{n}}} \sum_{k_{0}=0,1} \sum_{k_{1}=0,1} \cdots \sum_{k_{n-1}=0,1} \prod_{l=0}^{n-1} \exp \left(i 2 \pi j k_{l} 2^{l-n}\right)\left|k_{l}\right\rangle \\
# &=\frac{1}{\sqrt{2^{n}}} \prod_{l=0}^{n-1}\left(\sum_{k_{l}=0,1} \exp \left(i 2 \pi j k_{l} 2^{l-n}\right)\left|k_{l}\right\rangle\right) \\
# &=\frac{1}{\sqrt{2^{n}}} \prod_{l=0}^{n-1}\left(|0\rangle+\exp \left(i 2 \pi j 2^{l-n}\right)|1\rangle\right) \\
# &=\frac{1}{\sqrt{2^{n}}} \prod_{l=0}^{n-1}\left(|0\rangle+\exp \left(i 2 \pi 0 . j_{n-1-l} \cdots j_{0}\right)|1\rangle\right) \\
# &=\frac{1}{\sqrt{2^{n}}}\left(|0\rangle+e^{i 2 \pi 0 . j_{0}}|1\rangle\right)\left(|0\rangle+e^{i 2 \pi 0 . j_{1} j_{0}}|1\rangle\right) \cdots\left(|0\rangle+e^{i 2 \pi 0 . j_{n-1} j_{n-2} \cdots j_{0}}|1\rangle\right)
# \end{aligned}
# $$

# $$
# |x\rangle=\sum_{i_{1} \ldots i_{k}=0}^{1} x_{i_{1}, \ldots, i_{n}}\left|i_{1}, \ldots, i,\right\rangle \in \underbrace{\mathbf{C}^{2} \otimes \cdots \otimes \mathbf{C}^{2}}_{, 1}
# $$

# $$
# \frac{\partial^{2} u}{\partial t^{2}}=c^{2}\left(\frac{\partial^{2} u}{\partial \xi^{2}}-2 \frac{\partial^{2} u}{\partial \xi \partial \eta}+\frac{\partial^{2} u}{\partial \eta^{2}}\right)
# $$

# In[ ]:




