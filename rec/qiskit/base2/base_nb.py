#!/usr/bin/env python
# coding: utf-8

# ## 2量子ビット
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/preface.html 
# 
# 前回は基本的な使い方と1量子ビットのゲート演算が中心でしたが、今回は2量子ビットの演算を理解します。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base2/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base2/base_nb.ipynb)
# 
# ### 筆者の環境

# In[8]:


get_ipython().system('sw_vers')


# In[9]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[21]:


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


# In[135]:


import qiskit
import json

dict(qiskit.__qiskit_version__)


# $ |q\rangle = \cos{\tfrac{\theta}{2}}|0\rangle + e^{i\phi}\sin{\tfrac{\theta}{2}}|1\rangle $

# In[ ]:





# In[ ]:





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





# In[29]:


from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram


# In[30]:


from qiskit_textbook.widgets import binary_widget
binary_widget(nbits=5)


# In[31]:


n = 8
n_q = n
n_b = n
qc_output = QuantumCircuit(n_q,n_b)


# In[32]:


for j in range(n):
    qc_output.measure(j,j)


# In[33]:


qc_output.draw()


# In[34]:


counts = execute(qc_output,Aer.get_backend('qasm_simulator')).result().get_counts()
plot_histogram(counts)


# In[35]:


from qiskit_textbook.widgets import bloch_calc
bloch_calc()


# In[ ]:





# In[36]:


qc = QuantumCircuit(1)
qc.x(0)
qc.draw('mpl')


# In[37]:


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





# In[38]:


# このセルのコードを実行してウィジェットを表示します。
from qiskit_textbook.widgets import gate_demo
gate_demo(gates='pauli+h')


# In[93]:


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
qc.draw('mpl')


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

# In[40]:


from qiskit_textbook.tools import array_to_latex


# In[256]:


qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0,1)
display(qc.draw('mpl'))

# Let's see the result
statevector_backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,statevector_backend).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector} = ", precision=1)
plot_bloch_multivector(final_state)


# In[ ]:





# In[ ]:





# In[50]:


from qiskit import *
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram


# In[94]:


qc = QuantumCircuit(3)
# Apply H-gate to each qubit:
for qubit in range(3):
    qc.h(qubit)
# See the circuit:
qc.draw('mpl')


# In[52]:


# Let's see the result
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()

# In Jupyter Notebooks we can display this nicely using Latex.
# If not using Jupyter Notebooks you may need to remove the 
# array_to_latex function and use print(final_state) instead.
from qiskit_textbook.tools import array_to_latex
array_to_latex(final_state, pretext="\\text{Statevector} = ")


# In[53]:


qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
qc.draw()


# In[ ]:





# In[ ]:





# In[54]:


from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
sim = Aer.get_backend('aer_simulator')  # Jupyterノートブックの場合


# In[55]:


# |0> 量子ビットに対してゲート作用させてみましょう。
qc = QuantumCircuit(1)
qc.x(0)
qc.draw()


# In[67]:


qc = QuantumCircuit(3)
qc.h(0)
qc.x(1)
qc.x(2)
qc.draw('mpl')


# In[63]:


backend = Aer.get_backend('unitary_simulator')
unitary = execute(qc,backend).result().get_unitary()


# In[64]:


# In Jupyter Notebooks we can display this nicely using Latex.
# If not using Jupyter Notebooks you may need to remove the 
# array_to_latex function and use print(unitary) instead.
from qiskit_textbook.tools import array_to_latex
array_to_latex(unitary, pretext="\\text{Circuit = }\n")


# $$
# X \otimes H=\left[\begin{array}{ll}
# 0 & 1 \\
# 1 & 0
# \end{array}\right] \otimes \frac{1}{\sqrt{2}}\left[\begin{array}{cc}
# 1 & 1 \\
# 1 & -1
# \end{array}\right]=\frac{1}{\sqrt{2}}\left[\begin{array}{cc}
# 0 \times\left[\begin{array}{cc}
# 1 & 1 \\
# 1 & -1
# \end{array}\right] & 1 \times\left[\begin{array}{cc}
# 1 & 1 \\
# 1 & -1
# \end{array}\right] \\
# 1 \times\left[\begin{array}{cc}
# 1 & 1 \\
# 1 & -1
# \end{array}\right] & 0 \times\left[\begin{array}{cc}
# 1 & 1 \\
# 1 & -1
# \end{array}\right]
# \end{array}\right]=\frac{1}{\sqrt{2}}\left[\begin{array}{cccc}
# 0 & 0 & 1 & 1 \\
# 0 & 0 & 1 & -1 \\
# 1 & 1 & 0 & 0 \\
# 1 & -1 & 0 & 0
# \end{array}\right]
# $$

# In[ ]:




