#!/usr/bin/env python
# coding: utf-8

# ## cirq 入門
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


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


# In[7]:


from cirq import LineQubit, Circuit, Simulator, measure
import cirq


# In[8]:


# 量子回路初期化
qr = [LineQubit(i) for i in range(2)]
qc = Circuit()

# 量子回路
qc = qc.from_ops(
    # オラクル(|11>を反転)　
    cirq.H(qr[0]),
    cirq.H(qr[1]),
    cirq.CZ(qr[0],qr[1]),
    cirq.H(qr[0]),
    cirq.H(qr[1]),
    
    # 振幅増幅
    cirq.X(qr[0]),
    cirq.X(qr[1]),
    cirq.CZ(qr[0],qr[1]),
    cirq.X(qr[0]),
    cirq.X(qr[1]),
    cirq.H(qr[0]),
    cirq.H(qr[1]),   

    # 測定 
    cirq.measure(qr[0], key='m0'),
    cirq.measure(qr[1], key='m1'),
)


# $$
# \text{SWAP}=
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 1 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 \\
# \end{pmatrix}
# $$

# In[1]:


from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.qubit import Qubit,QubitBra
init_printing() # ベクトルや行列を綺麗に表示するため
psi = Qubit('0')
psi
represent(psi)


# <div>
# $$
# \text{SWAP}=
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 1 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 \\
# \end{pmatrix}
# $$
# </div>

# In[ ]:




