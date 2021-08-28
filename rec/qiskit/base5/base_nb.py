#!/usr/bin/env python
# coding: utf-8

# ## 量子フーリエ変換
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-algorithms/quantum-fourier-transform.html
# 
# 量子フーリエ変換になります。
# 量子フーリエ変換になります。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base5/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base5/base_nb.ipynb)
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


# In[ ]:





# In[ ]:





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

# 

# In[ ]:




