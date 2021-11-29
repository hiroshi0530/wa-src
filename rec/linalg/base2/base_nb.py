#!/usr/bin/env python
# coding: utf-8

# ## 疑似逆行列と射影行列
# 
# 疑似逆行列と射影行列もしばしば推薦システムでも利用されるので、メモしておきます。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/linalg/base2/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/linalg/base2/base_nb.ipynb)
# 
# ### 筆者の環境
# 
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy  version :', scipy.__version__)
print('numpy  version :', np.__version__)


# ## 疑似逆行列の計算
# 
# 通常の行列は常に逆行列を持つとは限りません。その場合、連立方程式、
# 
# $$
# Ax=b
# $$
# 
# をという事が出来ません。
# 
# そこで擬似的な逆行列を定義します。
# 
# $$
# A^{+}=\left(A^{T} A\right)^{-1} A^{T}
# $$
# 
# この$A^{+}$を左からかけて計算を進めると、
# 
# $$
# A^{+}A=\left(A^{T} A\right)^{-1} A^{T}A=I
# $$
# 
# となり、$x$を計算することが可能です。

# In[5]:


A = np.arange(25).reshape((5,5))
A


# この行列のランクを計算します。

# In[7]:


np.linalg.matrix_rank(A)


# 2なので、逆行列を計算することが出来ません。

# In[6]:


np.linalg.inv(A)


# エラーが出て終了となります。
# 
# 疑似逆行列はnumpyのpinvで計算することが出来ます。

# In[11]:


np.linalg.pinv(A).round(3)


# となり計算することが出来ます。
# 
# ## 射影行列
# 
# 射影行列は以下の様に定義されます。
# 
# $$
# P^{2}=P
# $$
# 
# ベクトル空間上である点を一度$P$で線形変換（回転や移動）させた後は、それ以上は移動させることが出来ないことを意味しています。
# 
# $$
# P^{N}=P
# $$
# 
# となるからです。
# 
# ### スペクトル分解
# 
# $A$をエルミート行列とし、固有値分解を行います。$u_i$を固有ベクトル、$\lambda_i$を固有値とします。
# 
# $$
# A=U \Lambda U^{T} \equiv\left[u_{1}, \ldots, u_{n}\right] \operatorname{diag}\left(\lambda_{1}, \ldots, \lambda{n} \right)\left[\begin{array}{c}
# u_{1}^{T} \\
# \vdots \\
# u_{n}^{T}
# \end{array}\right]
# $$
# 
# よって、
# 
# $$
# A = \sum_{i} \lambda_i u_i u_i^T
# $$
# 
# と分解することが可能です。
# 
# $U$はユニタリ行列で
# 
# $$
# U^TU=I
# $$
# 
# が成り立ちます。ここで、$u_i u_i^T$は、
# 
# $$
# (u_i u_i^T)^2 = u_i u_i^Tu_i u_i^T = u_i u_i^T
# $$
# 
# となり、固有空間への射影行列となります。よって、
# 
# $$
# A = \sum_{i} \lambda_i u_i u_i^T
# $$
# 
# は$A$という行列を固有空間への射影行列に固有値を重みとして和を取っている形に分解することが出来ます。
# これをスペクトル分解といい、推薦システムでも様々な場面で出てくる分解の形になっています。

# ## まとめ
# 
# 疑似逆行列と射影行列についてまとめてみました。まだまだ理解していない事が多いので、随時必要な時に追記していきます。
