#!/usr/bin/env python
# coding: utf-8

# ## [線型代数] 特異値分解と主成分分析
# 
# 主に推薦システムの理解に必要な線型代数の知識をまとめていこうと思います。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/linalg/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/linalg/base/base_nb.ipynb)
# 
# ### 筆者の環境
# 
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。

# In[2]:


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


# $$
# \begin{aligned}
# \mathbf{A} \mathbf{v} &=\sigma \mathbf{u} \\
# \mathbf{A}^{T} \mathbf{u} &=\sigma \mathbf{v}
# \end{aligned}
# $$

# $$
# \begin{aligned}
# &\mathbf{A}^{T} \mathbf{A} \mathbf{v}=\sigma \mathbf{A}^{T} \mathbf{u}=\sigma^{2} \mathbf{v} \\
# &\mathbf{A} \mathbf{A}^{T} \mathbf{u}=\sigma \mathbf{A} \mathbf{v}=\sigma^{2} \mathbf{u}
# \end{aligned}
# $$

# $u$と$v$は左特異ベクトル、右特異ベクトルと呼ばれ、$u$と$v$は$AA^{T}, A^{T}A$の固有ベクトル。

# 

# In[ ]:





# In[ ]:





# In[ ]:




