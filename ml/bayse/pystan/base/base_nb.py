#!/usr/bin/env python
# coding: utf-8

# ## python + stan によるベイズモデリングの基礎
# 
# pythonとstan を用いて改めて1からベイズモデリングを勉強してみようと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_stock/lstm_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_stock/lstm_nb.ipynb)
# 
# ### 筆者の環境
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

import pystan

print('matplotlib version :', matplotlib.__version__)
print('scipy  version :', scipy.__version__)
print('numpy  version :', np.__version__)
print('pystan version :', pystan.__version__)


# In[ ]:


schools_dat = {
 'J': 8,
 'y': [28,  8, -3,  7, -1,  1, 18, 12],
 'sigma': [15, 10, 16, 11,  9, 11, 10, 18]
}

fit = pystan.stan(file='8schools.stan', data=schools_dat, iter=100, chains=4)
print(fit)


# In[ ]:




