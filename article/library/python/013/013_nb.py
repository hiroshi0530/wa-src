#!/usr/bin/env python
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモを残しておきます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/013/013_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/013/013_nb.ipynb)
# 
# ### 筆者の環境

# In[20]:


get_ipython().system('sw_vers')


# In[21]:


get_ipython().system('python -V')


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import japanize_matplotlib


# ## NaNがある配列の平均や標準偏差をとる
# 
# nanが含まれる配列の平均を計算する際、`np.mean`を利用すると`np.nan`が返ってくる。
# これは不便なので、`np.nan`を除いた上で、平均や標準偏差を計算する事ができる`np.nanmean`や`np.nanstd`が便利。

# In[23]:


a = np.array([i for i in range(5)])
a = np.append(a, np.nan)
a = np.append(a, np.nan)
a = np.append(a, np.nan)
a


# In[24]:


b = np.array([i for i in range(5)])


# In[25]:


np.nanmean(a) == np.nanmean(b)
np.nanmean(a)


# 結果は同じ事が分かる。

# In[26]:


np.nanstd(a) == np.nanstd(b)
np.nanstd(a)


# 標準偏差も同様に結果は同じ事が分かる。

# ## 最大や最小
# 
# maxやminも同様に計算できる。

# In[29]:


a


# In[28]:


np.nanmax(a)


# In[30]:


np.nanmin(a)


# とても便利！
