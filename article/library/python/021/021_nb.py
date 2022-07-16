#!/usr/bin/env python
# coding: utf-8

# ## pandasとnumpyの標準偏差の計算の定義
# 
# pandasとnumpyで標準偏差を使った式を計算したときに、微妙に結果が違って調べたのでメモ。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/021/021_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/021/021_nb.ipynb)
# 
# ### 実行環境

# In[17]:


get_ipython().system('sw_vers')


# In[18]:


get_ipython().system('python -V')


# ### 何も考えず実行

# In[1]:


import pandas as pd
import numpy as np

pd.Series([i for i in range(5)]).std()


# In[2]:


np.std([i for i in range(5)])


# 両者の結果が異なる。
# ドキュメントを確認すると、numpyがデフォルトで自由度$n$、pandasが自由度$n-1$で計算しているからである。
# 
# - 自由度 $n$
# 
# $$
# s=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}
# $$
# 
# 
# - 自由度 $n - 1$
# 
# $$
# s=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}
# $$

# 引数で自由度を明示すれば一致する。

# In[15]:


print(pd.Series([i for i in range(5)]).std(ddof=0))
print(np.std([i for i in range(5)], ddof=0))


# In[16]:


print(pd.Series([i for i in range(5)]).std(ddof=1))
print(np.std([i for i in range(5)], ddof=1))


# 定義通り計算すると、以下の通りで結果は一致する。

# In[13]:


np.sqrt(np.sum(np.array([i * i for i in range(5)]) - np.power(np.mean([i for i in range(5)]), 2)) / 5)


# In[12]:


np.sqrt(np.sum(np.array([i * i for i in range(5)]) - np.power(np.mean([i for i in range(5)]), 2)) / 4)


# データ量が多くなればほとんど差はなくなるが、値が微妙に違う結果になって調べたのでメモ。
