#!/usr/bin/env python
# coding: utf-8

# ## scikit-learn 公式データセット
# 
# ### sickit-learn 解説目次
# 
# 1. [公式データセット](/article/library/sklearn/datasets/) <= 本節
# 2. データの作成
# 3. 線形回帰
# 4. ロジスティック回帰
# 
# scikit-learnは機械学習に必要なデータセットを用意してくれています。ここでは公式サイトにそってサンプルデータの概要を説明します。
# 
# 1. toy dataset
# 2. 実際のデータセット
# 
# 
# 
# 詳細は公式ページを参考にしてください。
# 
# 筆者の環境は以下の通りです。

# In[3]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[1]:


import sklearn

sklearn.__version__


# In[ ]:





# In[1]:



import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

boston = load_boston()


# In[ ]:





# ## 参考資料
# - [scikit-learn 公式ページ](https://scikit-learn.org/stable/datasets/index.html)
# - いつも参考にしている[nkmkさん](https://note.nkmk.me/python-sklearn-datasets-load-fetch/)の記事です。
