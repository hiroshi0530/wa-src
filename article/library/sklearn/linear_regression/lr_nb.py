
# coding: utf-8

# ## scikit-learn 公式データセット
# 
# scikit-learnは機械学習に必要なデータセットを用意してくれています。ここでは公式サイトにそってサンプルデータの概要を説明します。
# 
# ### sickit-learn 解説目次
# 
# 1. 公式データセット
# 2. データの作成
# 3. [線形回帰](/article/library/sklearn/linear_regression/) <= 本節
# 4. ロジスティック回帰
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sklearn/linear_regression/lr_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sklearn/linear_regression/lr_nb.ipynb)
# 
# ### abc
# 
# 1. toy dataset
# 2. 実際のデータセット
# 
# 詳細は公式ページを参考にしてください。
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[3]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[1]:


import sklearn

sklearn.__version__


# In[1]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

boston = load_boston()

