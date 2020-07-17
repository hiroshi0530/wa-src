
# coding: utf-8

# ## scikit-learn でロジスティック回帰
# 
# scikit-learnを使えば手軽に線形回帰を実践できるので、備忘録として残しておきます。scikit-learnを用いれば、学習(fitting)や予測(predict)など手軽行うことが出来ます。ここでは2つの説明変数の場合の線形回帰をscikit-learnを用いて実行してみます。2変数なので重回帰といわれる回帰です。説明変数が一つの場合は単回帰といわれます。
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


# In[1]:


import numpy as np

from

