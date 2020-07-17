#!/usr/bin/env python
# coding: utf-8

# ## scikit-learn でロジスティック回帰
# 
# scikit-learnを使えば手軽にロジスティック回帰を実践できるので、備忘録として残しておきます。scikit-learnを用いれば、学習(fitting)や予測predict)など手軽行うことが出来ます。ロジスティック回帰は回帰となっていますが、おそらく分類問題を解く手法だと思います。
# 
# ### sickit-learn 解説目次
# 
# 1. 公式データセット
# 2. データの作成
# 3. 線形回帰
# 4. [ロジスティック回帰](/article/library/sklearn/logistic_regression/) <= 本節
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sklearn/logistic_regression/lr_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sklearn/logistic_regression/lr_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[3]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[7]:


import sklearn

sklearn.__version__


# 必要なライブラリを読み込みます。

# In[6]:


import numpy as np
import scipy
from scipy.stats import binom

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("numpy version :", np.__version__)
print("matplotlib version :", matplotlib.__version__)
print("sns version :",sns.__version__)


# In[ ]:





# In[ ]:





# このオッズという言葉は競馬でよく聞くオッズと同じなんでしょうか。競馬はやらないのでわかりませんね。誰か教えてください。とりあえず、ある事象が起こる確率が$p$であるとき、$$\frac{p}{1-p}$$をオッズと言うそうです。その対数を$$\log p - \log(1-p)$$を対数オッズと言います。
# 
# $$
# a_0 x_0 + a_1 x_1 + a_2 x_2 \cdots a_n x_n = \log \frac{p}{1-p}
# $$
# これを$p$について解くと、
# 
# $$
# p = \frac{1}{1 + \exp^{-\sum_i x_i x_i}}
# $$

# ### ロジスティック関数
# 
# 一般に、$$ f(x)= \frac{1}{1+e^{-x}}$$をロジスティック関数と言います。ロジスティック関数をグラフ化してみます。scipyにモジュールとしてあるようなので、それを使います。

# In[18]:


from scipy.special import expit

x = np.linspace(-8,8,100)
y = expit(x)

plt.grid()
plt.plot(x,y)


# In[15]:


from scipy.stats import logistic

get_ipython().run_line_magic('pinfo', 'logistic')

