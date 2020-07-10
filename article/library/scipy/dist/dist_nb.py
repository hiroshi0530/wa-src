#!/usr/bin/env python
# coding: utf-8

# ## scipyによる確率分布と特殊関数
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/scipy/dist/dist_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/scipy/dist/dist_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)


# ## 基本的な確率分布
# 
# ここではベイズ統計を理解するために必要な基本的な確率分布と必要な特殊関数の説明を行います。基本的な性質をまとめています。また、サンプリングするpythonのコードも示しています。基本的にはscipyモジュールの統計関数を利用するだけです。
# 
# ### 正規分布
# 
# 正規分布は最も良く使われる確率分布であり、その詳細は教科書に詳細に書かれているので割愛します。ここではscipyによる利用方法にとどめます。
# 
# #### 表式
# $
# \displaystyle P\left( x \| \mu, \sigma \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{x-\mu}{2 \sigma^2} \right)
# $
# 
# #### 平均
# $
# E\left[x \right]=\mu
# $
# 
# #### 分散
# $
# V\left[x\right]=\sigma^2
# $
# 
# #### 確率密度関数の取得
# pythonで正規分布の確率密度関数の生成方法は以下の通りです。中心0、標準偏差1のグラフになります。

# In[4]:


from scipy.stats import norm

mu = 0
sigma = 1

X = np.arange(-5, 5, 0.1)
Y = norm.pdf(X, mu, sigma)

plt.xlabel('$x$')
plt.ylabel('$P$')
plt.grid()
plt.plot(X, Y)


# 実際に定量的なデータ分析の現場では、確率密度関数自体はそれほど使いません。統計モデリングなどでは、仮定された確率分布からデータを仮想的にサンプリングすることがしばしば行われます。確率分布を仮定した場合のシミュレーションのようなものです。

# #### サンプリング

# In[5]:


from scipy.stats import norm

mu = 0
sigma = 1

x = norm.rvs(mu, sigma, size=1000)
plt.grid()
plt.hist(x, bins=20)


# #### help
# これ以外にもscipy.stats.normには様々な関数が用意されています。
# 
# ```python
# norm?
# ```
# 
# これを実行するとどのような関数が用意されているかわかります。
# `rvs: random variates`や`pdf : probability density function`、`logpdf`、`cdf : cumulative distribution function`などを利用する場合が多いと思います。

# ```txt
# 
# rvs(loc=0, scale=1, size=1, random_state=None)
#     Random variates.
# pdf(x, loc=0, scale=1)
#     Probability density function.
# logpdf(x, loc=0, scale=1)
#     Log of the probability density function.
# cdf(x, loc=0, scale=1)
#     Cumulative distribution function.
# logcdf(x, loc=0, scale=1)
#     Log of the cumulative distribution function.
# sf(x, loc=0, scale=1)
#     Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
# logsf(x, loc=0, scale=1)
#     Log of the survival function.
# ppf(q, loc=0, scale=1)
#     Percent point function (inverse of ``cdf`` --- percentiles).
# isf(q, loc=0, scale=1)
#     Inverse survival function (inverse of ``sf``).
# moment(n, loc=0, scale=1)
#     Non-central moment of order n
# stats(loc=0, scale=1, moments='mv')
#     Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
# entropy(loc=0, scale=1)
#     (Differential) entropy of the RV.
# fit(data, loc=0, scale=1)
#     Parameter estimates for generic data.
# expect(func, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
#     Expected value of a function (of one argument) with respect to the distribution.
# median(loc=0, scale=1)
#     Median of the distribution.
# mean(loc=0, scale=1)
#     Mean of the distribution.
# var(loc=0, scale=1)
#     Variance of the distribution.
# std(loc=0, scale=1)
#     Standard deviation of the distribution.
# interval(alpha, loc=0, scale=1)
#     Endpoints of the range that contains alpha percent of the distribution
# 
# ```

# ### ベルヌーイ分布
#  コイン投げにおいて、表、もしくは裏が出る分布のように、試行の結果が2通りしか存在しない場合の分布を決定します。
# 
# #### 表式
# $
# P\left( x \| \mu \right) = \mu^{x}\left(1-\mu \right)^{1-x} \quad \left(x = 0 \text{ or }1 \right) 
# $
# 
# もしくは、
# 
# <div>
# $$
#  P\left(x | \mu \right)= \begin{cases}
#    1 - \mu &\text{if } x = 0 \\
#    \mu &\text{if } x = 1
# \end{cases}
# $$
# </div>
# 
# となります。$x$が0か1しか取らないことに注意してください。
# 
# #### 平均
# $
# E\left[x \right]=\mu
# $
# 
# #### 分散
# $
# V\left[x\right]=\mu\left(1-\mu \right)
# $
# 
# #### python code

# In[7]:


from scipy.stats import bernoulli

mu=0.3
size=100

print(bernoulli.rvs(mu, size=size))


# ### 二項分布
# コイン投げを複数回行った結果、表が出来る回数の確率が従う分布になります。回数が1の時はベルヌーイ分布に一致します。$n$がコインを投げる回数、$p$が表が出る確率、$k$が表が出る回数で確率変数になります。詳細な計算は別途教科書を参照してください。
# 
# #### 表式
# $ \displaystyle
# P\left(k | n,p \right) = \frac{n!}{k!\left(n-k \right)!} p^{k}\left( 1-p\right)^{n-k}
# $
# 
# #### 平均
# $ \displaystyle
# E\left[k \right] = np
# $
# 
# #### 分散
# $ \displaystyle
# V\left[k \right] = np\left(1-p \right) 
# $
# 
# #### python code と グラフ
# 
# $p=0.3, 0.5, 0.9$の場合の二項分布の確率質量関数になります。

# In[8]:


import numpy as np
import scipy
from scipy.stats import binom
import matplotlib.pyplot as plt

n = 100
x = np.arange(n)

# case.1
p = 0.3
y1 = binom.pmf(x,n,p)

# case.2
p = 0.5
y2 = binom.pmf(x,n,p)

# case.3
p = 0.9
y3 = binom.pmf(x,n,p)

# fig, ax = plt.subplots(facecolor="w")
plt.plot(x, y1, label="$p=0.3$")
plt.plot(x, y2, label="$p=0.5$")
plt.plot(x, y3, label="$p=0.9$")

plt.xlabel('$k$')
plt.ylabel('$B(n,p)$')
plt.title('binomial distribution n={}'.format(n))
plt.grid(True)

plt.legend()


# ### カテゴリ分布
# #### 平均
# $ \displaystyle
# \left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $

# In[ ]:





# ### 多項分布
# #### 平均
# $ \displaystyle
# E\left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $

# In[ ]:





# ### ベータ分布
# #### 平均
# $ \displaystyle
# E\left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $

# In[ ]:





# ### ガンマ分布
# #### 平均
# $ \displaystyle
# E\left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $

# In[ ]:





# ### カイ二乗分布
# #### 平均
# $ \displaystyle
# E\left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $ 

# In[ ]:





# ### ステューデントのt分布
# #### 平均
# $ \displaystyle
# E\left[x \right] = 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 
# $

# In[ ]:




