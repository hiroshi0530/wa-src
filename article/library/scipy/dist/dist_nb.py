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
# \displaystyle P\left( x \| \mu, \sigma \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x-\mu)^2}{2 \sigma^2} \right)
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
# 標準正規分布から1000個データをサンプリングして、ヒストグラム表示してみます。ちゃんと確率密度関数を再現できていることがわかります。

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
# #### 確率質量関数

# In[6]:


from scipy.stats import bernoulli

mu=0.3

print(bernoulli.pmf(0, mu))
print(bernoulli.pmf(1, mu))


# #### サンプリング
# 1000個サンプリングして、ヒストグラムで表示してみます。

# In[7]:


from scipy.stats import bernoulli

mu=0.3
size=1000

x = bernoulli.rvs(mu, size=size)
plt.hist(x, bins=3)


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
# #### 確率質量関数
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


# #### サンプリング
# 
# $n=100, p=0.3$のパラメタを持つ二項分布からサンプリングを行い、確率質量関数が正しいことを確認します。

# In[9]:


from scipy.stats import binom

n = 100
p = 0.3

x = binom.rvs(n,p,size=10000)

plt.xlim(0, 60)
plt.grid()
plt.hist(x, bins=15)


# ### マルチヌーイ分布（カテゴリ分布）
# ベルヌーイ分布の多変数化になります。ベルヌーイ分布がコインの裏表が出る確率分布を表現するのであれば、サイコロの面が出る確率分布を表現します。次に説明する多項分布の$N=1$の場合に相当します。
# 
# #### 表式
# 
# $ \displaystyle
# P\left(x_1, x_2, \cdots x_n | n,p_1,p_2 \cdots p_n \right) = p_1^{x_1}p_2^{x_2}\cdots p_n^{x_n}
# $
# 
# ただし、$p_i$と$x_i$は $ \displaystyle \sum_i p_i = 1 , \sum_i x_i = 1 $を満たす。
# 
# #### 平均
# $ \displaystyle
# E\left[x_i \right] = p_i 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x_i \right] = p_i(1-p_i)
# $
# 
# #### 確率質量関数
# 
# scipyではマルチヌーイ分布専用の関数はないようなので、多項分布の$N=1$の場合を利用します。

# In[10]:


from scipy.stats import multinomial

rv = multinomial(1,[0.1,0.2,0.25,0.15,0.2,0.1])
print('確率 : ',str(rv.pmf([1,2,1,3,2,1]))[:8])


# ### 多項分布
# 二項分布の多変数版です。いかさまの可能性があるサイコロを複数回振って、それぞれの目が出る回数が従う確率分布になります。サイコロの面が$n$個あり、それぞれが一回の試行で出る確率が$p_1,p_2, \cdots , p_n$で、$N$回そのサイコロを振ったとき、それぞれの面が出る確率を$x_1,x_2, \cdots , x_n$とします。
# 
# #### 表式
# $ \displaystyle
# P\left(x_1, x_2, \cdots x_n | n,p_1,p_2 \cdots p_n \right) = \frac{N!}{x_1!x_2! \cdots x_n!} p_1^{x_1}p_2^{x_2}\cdots p_n^{x_n}
# $
# 
# ただし、$p_i$と$x_i$は $ \displaystyle \sum_i p_i = 1 , \sum_i x_i = N $を満たす。
# 
# #### 平均
# $ \displaystyle
# E\left[x_i \right] = Np_i 
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x_i \right] = Np_i\left(1-p_i \right) 
# $
# 
# #### 確率質量関数
# 6面体のサイコロで、1から6それぞれの目が出る確率は、$0.1,0.2,0.25,0.15,0.2,01$とします。そのサイコロを10回振って、それぞれの目が1回,2回,1回,3回,2回,1回出る確率を計算します。

# In[11]:


from scipy.stats import multinomial

rv = multinomial(10,[0.1,0.2,0.25,0.15,0.2,0.1])
print('確率 : ',str(rv.pmf([1,2,1,3,2,1]))[:8])


# #### サンプリング
# 上記のサイコロを10回投げて、それぞれの目が出る回数をサンプリングします。とりあえず一回だけサンプリングしてみます。

# In[12]:


multinomial.rvs(10, [0.1,0.2,0.25,0.15,0.2,0.1], size=1)


# 10000回サンプリングしてみます。

# In[13]:


from collections import Counter

array = multinomial.rvs(10, [0.1,0.2,0.25,0.15,0.2,0.1], size=10000)
array = map(str, array)

print('出現する上位10個のデータ')
for i in Counter(array).most_common()[:10]:
  print(i)


# 結果として、一番出現する確率する3の目が出る出現する目のセットが多いことがわかります。

# ### ベータ分布
# 
# #### 表式
# $ \displaystyle
# P\left( x | \alpha, \beta \right) = \frac{x^{\alpha - 1}(1-x)^{\beta - 1}}{B \left(\alpha, \beta \right)}
# $
# 
# ここで$\displaystyle B(\alpha, \beta)$はベータ関数で$\alpha, \beta > 0$となります。
# 
# #### 平均
# $ \displaystyle
# E\left[x \right] = \frac{\alpha}{\alpha + \beta}
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
# $
# 
# 
# #### 確率質量関数

# In[14]:


from scipy.stats import beta

alpha_list = [2,3,4]
beta_list = [2,2,3]

for _alpha,_beta in zip(alpha_list, beta_list):
  x = np.linspace(0,1,100)[1:-1]
  y = beta.pdf(x, _alpha, _beta)
  plt.plot(x,y,label="$\\alpha={}, \\beta={}$".format(_alpha,_beta))

plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.grid()
plt.legend()


# #### サンプリング
# 
# $\alpha=2, \beta=2$のベータ関数に対してサンプリングを行いヒストグラム表示して、確率密度関数が確かに正しいことを確認します。

# In[15]:


from scipy.stats import beta

_alpha = 2
_beta = 2

plt.grid()
plt.hist(beta.rvs(_alpha, _beta, size=100000))


# となり、上記の$\alpha=2, \beta=2$のベータ関数と形状が一致する事が分かります。

# ### ガンマ分布
# 
# #### 表式
# $ \displaystyle
# P(x|\alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}
# $
# 
# もしくは、
# 
# $ \displaystyle
# P(x|\alpha, \theta) = \frac{x^{\alpha-1}e^{-\frac{x}{\theta}}}{\Gamma(\alpha)\theta^\alpha }
# $
# 
# #### 平均
# $ \displaystyle
# E\left[x \right] = \frac{\alpha}{\beta}
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = \frac{\alpha}{\beta^2}
# $
# 
# #### 確率質量関数

# In[16]:


from scipy.stats import gamma

_alpha_list = [1.0, 2.0, 3.0, 9.0]
_beta_list = [2.0, 2.5, 4.5, 5]

for _alpha, _beta in zip(_alpha_list, _beta_list):
  x = np.linspace(0,4,100)
  y = gamma.pdf(x,_alpha,scale=1/_beta)
  plt.plot(x,y,label="$\\alpha = {}, \\beta = {}$".format(_alpha, _beta))

plt.grid()
plt.legend()


# #### サンプリング
# ベータ分布と同様に、サンプリングを行い、ヒストグラムを作成し、上記の確率密度関数が正しいことを確認します。$\alpha=2.0, \beta=2.5$のヒストグラムを作成します。

# In[17]:


_alpha = 2.0
_alpha = 2.5

plt.grid()
plt.hist(gamma.rvs(_alpha, _beta, size=10000), bins=10)


# ### カイ二乗分布
# 
# 標準正規分布に従う確率変数の二乗和の従う分布だそうです。正直あまり使う機会はないですが、区間推定やカイ二乗検定などに利用します。
# 
# $x_i \sim N(0,1)$であり、
# 
# $$ \displaystyle
# Z = \sum_{i=1}^{k}x_i^2
# $$
# 
# が従う分布という事になります。
# 
# #### 表式
# 
# $ \displaystyle
# f(x|k) = \frac{x^{\frac{k}{2}-1}e^{-\frac{x}{2}}}{2^{\frac{k}{2}}\Gamma\left(\frac{k}{2}\right)} (x \gt 0)
# $
# 
# #### 平均
# $ \displaystyle
# E\left[x \right] =  k
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = 2k
# $ 
# 
# 
# #### 確率質量関数

# In[18]:


from scipy.stats import chi2

k = 3.0

x = np.linspace(0, 20 ,100)
y = chi2.pdf(x, k)

plt.grid()
plt.plot(x, y)
plt.show()


# #### サンプリング
# $k=3$の場合について、ヒストグラムを作成し、確率密度関数の様子を確認してみます。

# In[19]:


plt.grid()
plt.hist(chi2.rvs(3,size=1000),bins=12)
plt.show()


# ### ステューデントのt分布
# 
# 単にt分布という事が多いようです。こちらもt検定などの検定に利用されます。サラリーマン時代、先輩にこの確率分布を紹介され、データ分析に利用したことを覚えています。得られるデータ数が少ない場合に適用可能で、製造業の工場などでも利用できると思います。
# 
# #### 表式
# 
# $ \displaystyle
# p(x | \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\left(\frac{\nu+1}{2}\right)}
# $
# 
# #### 平均
# $ \displaystyle
# E\left[x \right] = 0 \cdots (k > 1)
# $
# 
# #### 分散
# $ \displaystyle
# V\left[x \right] = \frac{k}{k-2} \cdots (k > 2)
# $
# 
# 
# #### 確率質量関数

# In[20]:


from scipy.stats import t

nu_list = [1,3,5]

plt.grid()

for nu in nu_list:

  x = np.linspace(-5,5,100)
  y = t.pdf(x,nu)
  
  plt.plot(x,y,label="$\\nu = {}$".format(nu))

plt.title("t-distribution")
plt.legend()
plt.show()


# #### サンプリング
# 
# $\nu = 3$の場合について、確率密度関数からサンプリングを行い、ヒストグラム表示してみます。

# In[21]:


from scipy.stats import t

nu = 3

plt.grid()
plt.xlim(-10,10)
plt.hist(t.rvs(nu, size=10000), bins=80)
plt.show()


# In[ ]:





# ## 指数型分布族

# ## 多変数ガウス分布
# 
# 統計モデリングでは相関関係のある多変数ガウス分布からサンプリングする事があります。
# 
# $$ \displaystyle
# P(x| \mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^{k} \det\Sigma}} \exp \left( - \frac{(x-\mu)^{T}\Sigma^{-1}(x-\mu)}{2} \right)
# $$
# 
# 

# In[40]:


from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,0],[0,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

print(X[0].shape)
print(X[1].shape)
print(np.ravel(X[0]))
X = np.array([np.ravel(X[0]), np.ravel(X[1])])
XX = []
for i,j in zip(np.ravel(X[0]), np.ravel(X[1])):
  XX.append([i,j])

Z = multivariate_normal.pdf(XX, mu,sigma)

sample


# In[26]:



sample = multivariate_normal.rvs(a,b,1000)
sample

