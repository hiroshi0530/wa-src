
# coding: utf-8

# ## Numpy個人的tips
# 
# numpyもデータ分析には欠かせないツールの一つです。個人的な備忘録としてメモを残しておきます。詳細は
# - [公式ページ](https://docs.scipy.org/doc/numpy/reference/)
# を参照してください。
# 
# ### 目次
# - [1. 基本的な演算](/article/library/numpy/base/)
# - [2. 三角関数](/article/library/numpy/trigonometric/)
# - [3. 指数・対数](/article/library/numpy/explog/)
# - [4. 統計関数](/article/library/numpy/statistics/)
# - [5. 線形代数](/article/library/numpy/matrix/)
# - [6. サンプリング](/article/library/numpy/sampling/) <= 今ここ
# - [7. その他](/article/library/numpy/misc/)
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/sampling/sampling_nb.ipynb)
# 
# ### 筆者の環境
# 筆者の環境とimportの方法は以下の通りです。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

print(np.__version__)
print(matplotlib.__version__)


# ## 一様分布からのサンプリング

# ### np.random.seed(N)
# 乱数を生成するためのシード値。これを設定すると、次回np.randomを用いて乱数を生成する場合、同一の値を得ることが出来る。

# In[4]:


for i in range(5):
  np.random.seed(1)
  print(np.random.rand())


# ### np.random.rand(N,M,･･･)

# In[5]:


np.random.rand(2,3)


# ### np.random.random(shape)

# In[6]:


np.random.random((2,3))


# In[7]:


np.random.random(10)


# ### np.random.randint(N[,size])

# In[8]:


np.random.randint(100,size=100)


# ## ランダムな置換、抽出

# ### np.random.choice(a,[ size])
# 配列aからランダムにsize分だけ抽出します。

# In[9]:


a = np.arange(10)
print('a      : ',a)
print('1つ抽出 : ',np.random.choice(a))
print('2つ抽出 : ',np.random.choice(a,2))
print('3つ抽出 : ',np.random.choice(a,3))


# ### np.random.shuffle(a)
# 配列をランダムにシャッフルします。破壊的メソッドで元の配列を置換します。

# In[10]:


a = np.arange(9)
b = np.random.shuffle(a)

print(a)
print(b)


# 行列以上の配列に対しては、最初の軸に対してのみシャッフルされます。

# In[11]:


a = np.arange(9).reshape(-1,3)
np.random.shuffle(a)
print(a)


# ### np.random.permutation(a)
# 配列をランダムにシャッフルします。非破壊的メソッドでcopyを作成し、そのオブジェクトを返します。

# In[12]:


a = np.arange(9)
b = np.random.permutation(a)

print(a)
print(b)


# 行列以上の配列に対しては、最初の軸に対してのみシャッフルされます。

# In[13]:


a = np.arange(9).reshape(-1,3)
b = np.random.permutation(a)

print('before')
print(a)
print('after')
print(b)


# ## 確率分布からのサンプリング
# ある特定の確率分布からのサンプリングや、確率の値を求めるのはscipyを用いていますが、numpyを用いたサンプリングの例を記載しておきます。

# ### np.random.normal([loc, scale, size])
# 正規分布からのサンプリングになります。平均、標準偏差を指定します。正規分布の表式は以下の通りです。$\mu$が平均、$\sigma$が標準偏差です。
# 
# $$
# P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
# $$

# In[14]:


x = np.random.normal(1,2,10000)

# サンプリングされた値から平均と標準偏差を求める
# おおよそ設定値と一致する
print('平均    : ',np.mean(x))
print('標準偏差 : ',np.std(x))


# In[15]:


# ヒストグラムを表示し確率密度分布を確認
plt.hist(x, bins=20)


# ついでなので、このヒストグラムと表式によって解析的に得られた値が正しいか確認してみます。

# In[16]:


# ヒストグラムを正規化する
plt.hist(x, bins=20, density=1,range=(-10,10))

# 正規分布を計算しプロットしてみる
mu = 1
sigma = 2

def get_norm(x):
  import math
  return math.exp(-1 * (x - mu)**2 / 2 / sigma**2) / math.sqrt(2 * np.pi * sigma**2)

def get_norm_list(x):
  return [get_norm(i) for i in x]

x1 = np.linspace(-10,10,1000)
y1 = get_norm_list(x1)

plt.plot(x1,y1)


# おおよそ一致していることがわかります。

# ### np.random.binomal(n,p[,size])
# 二項分布です。正の整数$n$と確率$p$をパラメタとして取ります。

# $$
# P(k)=\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}
# $$

# In[17]:


x = np.random.binomial(30,0.2,10000)

plt.hist(x, bins=12)


# ### np.random.poisson(lambda[,size])
# ポアソン分布です。$\lambda$をパラメタとして取り、以下の様に表されます。
# $$
# P(k) = \frac{e^{-\lambda}\lambda^k}{k!}
# $$

# In[18]:


x = np.random.poisson(5,10000)

plt.hist(x, bins=15)


# ### np.random.beta(a,b[,size])
# ベータ分布です。$\alpha$と$\beta$という二つのパラメタを取ります。

# $$
# P(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}
# $$

# ここで$B(\alpha,\beta)$はベータ関数で、
# $$
# B(\alpha,\beta) = \int_0^1x^{\alpha-1}(1-x)^{\beta-1}dx
# $$

# In[19]:


x = np.random.beta(2,4,10000)

plt.hist(x,bins=20)


# ### np.random.gamma(a,b[,size])
# ガンマ分布です。$\alpha$と$\beta$の二つのパラメタを持ちます。表式は以下の通りです。
# 
# $$
# P(x) = \frac{x^{\alpha - 1}\exp(-\frac{x}{\beta})}{\Gamma(\alpha)\beta^\alpha} \quad (x \geq 0)
# $$
# 
# $\Gamma$はガンマ関数で
# 
# $$
# \Gamma(x) = \int_0^\infty t^{x-1}e^{-t} dt \quad (x \geq 0)
# $$
# 
# と定義される超関数である。

# In[20]:


x = np.random.gamma(3,2,10000)

plt.hist(x, bins=20)


# ### np.random.multivariate_normal(mean, cov[, size, check_valid, tol])
# 平均と共分散行列を指定し、相関のある多次元の正規分布からサンプリングします。
# 
# 共分散行列が
# 
# $$
# \left(
#   \begin{array}{cc}
#     1 & 0 \\\\
#     0 & 1 
#   \end{array}
# \right)
# $$
# 
# の場合、無相関な二つの正規分布からサンプリングすることになります。
# 
# $$
# \left(
#   \begin{array}{cc}
#     1 & 1 \\\\
#     1 & 1 
#   \end{array}
# \right)
# $$
# 
# で完全相関です。
# 
# $$
# \left(
#   \begin{array}{cc}
#     1 & 0.5 \\\\
#     0.5 & 1 
#   \end{array}
# \right)
# $$
# 
# で共分散が0.5の二つの正規分布からのサンプリングになります。
# 
# $$
# \left(
#   \begin{array}{cc}
#     1 & -0.7 \\\\
#     -0.7 & 1 
#   \end{array}
# \right)
# $$
# 
# で負の相関になります。

# In[21]:


a = np.array([0,0])
b = np.array([[1,0],[0,1]])

sample = np.random.multivariate_normal(a,b,size=1000)

import seaborn as sns

sns.set(style='darkgrid')
sns.jointplot(x=sample[:,0], y=sample[:,1], kind='kde')


# In[22]:


a = np.array([0,0])
b = np.array([[1,0.5],[0.5,1]])

sample = np.random.multivariate_normal(a,b,size=1000)
sns.jointplot(x=sample[:,0], y=sample[:,1], kind='kde')


# In[23]:


a = np.array([0,0])
b = np.array([[1,0.8],[0.8,1]])

sample = np.random.multivariate_normal(a,b,size=1000)
sns.jointplot(x=sample[:,0], y=sample[:,1], kind='kde')


# In[24]:


a = np.array([0,0])
b = np.array([[1,0.95],[0.95,1]])

sample = np.random.multivariate_normal(a,b,size=1000)
sns.jointplot(x=sample[:,0], y=sample[:,1], kind='kde')


# In[25]:


a = np.array([0,0])
b = np.array([[1,-0.7],[-0.7,1]])

sample = np.random.multivariate_normal(a,b,size=1000)
sns.jointplot(x=sample[:,0], y=sample[:,1], kind='kde')


# 機械学習に必要な確率分布は外にもたくさんありますが、numpyでのサンプリングはここまでにしておきます。より詳細には[scipy](/tags/scipy/)で説明します。
