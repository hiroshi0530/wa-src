
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

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import sklearn

sklearn.__version__


# 必要なライブラリを読み込みます。

# In[4]:


import numpy as np
import scipy
from scipy.stats import binom

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("numpy version :", np.__version__)
print("matplotlib version :", matplotlib.__version__)
print("sns version :",sns.__version__)


# ## 概要
# 
# ロジスティック回帰は二値分類の問題に適用します。ある人物が目的の商品を購入するか否か、ある人物に投票するか否かなどの予想に利用します。マーケティングなどの分野でもよく利用される回帰方法です。
# 
# 例えば、ある商品を購入する確率を$p$として、説明変数を$x_1,x_2,x_3 \cdots$として、確率$p$の対数オッズに対して線形回帰の式を当てはめます。対数オッズの定義は後に説明します。
# 
# 
# $$
# a_0 x_0 + a_1 x_1 + a_2 x_2 \cdots a_n x_n = \log \frac{p}{1-p}
# $$
# 
# これを$p$について解くと、
# 
# $$
# \displaystyle p = \frac{1}{1 + \exp^{ -\sum_{i=0}^n a_i x_i}}
# $$
# 
# となり、確率密度の式がロジスティック関数の形になっています。
# 
# 
# ## 実装
# では、scikit-learnでのロジスティック回帰の実装をしていきます。
# 最初に適当なデータを作成します。簡単のため1次元の場合に限定します。

# In[5]:


x = np.linspace(0,1,30)
y = np.array(list(map(lambda x: 1 if x > 0.5 else 0,x)))

plt.grid()
plt.plot(x,y,"o")
plt.show()


# ちょっと荒いですが、このデータから得られるロジスティック回帰によって、データを予測してみます。

# In[6]:


from sklearn.linear_model import LogisticRegression

x = np.linspace(0,1,30)
y = np.array(list(map(lambda x: 1 if x > 0.5 else 0,x)))

# 現在のversionだとsolverを指定しないとwarningがはかれます。詳細は公式ページにありますが、デフォルトのソルバーはL1正規化をサポートしていない模様です。
lr = LogisticRegression(solver='lbfgs', penalty='l2')

x = x.reshape(30,-1)
lr.fit(x,y)

# 予測してみる
for i in range(10):
  print('x = {:.1f}, predit ='.format(i * 0.1),lr.predict([[i * 0.1]])[0])


# となり、ちゃんと予測できています。何回か実行して$x=0.5$の時の値が多少ブレがある程度でしょう。

# ## ロジット関数
# 
# このオッズという言葉は競馬でよく聞くオッズと同じなんでしょうか。競馬はやらないのでわかりませんね。誰か教えてください。とりあえず、ある事象が起こる確率が$p$であるとき、
# 
# $$
# \frac{p}{1-p}
# $$
# 
# をオッズと言うそうです。その対数を
# 
# $$
# \log p - \log(1-p)
# $$
# 
# を対数オッズと言います。
# 
# ### ロジット関数の形
# 一般に
# 
# $$
# y = \log \frac{x}{1-x} \\,\\,\\,\\,\\, (0 < x < 1)
# $$
# 
# という関数の形をロジット関数と言うようです。scipyを用いてロジット関数の概要を見てみます。$x=0$と$x=1$で発散してしまいます。

# In[7]:


from scipy.special import logit

x = np.linspace(0,1,100)
y = logit(x)

plt.grid()
plt.plot(x,y)


# ## ロジスティック関数 (シグモイド関数)
# 
# 一般に、
# 
# $$
# f(x)= \frac{1}{1+e^{-x}}
# $$
# 
# をロジスティック関数と言います。シグモイド関数とも言います。ロジット関数の逆関数になっています。ロジスティック関数をグラフ化してみます。scipyにモジュールとしてあるようなので、それを使います。

# In[8]:


from scipy.special import expit

x = np.linspace(-8,8,100)
y = expit(x)

plt.grid()
plt.plot(x,y)

