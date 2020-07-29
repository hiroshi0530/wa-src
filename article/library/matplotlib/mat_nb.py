#!/usr/bin/env python
# coding: utf-8

# ## matplotlibの使い方メモ
# 
# matplotlibの使い方をメモしておきます。新しい使い方を覚えるごとに更新していきます。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/matplotlib/mat_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/matplotlib/mat_nb.ipynb)
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


# ## 簡単なグラフを書く
# 
# $y = \sin x$の関数を書いてみます。マニアックな使い方が色々あるようですが、データ分析の仕事をしていると、この形で使うことが一番多いような気がしています。グリッドやラベルの設定は重要です。matplotlibの中にtexを使えるかどうかは環境による気がします。
# 
# - plt.grid()
# - plt.title()
# - plt.xlabel()
# - plt.ylabe()
# - plt.xlim()
# - plt.legend()

# In[4]:


x = np.linspace(0,10,100)
y = np.sin(x)

plt.grid()
plt.title("sin function")
plt.xlabel("$x$")
plt.ylabel("$y = \\sin(x)$")
plt.xlim(0,8)
plt.ylim(-1.2,1.2)
plt.plot(x,y,label="$y=\\sin x$")

plt.legend()


# ## 複数のグラフを書く
# 

# In[5]:


x = np.linspace(0,10,100)
y1 = np.sin(x)
y2 = 0.8 * np.cos(x)

plt.grid()
plt.title("multi function")
plt.xlabel("$x$")
plt.xlim(0,8)
plt.ylim(-1.2,1.2)
plt.plot(x,y1, label="$y = \\sin x$")
plt.plot(x,y2, label="$y = 0.8 \\times \\cos x$")
plt.legend()


# ### グラフの線種や色を変更

# In[6]:


x = np.linspace(0,10,100)
y1 = np.sin(x)
y2 = 0.8 * np.cos(x)

plt.grid()
plt.title("multi function")
plt.xlabel("$x$")
plt.xlim(0,8)
plt.ylim(-1.2,1.2)
plt.plot(x, y1, "o", color="red", label="$y = \\sin x$")
plt.plot(x, y2, "x", color="blue", label="$y = \\sin x$")
plt.legend()


# ## ヒストグラムを作成する
# 
# 正規分布から10000サンプルをヒストグラムで表示しています。
# 密度分布を可視化することと同義です。

# In[7]:


x = np.random.randn(10000)

plt.hist(x)
print(np.random.rand())


# binなどを設定する事ができます。この辺は数をこなせば自然と覚えてしまいます。

# In[8]:


x = np.random.randn(10000)
plt.hist(x, bins=20,color="red")


# ## 三次元のグラフを書く
# それほど頻度が多いわけではないですが、3次元のグラフを書いてみます。データ分析は数百次元とか当たり前ですが、人間の感覚で理解できる次元数は3次元がやっとですね。私は3次元でも正直辛いです。。。
# 
# mplot3dというモジュールを利用します。また、3次元のグラフに特有なのがmeshgridというnumpyの関数を利用します。
# 
# 通常、$ z = x + y$という平面を$xyz$空間にプロットするには、$x$と$y$の要素数は$N(x) \times N(y)$で決定されます。本来ならばこの分だけ配列を作成する必要がありますが、これを自動的に作成してくれるのがmeshgridになります。
# 
# 例を見てみます。

# In[9]:


x = np.array([i for i in range(5)])
y = np.array([i for i in range(5)])

print('x :', x)
print('y :', y)
print()
xx, yy = np.array(np.meshgrid(x, y))
print('xx :', xx)
print()
print('yy :', yy)


# $xx$が25個の座標の$x$座標です。同様に、$yy$が25個の座標の$y$座標になります。とても便利です。

# In[10]:


from mpl_toolkits.mplot3d import Axes3D

# 2変数関数を適当に作成
def get_y(x1,x2):
  return x1 + 3 * x2 * x2 + 1

x1 = np.linspace(0,10,20)
x2 = np.linspace(0,10,20)

X1, X2 = np.meshgrid(x1, x2)
Y = get_y(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")

ax.plot(np.ravel(X1), np.ravel(X2), np.ravel(Y), "o", color='blue')
plt.show()


# In[11]:


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$f(x_1, x_2)$")

ax1.scatter3D(np.ravel(X1), np.ravel(X2), np.ravel(Y))
plt.show()


# 3次元のプロットでは、pyplot(plt)を直接いじると言うよりは、`fig=plt.figure()`でfigureオブジェクトを作り、その中でグラフを作成します。

# `plot_surface`を利用して、表面にその値に応じた色をつけることが出来ます。
# 多変数ガウス分布をプロットしてみます。
# 
# 少々負の相関関係が強いガウス分布の確率密度を取得してみます。

# In[12]:


from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,-0.8],[-0.8,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]
Z = multivariate_normal.pdf(X, mu,sigma).reshape(100, -1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='bwr', linewidth=0)
fig.show()

