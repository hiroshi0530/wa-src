#!/usr/bin/env python
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
# - [6. サンプリング](/article/library/numpy/sampling/)
# - [7. その他](/article/library/numpy/misc/) <= 今ここ
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/matrix/matrix_nb.ipynb)
# 
# ### numpy 読み込み
# 筆者の環境とimportの方法は以下の通りです。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import numpy as np

np.__version__


# ## 便利関数

# ### np.sqrt(x)
# 平方根を計算します。

# In[4]:


np.sqrt(4)


# ### np.cbrt(x)
# 三乗根を計算します。

# In[5]:


np.cbrt(8)


# ### np.square(x)
# 2乗を計算します。

# In[6]:


np.square(2)


# ### np.absolute(x)
# 絶対値を計算します。複素数に対応しています。

# In[7]:


print(np.absolute(-4))
print(np.absolute([1,-2,-4]))
print(np.absolute(complex(1,1))) # => sqrt(2) 


# ### np.convolve(x,y)
# 畳み込みを計算します。

# In[8]:


a = np.array([1,2,3])
b = np.array([0,1.5,2])

print(np.convolve(a,b, mode='full')) # defalut mode = full
print(np.convolve(a,b, mode='same'))
print(np.convolve(a,b, mode='valid'))


# ### np.diff(a,N)
# 要素間の差分を取ります。離散値の微分と同じような作用をします。
# 
# $$
# d1_a[x] = a[x+1] - a[x]
# $$
# 
# Nに整数を代入することで、N階微分の値を求める事ができます。
# 
# $$
# d2_a[x] = d1_a[x+1] - d1_a[x]
# $$
# 

# In[9]:


a = np.random.randint(10,size=10)
print('a      : ', a)
print('1次微分 : ', np.diff(a))
print('2次微分 : ', np.diff(a, 2))


# ### np.cumsum(a)
# 要素の足し合わせになります。概念としては離散値の積分に近いです。
# $$
# s_a[x] = \sum_{i=0}^{x}a[x]
# $$

# In[10]:


a = np.random.randint(10,size=10)
print('a   : ', a)
print('積分 : ', np.cumsum(a))


# ### np.heaviside(x,c)
# 
# ヘヴィサイドの階段関数です。
# <div>
# 
# $$
# H_c(x)=
# {
#   \begin{cases}
#   1\ (x \gt 0)\\\\
#   c\ (x = 0)\\\\
#   0\ (x \lt 0)
#   \end{cases}
# }
# $$
# </div>
# 
# データ分析などではそれほど使う機会はありませんが、一応記載しておきます。
# ```python
# np.heaviside(a, 10)
# ```
# と表記し、$c=10$に対応します。

# In[11]:


a = [i for i in range(-2,3)]
print(a)
print(np.heaviside(a, 10))


# ### np.interp(x',x,y)
# 線形補間した値を返します。
# 
# ```python
# x = [0,1,2]
# y = [2,100,50]
# 
# x1 = [0.5, 1.8]
# y1 = np.interp(x1, x,y)
# ```
# 
# このように定義することで、$(x,y) = (0,2), (1,100)$を結ぶ直線の$x=0.5$の値と、$(x,y) = (1,100),(2,50)$を結ぶ直線の$x=1.8$の値を求める事ができます。以下の様にグラフに書くとわかりやすいかと思います。
# 

# In[12]:


import matplotlib.pyplot as plt

x = [0,1,2]
y = [2,100,50]

plt.grid()
plt.plot(x,y,marker='o')

x1 = [0.5, 1.8]
y1 = np.interp(x1, x,y)

print('x1 : ', x1)
print('y1 : ', y1)
plt.scatter(x1,y1,marker='^',c='red')


# ## 配列の操作

# ### ndarray.reshape(N,M,･･･)
# 配列の形状を変更します。配列のトータルのサイズは変更前後で一致する必要があります。
# 
# サイズが12の一次元配列を3x4の二次元配列に変換します。

# In[13]:


a = np.arange(12)
b = a.reshape(3,4)

print('before shape : ',a.shape)
print('after shape  : ',b.shape)


# ### np.tile(a,(N,M,･･･)
# aをタイル上に配置します。例を見た方がわかりやすいと思います。

# In[14]:


a = np.arange(5)

np.tile(a,(2,1))


# ### ndarray.flatten()
# 二次以上の配列を一次元の配列に変換します。copyを作成し、一次元に平坦化し、そのオブジェクトを返します。

# In[15]:


a = np.arange(12).reshape(3,4)
b = a.flatten()

print('a : ',a)
print('b : ',b)


# ### ndarray.ravel()
# 二次以上の配列を一次元の配列に変換します。copyを作成せず、一次元に平坦化します。一般的にflattenよりハードウェアに対する負荷が少ないのでこちらを利用する事が推奨されています。

# In[16]:


a = np.arange(12).reshape(3,4)
b = a.ravel()

print('a : ',a)
print('b : ',b)


# In[17]:


a = np.array([1,2,3])
a = [1,2,3]
b = a[:]

b[0] = 100

print(id(a))
print(id(b))
print(id(a[0]))
print(id(b[0]))
print(a)
print(b)

print(type(a))
print(type(a[:]))
print(id(a))
print(id(a[:]))


# ### ndarray.flatten と ndarray.ravelの違い
# 
# viewの場合は、元の配列が変更されたら、それを参照している配列も変更されます。
# 
# #### viewとcopy
# 
# 一般にndarray型の代入はアドレスがコピーされます。よって、参照元が変更されると、参照先も変更されます。また、メモリアドレスも一致し、オブジェクトが使用しているメモリサイズも同一です。
# 
# また、ndarray型にはbaseというメソッドがありますが、これはNoneとなっています。baseについて後述します。

# In[18]:


import sys

a = np.arange(10000)
b = a

a[1] = 100

print('a      = ', a)
print('b      = ', b)
print('id(a)  = ', id(a))
print('id(b)  = ', id(b))
print('a mem  = ', sys.getsizeof(a))
print('b mem  = ', sys.getsizeof(b))
print('a base = ', a.base)
print('b base = ', b.base)


# In[19]:


import sys

a = np.arange(10000)
b = a.copy()

a[1] = 100

print('a      = ', a)
print('b      = ', b)
print('id(a)  = ', id(a))
print('id(b)  = ', id(b))
print('a mem  = ', sys.getsizeof(a))
print('b mem  = ', sys.getsizeof(b))
print('a base = ', a.base)
print('b base = ', b.base)


# 次にbをaの先頭から10個の配列をスライスして作成してみます。そうすると、メモリアドレスが異なる別のオブジェクトが作成されます。しかし、`a[1]=100`とすると、bも変更されます。よって、bは別のオブジェクトであるが、aを参照しているオブジェクトである事がわかります。
# 
# 
# また、サイズも96バイトと、かなり小さいです。dtypeはint64なので、スライスされた値がそのまま格納されているのなら、どんなに少なくても8バイトx20=160バイトは欲しいのですが、それ以下になっています。よって、aのメモリアドレスが格納されていると考えて良いかと思います。

# In[20]:


import sys

a = np.arange(10000)
b = a[:20]

a[1] = 100

print('a     = ', a)
print('b     = ', b)
print('id(a) = ', id(a))
print('id(b) = ', id(b))
print('a mem = ', sys.getsizeof(a))
print('b mem = ', sys.getsizeof(b))
print('a base = ', a.base)
print('b base = ', b.base)


# reshapeについてもやってみます。

# In[21]:


import sys

a = np.arange(10000)
b = a.reshape(100,100)

a[1] = 100

print('a     = ', a)
print('b     = ', b)
print('id(a) = ', id(a))
print('id(b) = ', id(b))
print('a mem = ', sys.getsizeof(a))
print('b mem = ', sys.getsizeof(b))
print('a base = ', a.base)
print('b base = ', b.base)


# In[ ]:





# 以下にflattenとravelのbaseと使用メモリの差を示します。上述したとおり、flattenにbaseとなる配列は存在せず、ravelの場合は元の配列がbaseとなります。また、flattenはcopyするため、オブジェクトがメモリ上に占有するサイズは元のオブジェクトと同一ですが、ravelの場合は最小限に抑えられています。
# 
# この辺は、やや低レベルレイヤーの話ですが、ちゃんと理解しているのとしていないとでは、技術者として大きな差となります。

# In[22]:


import sys

a = np.arange(120000,dtype='int64')
b1 = a.reshape(300,400)

for i in [0,1]:
  
  if i == 0:
    print('######## flatten ########')
    b2 = b1.flatten()
  elif i == 1:
    print('######## ravel ########')
    b2 = b1.ravel()
    
  print('id')
  print('a  : ', id(a))
  print('b1 : ', id(b1))
  print('b2 : ', id(b2))
  print('')

  print('base')
  print('b1 : ', b1.base)
  print('b2 : ', b2.base)
  print('')
  
  print('オブジェクトの使用メモリ')
  print('a  : ',sys.getsizeof(a))
  print('b1 : ',sys.getsizeof(b1))
  print('b2 : ',sys.getsizeof(b2))
  print('')

一番の違いは、使用メモリがflattenの場合は960kバイトで、ravelが96バイトとなっています。かなりのメモリ削減効果が見込めます。
# ### np.hstack
# ndarray型の連結です。水平(horizontal, axis=1)方向に連結します。割とよく使います。

# In[23]:


a = np.array([1,2,3])
b = np.array([4,5,6])

print('a      : ',a)
print('b      : ',b)
print('hstack : ',np.hstack((a,b)))


# 結合したい方向の要素サイズが合っていないとエラーが生じます。例えば、
# `shape=(1,2)`と`shape=(2,1)`はエラーが生じます。

# In[24]:


a = np.array([[1,2]])
b = np.array([[1],[2]])

try:
  print('[error 発生]')
  print('hstack : ',np.hstack((a,b)))
except Exception as e:
  print(e)


# ### np.vstack
# ndarray型の連結です。垂直(vertical, axis=0)方向に連結します。こちらもかなりの頻度で利用します。また、結合したい方向にサイズが合っていないとエラーになります。

# In[25]:


a = np.array([1,2,3])
b = np.array([4,5,6])

print('a      : ',a)
print('b      : ',b)
print('hstack : ',np.vstack((a,b)))


# ### np.r_, np.c_
# 
# こちらも配列の結合です。vstackやhstackより簡単で、私はこちらの方をよく利用します。
# 
# 特にc_の方は1次元の二つの配列から、それぞれの要素を持つ2次元の配列を作れるので、重宝します。たまに忘れますが･･･

# In[26]:


x = [i for i in range(5)]
y = [i for i in range(5,10)]

print('np.c_ :',np.c_[x,y])
print()
print('np.r_ :',np.r_[x,y])


# ## 連番の作成

# ### np.arange([start, ]stop, [step, ]dtype=None)
# 連続する整数や等差数列を作成します。引数の考え方はpythonのrange()と同じです。引数の詳細は[numpy.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)を参照してください。

# In[27]:


np.arange(10)


# In[28]:


np.arange(4,12)


# In[29]:


np.arange(3,12,2)


# In[30]:


np.arange(1.5,4.5,0.5)


# ### np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
# 分割したい範囲と分割点数を指定し、等差数列を生成します。とても便利な関数なので至るところで利用します。いちいち分割点がいくらなのか計算しなくていいです。endpoint=Trueでstopを分割点に含み、分割点は、
# 
# $$
# start, start + \frac{stop - start}{num -1}, start + \frac{stop - start}{num -1} \times 2,start + \frac{stop - start}{num -1} \times 3, \cdots
# $$
# 
# となります。endpoint=Falseでstopを分割点に含まず、分割点は、
# 
# $$
# start, start + \frac{stop - start}{num}, start + \frac{stop - start}{num} \times 2,start + \frac{stop - start}{num} \times 3, \cdots
# $$
# 
# 詳細は[numpy.linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)を参照してください。

# In[31]:


np.linspace(0,1,3)


# In[32]:


np.linspace(0,1,3,endpoint=False)


# In[33]:


np.linspace(0,-11,20)


# この関数を使うと、グラフなどを簡単に描画できます。

# In[34]:


x = np.linspace(- np.pi, np.pi, 100)
y = np.sin(x)

plt.grid()
plt.title('$y=\sin x$')
plt.plot(x,y)

