#!/usr/bin/env python
# coding: utf-8

# ## Numpy個人的tips
# 
# numpyもデータ分析や数値計算には欠かせないツールの一つです。機械学習などを実装していると必ず必要とされるライブラリです。個人的な備忘録としてメモを残しておきます。詳細は以下の公式ページを参照してください。
# - [公式ページ](https://docs.scipy.org/doc/numpy/reference/)
# 
# ### 目次
# - [1. 基本的な演算](/article/library/numpy/base/)
# - [2. 三角関数](/article/library/numpy/trigonometric/)
# - [3. 指数・対数](/article/library/numpy/explog/)
# - [4. 統計関数](/article/library/numpy/statistics/) <= 今ここ
# - [5. 線形代数](/article/library/numpy/matrix/)
# - [6. サンプリング](/article/library/numpy/sampling/)
# - [7. その他](/article/library/numpy/misc/)
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/statistics/statistics_nb.ipynb)
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

import numpy as np

np.__version__


# ## 統計情報の取得

# ### np.max(x)
# 配列の最大値を返します。

# 2階のテンソルとして$a$を定義します。

# In[4]:


a = np.array([
    [1,8,3],
    [6,5,4],
    [7,2,9]
  ]
)


# 3階のテンソルとして$b$を定義します。

# In[5]:


b = np.array([
  [
    [1,8,3],
    [6,5,4],
    [7,2,9]
  ],
  [
    [1,9,4],
    [7,2,5],
    [6,8,3]
  ]
])


# In[6]:


print('-' * 20)
print('a   : \n',a)
print()
print('np.max(a) : \n',np.max(a))
print()
print('np.max(a, axis=0) : \n',np.max(a, axis=0))
print()
print('np.max(a, axis=1) : \n',np.max(a, axis=1))
print()

print('-' * 20)
print('b   : \n',b)
print()
print('np.max(b) : \n',np.max(b))
print()
print('np.max(b, axis=0) : \n',np.max(b, axis=0))

print()
print('np.max(b, axis=1) : \n',np.max(b, axis=1))

print()
print('np.max(b, axis=2 : \n',np.max(b, axis=2))


# In[7]:


print('-' * 20)
print('a   : \n',a)
print()
print('np.argmax(a) : \n',np.argmax(a))
print()
print('np.argmax(a, axis=0) : \n',np.argmax(a, axis=0))
print()
print('np.argmax(a, axis=1) : \n',np.argmax(a, axis=1))
print()

print('-' * 20)
print('b   : \n',b)
print()
print('np.argmax(b) : \n',np.argmax(b))
print()
print('np.argmax(b, axis=0) : \n',np.argmax(b, axis=0))

print()
print('np.argmax(b, axis=1) : \n',np.argmax(b, axis=1))

print()
print('np.argmax(b, axis=2 : \n',np.argmax(b, axis=2))


# ### np.argmax(x)
# 配列の最大値の位置を返します。

# In[8]:


a = np.random.randint(100,size=10)

print('a            : ',a)
print('max position : ',np.argmax(a))


# ### np.min(x)
# 配列の最小値を返します。

# In[9]:


a = np.random.randint(100,size=10)

print('a   : ',a)
print('min : ',np.min(a))


# ### np.argmax(x)
# 配列の最小値の位置を返します。

# In[10]:


a = np.random.randint(100,size=10)

print('a            : ',a)
print('min position : ',np.argmin(a))


# ### np.maximum(x,y)
# 二つの配列を比較し、大きい値を選択し新たなndarrayを作ります。

# In[11]:


a = np.random.randint(100,size=10)
b = np.random.randint(100,size=10)

print('a   : ',a)
print('b   : ',b)
print('max : ',np.maximum(a,b))


# ### np.minimum(x,y)
# 二つの配列を比較し、小さい値を選択し新たなndarrayを作ります。

# In[12]:


a = np.random.randint(100,size=10)
b = np.random.randint(100,size=10)

print('a   : ',a)
print('b   : ',b)
print('min : ',np.minimum(a,b))


# ### np.sum(a, axis=None, dtype=None, out=None, keepdims=[no value], initial=[no value], where=[no value])

# In[35]:


a = np.arange(10)
np.sum(a)


# axisを指定して計算してみます。

# In[38]:


a = np.arange(12).reshape(3,4)

print('a : ')
print(a)
print('sum axis=0 : ', np.sum(a, axis=0))
print('sum axis=1 : ', np.sum(a, axis=1))


# ### np.average(a, axis=None, weights=None, returned=False)
# 平均を求めます。重み付きの平均も求める事が出来ます。
# 
# 単純に配列の平均です。

# In[13]:


a = np.arange(10)
np.average(a)


# axisを指定した平均です。

# In[17]:


a = np.arange(12).reshape(3,4)

print('a : ', a)
print('average axis = 0 : ',np.average(a, axis=0))
print('average axis = 1 : ',np.average(a, axis=1))


# 重みを指定します。

# In[22]:


a = np.arange(5)

# 適当に重みを設定
w = np.array([0.1,0.2,0.5,0.15,0.05])

np.average(a,weights=w)


# ### np.mean(a, axis=None, dtype=None, out=None, keepdims=[no value])
# 平均を求めます。こちらは重み付きの平均を求める事が出来ません。しかし、計算時の型を指定することが出来ます。

# In[14]:


x = np.arange(10)
np.mean(x)


# 整数型を指定して計算する。

# In[25]:


x = np.arange(10)
np.mean(x, dtype='int8')


# ### np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=[no value])
# 標準偏差を求めます。

# In[26]:


x = np.arange(10)
np.std(x)


# ### np.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=[no value])
# 分散を求めます。

# In[30]:


x = np.arange(10)
np.var(x)


# ### np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)

# In[34]:


x = np.arange(10)
print(x)
print('median x : ',np.median(x))
print()

x = np.arange(11)
print(x)
print('median x : ',np.median(x))


# ### np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
# 
# bias=Trueで標本分散を求める。
# yで追加の配列を指定可能。

# In[87]:


a = np.random.randint(10,size=9).reshape(3,3)
b = np.arange(3)

print('a : ')
print(a)
print()

print('不偏分散での共分散行列')
print(np.cov(a))
print()

print('標本分散での共分散行列')
print(np.cov(a, bias=True))
print()

print('それぞれの成分の標本分散 : 共分散行列の対角成分と一致')
print('var a[0] = ', np.var(a[0]))
print('var a[1] = ', np.var(a[1]))
print('var a[2] = ', np.var(a[2]))
print()

print('bを追加')
print('b : ')
print(b)
print(np.cov(a,b, bias=True))


# ### np.corrcoef(x, y=None, rowvar=True, bias=[no value], ddof=[no value])

# In[78]:


a = np.random.randint(10,size=9).reshape(3,3)
np.corrcoef(a)


# 
