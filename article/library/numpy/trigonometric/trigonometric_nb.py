#!/usr/bin/env python
# coding: utf-8

# ## Numpy個人的tips
# 
# numpyもデータ分析や数値計算には欠かせないツールの一つです。機械学習などを実装していると必ず必要とされるライブラリです。個人的な備忘録としてメモを残しておきます。詳細は以下の公式ページを参照してください。
# - [公式ページ](https://docs.scipy.org/doc/numpy/reference/)
# 
# ### 目次
# - [1. 基本的な演算](/article/library/numpy/base/)
# - [2. 三角関数](/article/library/numpy/trigonometric/) <= 今ここ
# - [3. 指数・対数](/article/library/numpy/explog/)
# - [4. 統計関数](/article/library/numpy/statistics/)
# - [5. 線形代数](/article/library/numpy/matrix/)
# - [6. サンプリング](/article/library/numpy/sampling/)
# - [7. その他](/article/library/numpy/misc/)
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/trigonometric/trigonometric_nb.ipynb)
# 
# ### 筆者の環境
# 筆者の環境とimportの方法は以下の通りです。

# In[1]:


get_ipython().system('sw_vers')


# In[1]:


get_ipython().system('python -V')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print(np.__version__)
print(matplotlib.__version__)


# ## 三角関数
# 
# ### np.sin(x)
# $\sin x$です。

# In[3]:


print(np.sin(0))
print(np.sin(np.pi / 2))
print(np.sin(np.pi))


# In[4]:


x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)

plt.grid()
plt.title('$y = \sin x$', fontsize=16)
plt.ylabel('$\sin x$')
plt.plot(x,y)


# ### np.cos(x)
# $\cos x$です。

# In[5]:


print(np.cos(0))
print(np.cos(np.pi / 2))
print(np.cos(np.pi))


# In[6]:


x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.cos(x)

plt.grid()
plt.title('$y = \cos x$', fontsize=16)
plt.ylabel('$\cos x$')
plt.plot(x,y)


# ### np.tan(x)
# $\tan x$です。

# In[7]:


print(np.tan(0))
print(np.tan(np.pi / 4))
print(np.tan(np.pi))


# In[8]:


x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.tan(x)

plt.grid()
plt.title('$y = \\tan x$', fontsize=16)
plt.ylabel('$\tan x$')
plt.ylim(-5,5)
plt.plot(x,y)


# ### np.arcsin(x)
# $\sin x$の逆関数です。

# In[9]:


print(np.arcsin(0))
print(np.arcsin(1))
print(np.arcsin(-1))


# In[10]:


x = np.linspace(-1, 1, 100)
y = np.arcsin(x)

plt.grid()
plt.title('$y = \\arcsin x$', fontsize=16)
plt.plot(x,y)


# ### np.arccos(x)
# $\cos x$の逆関数です。

# In[11]:


print(np.arccos(0))
print(np.arccos(1))
print(np.arccos(-1))


# In[12]:


x = np.linspace(-1, 1, 100)
y = np.arccos(x)

plt.grid()
plt.title('$y = \\arccos x$', fontsize=16)
plt.plot(x,y)


# ### np.arctan(x)
# $\tan x$の逆関数です。

# In[13]:


print(np.arctan(0))
print(np.arctan(1))
print(np.arctan(-1))


# In[14]:


x = np.linspace(-np.pi, np.pi, 100)
y = np.arctan(x)

plt.grid()
plt.title('$y = \\arctan x$', fontsize=16)
plt.plot(x,y)


# ### np.sinh(x)
# 双曲線正弦関数です。
# <div>
# $
# \displaystyle \sinh x = \frac{e^x - e^{-x}}{2}
# $
# </div>

# In[15]:


print(np.sinh(0))
print(np.sinh(-1))
print(np.sinh(1))


# In[16]:


x = np.linspace(-np.pi, np.pi, 100)
y = np.sinh(x)

plt.grid()
plt.title('$y = \sinh x$', fontsize=16)
plt.plot(x,y)


# ### np.cosh(x)
# 双曲線余弦関数です。
# <div>
# $
# \displaystyle \cosh x = \frac{e^x + e^{-x}}{2}
# $
# </div>

# In[17]:


print(np.cosh(0))
print(np.cosh(-1))
print(np.cosh(1))


# In[18]:


x = np.linspace(-np.pi, np.pi, 100)
y = np.cosh(x)

plt.grid()
plt.title('$y = \cosh x$', fontsize=16)
plt.plot(x,y)


# ### np.tanh(x)
# 双曲線正接関数です。
# <div>
# $
# \displaystyle \tanh x = \frac{\sinh x}{\cosh x}
# $
# </div>
# 
# 深層学習の活性化関数に利用される事があります。

# In[19]:


print(np.tanh(0))
print(np.tanh(-1))
print(np.tanh(1))


# In[20]:


x = np.linspace(-np.pi, np.pi, 100)
y = np.tanh(x)

plt.grid()
plt.title('$y = \\tanh x$', fontsize=16)
plt.plot(x,y)


# ### np.arcsinh(x)
# $\sinh x$の逆関数です。

# In[21]:


print(np.arcsinh(0))
print(np.arcsinh(1))
print(np.arcsinh(-1))


# In[22]:


x = np.linspace(-np.pi, np.pi, 100)
y = np.arcsinh(x)

plt.grid()
plt.title('$y = \\arcsinh x$', fontsize=16)
plt.plot(x,y)


# ### np.arccosh(x)
# $\cosh x$の逆関数です。

# In[23]:


print(np.arccosh(1))


# In[24]:


x = np.linspace(1, np.pi, 100)
y = np.arccosh(x)

plt.grid()
plt.title('$y = \\arccosh x$', fontsize=16)
plt.plot(x,y)


# ### np.arctanh(x)
# $\tanh x$の逆関数です。

# In[25]:


print(np.arctanh(0))
print(np.arctanh(0.5))
print(np.arctanh(-0.5))


# In[26]:


x = np.linspace(-0.99, 0.99, 100)
y = np.arctanh(x)

plt.grid()
plt.title('$y = \\arctanh x$', fontsize=16)
plt.plot(x,y)


# ### np.deg2rad(x)
# 弧度法からラジアン表記に変換します。

# In[27]:


np.deg2rad(45) # => pi / 4 


# ### np.rad2deg(x)
# 弧度法からラジアン表記に変換します。

# In[28]:


np.rad2deg(np.pi / 4)

