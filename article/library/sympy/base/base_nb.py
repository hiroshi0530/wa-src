#!/usr/bin/env python
# coding: utf-8

# ## sympy
# 
# これまで数値計算はすべてmathematicaを用いてやってきましたが、webシステム開発者としてpythonに移行しようと思います。
# 
# まずは最初の基本として、
# 
# - http://www.turbare.net/transl/scipy-lecture-notes/packages/sympy.html
# 
# に記載された問題を実際に手を動かしてやって行こうと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)
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


# ## sympyの3つの数値型
# 
# sympyは
# 
# - Real
# - Rational
# - Integer
# 
# の三つの型を持つようです。
# 

# ### 分数の扱い

# In[2]:


from sympy import *

a = Rational(1,2)
a


# ### 円周率の扱い

# In[5]:


pi.evalf()


# In[7]:


pi * 2


# In[8]:


pi ** 2


# ### 自然対数の扱い

# In[11]:


exp(1) ** 2


# In[12]:


(exp(1) ** 2).evalf()


# ### 無限大

# In[13]:


oo > 999


# In[14]:


oo + 1


# ### evalfで桁数の指定
# 
# 引数に桁数を入れるようです。

# In[15]:


pi.evalf(1000)


# ### 対数

# In[16]:


log(10)


# In[19]:


log(2,2**4).evalf()


# In[23]:


E.evalf()


# ## シンボルの定義
# 
# 代数的な操作を可能にするために、$xや$yなどの変数を宣言します。

# In[25]:


x = Symbol('x')
y = Symbol('y')


# In[26]:


x + y ** 2 + x


# In[27]:


(x + y) ** 3


# ## 代数操作
# 
# expandやsimplifyはmathematicaと同じのようで助かります。

# ### 展開

# In[28]:


expand((x + y)**3)


# ### 簡易化

# In[30]:


simplify((x + x*y) / x)


# ### 因数分解

# In[31]:


factor(x**3 + 3*x**2*y + 3*x*y**2 + y**3)


# ## 微積分

# ### 極限

# In[32]:


limit(sin(x)/ x, x, 0)


# In[33]:


limit(x, x, oo)


# In[34]:


limit(1 / x, x, oo)


# In[35]:


limit(x**x, x, 0)


# ### 微分

# In[36]:


diff(sin(x), x)


# In[37]:


diff(sin(x) ** 2, x)


# In[38]:


diff(sin(2 * x) , x)


# In[39]:


diff(tan(x), x)


# ### 微分が正しいかの確認
# 極限をとり、微分が正しいかどうかチェックできます。微分の定義に従って計算させるだけです。

# In[40]:


limit((tan(x+y) - tan(x))/y, y, 0)


# 高階微分も可能です。

# In[45]:


diff(sin(x), x, 1)


# 2階微分で元の値にマイナスをかけた値になっている事がわかります。

# In[46]:


diff(sin(x), x, 2)


# In[47]:


diff(sin(x), x, 3)


# In[48]:


diff(sin(x), x, 4)


# ### 級数展開
# Taylor展開も可能です。

# In[49]:


series(exp(x), x)


# In[50]:


series(cos(x), x)


# In[51]:


series(sin(x), x)


# 第三引数で展開する次数を指定出来るかと思いやってみました。

# In[52]:


series(exp(x), x, 6)


# 違うみたいで、第三引数に数値計算する際の中心の数で、第四引数で展開させる次数のようです。

# In[54]:


series(exp(x), x, 0, 6)


# ### 積分
# 微分が出来れば、積分ももちろん対応しています。

# In[55]:


integrate(x**3,x)


# In[56]:


integrate(-sin(x),x)


# In[57]:


integrate(exp(x),x)


# In[58]:


integrate(log(x),x)


# In[59]:


integrate(exp(-x**2),x)


# 積分区間も指定することが出来ます

# In[60]:


integrate(x**2, (x, -1, 1))


# In[61]:


integrate(sin(x), (x, 0, pi/2))


# 範囲が無限大の広義積分も可能です。

# In[62]:


integrate(exp(-x), (x, 0, oo))


# ガウス積分をやってみます。

# In[63]:


integrate(exp(-x**2), (x, -oo, oo))


# In[64]:


integrate(exp(-x**2 / 3), (x, -oo, oo))


# ### 方程式を解く
# 代数方程式を解くことも可能です。ほぼmathematicaと同じインターフェースです。

# In[65]:


solve(x**3 - 1, x)


# In[66]:


solve(x**4 - 1, x)


# 多変数の連立方程式も対応しています。

# In[67]:


solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y])


# オイラーの式も計算してくれるようです。

# In[68]:


solve(exp(x) + 1, x)


# ##  行列演算
# 
# 行列演算はnumpyでずっとやっているのでおそらく行列に関しては利用していませんが、一応ここでは手を動かして実行してみます。

# In[74]:


from sympy import Matrix

a = Matrix([[1,0], [0,1]])
b = Matrix([[1,1], [1,1]])


# In[72]:


a


# In[75]:


b


# In[77]:


a + b


# In[78]:


a * b


# ## 微分方程式
# 
# 常微分方程式も解く事が可能です。dsolveを利用します。

# In[79]:


f, g = symbols('f g', cls=Function)
f(x)


# In[80]:


f(x).diff(x,x) + f(x)


# In[81]:


dsolve(f(x).diff(x,x) + f(x), f(x))


# 実際に数値計算に利用されているのがよくわかる高機能ぶりでした。
# 
# もちろん有料であるmathematicaよりも劣る部分もあると思いますが、積極的にmathematicaからの移行を進めていきます。すごいなぁ〜

# ## 参考ページ
# 
# こちらのページを参考にしました。
# 
# - http://www.turbare.net/transl/scipy-lecture-notes/packages/sympy.html
