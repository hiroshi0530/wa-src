
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## timeit
# 関数の実行時間を計測するためのモジュールです。
# 
# [https://docs.python.org/2/library/timeit.html](https://docs.python.org/2/library/timeit.html)

# In[3]:


import timeit

timeit.timeit('[ i for i in range(10)]')
timeit.timeit('[ i for i in range(10)]')


# 平均を取るための繰り返し回数`number`を指定することが出来ます。デフォルトは1000000(100万回)です。

# In[4]:


number = 1000000
timeit.timeit('[ i for i in range(10)]', number=number)


# timeit.repeat()関数を用いて、repeatオプションを用いることにより、timeitiを多数回繰り返すことが出来ます。

# In[5]:


repeat = 5
number = 1000000
timeit.repeat('[ i for i in range(10)]', number=number,repeat=repeat)


# ## %timeit, %%timeit
# jupyter notebook形式で処理の時間を計るためのマジックコマンドです。%timeitは引数となるコマンドが対象、%%timeitはセル全体が対応します。
# また、`-r`と`-n`オプションによりtimeitのrepeatとnumberに対応させることが出来ます。

# In[6]:


get_ipython().run_line_magic('timeit', '[i ** 2 for i in range(10000)]')


# In[7]:


get_ipython().run_line_magic('timeit', '-r 5 -n 1000 [i ** 2 for i in range(10000)]')


# In[8]:


get_ipython().run_cell_magic('timeit', '', 'a = [i for i in range(10000)]\nb = list(map(lambda x: x **2, a))')


# ## 内包表記
# forなどを利用せずに、リストを作成します。リストを作成するアルゴリズムは高速化されており、推奨されているようです。
# 
# ### リスト型

# In[9]:


[i for i in range(5)]


# In[10]:


[i * 2 for i in range(10)]


# ### ifがある場合

# In[11]:


[i * 2 for i in range(10) if i % 2 == 0]


# In[12]:


[i * 2 if i % 2 == 0 else 1 for i in range(10)]


# ### 文字列もOK

# In[13]:


[ord(i) for i in "TensorFlow"]


# ### 2次配列もOK

# In[14]:


[[i for i in range(10)] for j in range(10)]


# In[15]:


[[j for i in range(10)] for j in range(10)]


# ### 時間測定
# 
# 内包表記と通常のやや冗長なfor文を用いたリスト作成方法の比較を行ってみます。

# In[16]:


get_ipython().run_line_magic('timeit', '[i for i in range(1000000)]')


# In[17]:


get_ipython().run_cell_magic('timeit', '', 'a = []\nfor i in range(1000000):\n  a.append(i)')


# 内包表記を利用すると、6割程度短縮できます。リスト型はすべて内包表記で作成した方が良さそうです。

# ### 辞書型
# 
# 辞書型にも内包表記は使えます。とても便利です。

# In[18]:


a = {'a':1, 'b':2, 'c':3}

print('a         : ',a)
print('reverse a : ',{j:i for i,j in a.items()})


# ## lambda
# 
# ### 基本
# 無名関数といわれるものです。わざわざ関数に名前を与えるまでもない関数に対して利用されます。単独で用いられることは少なく、次に説明するmapやfilterなどの高階関数、sortなどと共に利用する場合が多いです。

# In[19]:


# defという関数定義を利用していない
a = lambda x: x ** 2

print(a(10))
print((lambda x: x ** 2)(10))


# ### 引数を二つ持つ場合

# In[20]:


# スカラーの足し算
(lambda a,b: a + b)(1,2)


# In[21]:


# listの足し算
(lambda a,b: a + b)([1,2,3],[4,5,6])


# ### if ~ else ~
# lambdaの中でもif~else~が利用できます。

# In[22]:


print((lambda a: a if a == 0 else -100)(-1))
print((lambda a: a if a == 0 else -100)(0))
print((lambda a: a if a == 0 else -100)(1))


# ## 高階関数
# 関数自体を引数や返り値に含む関数の事です。引数にlambdaを利用する場面が多いと思います。
# 
# ### map
# 利用例は以下の通りです。リストのそれぞれの要素に対して、一律に引数である関数の処理を実行させます。

# In[23]:


a = [i for i in range(10)]

print('a   : ',a)
print('map : ',list(map(lambda x: x ** 2, a)))

lambdaの中にif~else~を入れた例です。
# In[24]:


a = [i for i in range(10)]

print('a   : ',a)
print('map : ',list(map(lambda x: x ** 2 if x % 3 == 0 else 100, a)))


# In[25]:


a = [i for i in range(10)]
b = [i for i in range(10,0,-1)]

print('a      : ',a)
print('b      : ',b)
print('lambda : ',list(map(lambda a,b: a + b, a,b)))


# ###  filter
# 利用例は以下の通りです。リストのそれぞれの要素に対して、一律に引数である関数の処理を実行させます。結果がfalseの要素は削除されます。

# In[26]:


a = [i for i in range(10)]

print('a       : ',a)
print('filter1 : ',list(filter(lambda x: x > 5,a)))
print('filter2 : ',list(filter(lambda x: x % 2 == 0,a)))


# ###  reduce
# 
# `resuce(f,x,[op])`で第一引数に引数を二つ持つ関数、第二引数に配列を取るように定義されています。配列の要素それぞれが逐次的に第一引数の関数の対象となります。

# In[27]:


import functools
x = [i for i in range(5)]

print('x      : ',x)
print('reduce : ',functools.reduce(lambda a,b:a+b, x))


# 計算の順序として以下の様なイメージです。
# 
# 1. `[0,1,2,3,4]` => `[0 + 1,2,3,4]` = `[1,2,3,4]`
# 2. `[1,2,3,4]` => `[1 + 2,3,4]` = `[3,3,4]`
# 3. `[3,3,4]` => `[3 + 3,4]` = `[6,4]`
# 4. `[6,4]` => `[6 + 4]` = `[10]`
# 
# 最終的に10が得られます。

# ## shutil
# 
# 使い方はたくさんあり、気まぐれで更新していきます。

# ### ディレクトリ中のファイルをすべて削除する場合
# 
# 一度ディレクトリを削除し、もう一度からのディレクトリを作成するのが良さそうです。

# In[28]:


import os
import shutil

_dir = './test/'

if os.path.exists(_dir):
  shutil.rmtree('./test')
  os.mkdir('./test')

# shutil.rmtree('./test')


# ## random
# 乱数関係のモジュールです。
# ### choice
# 与えられたリストの中から一つの要素をランダムに抽出します。

# In[29]:


import random 
random.choice(range(10))


# ### shuffle
# リストをランダムにシャッフルします。破壊的なメソッドで、オブジェクトそのものを更新します。

# In[30]:


import random
a = [i for i in range(10)]
random.shuffle(a)
print('a : ',a)


# ### sample
# リストをランダムにシャッフルした配列を返します。非破壊的なメソッドで新たなオブジェクト作成します。

# In[31]:


import random
a = [i for i in range(10)]
b = random.sample(a, len(a))
print('a : ',a)
print('b : ',b)


# ## sort
# ### sortメソッド
# 破壊的なメソッドです。元のオブジェクトを更新します。

# In[47]:


import numpy as np 
a = list(np.random.randint(10, size=10))
print('before a : ',a)
a.sort()
print('sorted a : ',a)


# ### sorted関数
# 非破壊的なメソッドです。ソート済みのオブジェクトを返します。

# In[48]:


import numpy as np 
a = list(np.random.randint(10, size=10))
print('before a : ',a)
b = sorted(a)
print('sorted a : ',b)


# ### リストやオブジェクトのソート
# keyオプションを利用して、ソートする要素を指定します。

# In[54]:


a = [
  ['a',1],
  ['b',6],
  ['c',3],
  ['d',2],
]

print('original                 : ',a)
b = sorted(a,key=lambda x:x[1])
print('sort by ascending order  : ', b)
c = sorted(a,key=lambda x:x[1], reverse=True)
print('sort by descending order : ', c)


# In[59]:


a = [
  {'a':1},
  {'a':6},
  {'a':3},
  {'a':2},
]

print('original                 : ',a)
b = sorted(a,key=lambda x:x['a'])
print('sort by ascending order  : ', b)
c = sorted(a,key=lambda x:x['a'], reverse=True)
print('sort by descending order : ', c)


# 辞書型の要素もソートして取得することが出来ます。

# In[68]:


a = {
  'a':1,
  'd':6,
  'c':3,
  'b':2,
}

print('keyでソート')
b = sorted(a.items(), key=lambda x:x[0])
c = sorted(a.items(), key=lambda x:x[0], reverse=True)
print('orig : ',a)
print('asc  : ',b)
print('des  : ',c)
print()

print('valueでソート')
b = sorted(a.items(), key=lambda x:x[1])
c = sorted(a.items(), key=lambda x:x[1], reverse=True)
print('orig : ',a)
print('asc  : ',b)
print('des  : ',c)


# ## その他
# 
# #### set型リストをdict型へキャスト
# 何かと便利なのでメモメモ。

# In[32]:


a = [
  ("a", 1),
  ("b", 2),
  ("c", 3),
  ("d", 4),
]

dict(a)

