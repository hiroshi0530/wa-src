
# coding: utf-8

# ## Numpy個人的tips
# 
# numpyもデータ分析や数値計算には欠かせないツールの一つです。機械学習などを実装していると必ず必要とされるライブラリです。個人的な備忘録としてメモを残しておきます。詳細は以下の公式ページを参照してください。
# - [公式ページ](https://docs.scipy.org/doc/numpy/reference/)
# 
# ### 目次
# - [1. 基本的な演算](/article/library/numpy/base/) <= 今ここ
# - [2. 三角関数](/article/library/numpy/trigonometric/)
# - [3. 指数・対数](/article/library/numpy/explog/)
# - [4. 統計関数](/article/library/numpy/statistics/)
# - [5. 線形代数](/article/library/numpy/matrix/)
# - [6. サンプリング](/article/library/numpy/sampling/)
# - [7. その他](/article/library/numpy/misc/)
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/base/base_nb.ipynb)
# 
# ### 筆者の環境
# 筆者の環境とimportの方法は以下の通りです。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import numpy as np

np.__version__


# ## スカラー、ベクトル、行列、テンソル
# 
# - スカラー : 0階のテンソル
# - ベクトル : 1階のテンソル
# - 行列 : 2階のテンソル

# ## 情報の取得
# ndarray型の情報を以下の様な属性値や組み込み関数を指定することで取得する事が出来ます。
# 
# - len()
#     - 最初の要素の次元の長さを取得
# - shape
#     - 各次元の大きさ（サイズ）
# - ndim
#     - 次元
# - size
#     - 全要素数
# - itemsize
#     - 要素のメモリ容量
# - nbytes
#     - バイト数
# - dtype
#     - 型
# - data
#     - メモリアドレス
# - flags
#     - メモリ情報
# 
# 使用例は以下の通りです。

# In[4]:


a = np.array([i for i in range(2)])
b = np.array([i for i in range(4)]).reshape(-1,2)
c = np.array([i for i in range(12)]).reshape(-1,2,2)

print('a            : ', a)
print('len(a)       : ', len(a))
print('a.shape      : ', a.shape)
print('a.ndim       : ', a.ndim)
print('a.size       : ', a.size)
print('a.itemsize   : ', a.itemsize)
print('a.nbytes     : ', a.nbytes)
print('a.dtype      : ', a.dtype)
print('a.data       : ', a.data)
print('a.flgas      : \n{}'.format(a.flags))
print()
print('b            : \n{}'.format(b))
print('len(b)       : ', len(b))
print('b.shape      : ', b.shape)
print('b.ndim       : ', b.ndim)
print('b.size       : ', b.size)
print('b.itemsize   : ', b.itemsize)
print('b.nbytes     : ', b.nbytes)
print('b.dtype      : ', b.dtype)
print('b.data       : ', b.data)
print('b.flgas      : \n{}'.format(b.flags))
print()
print('c            : \n{}'.format(c))
print('len(c)       : ', len(c))
print('c.shape      : ', c.shape)
print('c.ndim       : ', c.ndim)
print('c.size       : ', c.size)
print('c.itemsize   : ', c.itemsize)
print('c.nbytes     : ', c.nbytes)
print('c.dtype      : ', c.dtype)
print('c.data       : ', c.data)
print('c.flgas      : \n{}'.format(c.flags))


# ### flagsについて
# flagsは様々な情報を返してくれます。ここでは変数のメモリの格納方法について説明します。
# 
# - [https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html)
# 
# リンクの公式ページを見ればわかりますが、配列のメモリへの割り当ての方法は2種類あります。一つは、C_CONTIGUOUSで、も一つが、F_CONTIGUOUSです。
# C_というのはC言語方式という意味で、F_というのはFORTRAN形式である事意味しています。C言語方式では
# 
# $$
#   \left(
#     \begin{array}{cc}
#       a & b  \\\\
#       c & d 
#     \end{array}
#   \right)
# $$
# 
# という変数をメモリ以上に
# 
# $$
# a,c,b,d
# $$
# 
# という順番で格納します。FORTRAN方式では、
# 
# $$
# a,b,c,d
# $$
# 
# とう順番で格納します。普段はあまり意識することはありませんが、備忘録として記載しておきます。

# ## numpyのデータ型
# 
# numpyの実際の数値計算部分はC言語で実装されています。よってデータを定義するときにデータの型を指定することが出来ます。この情報によりメモリ上に確保する量を最適化することが出来ます。大規模な数値計算に慣れなるほど、重要なプロパティになります。
# 
# 本家の[サイト](https://numpy.org/devdocs/user/basics.types.html)にはたくさんのデータタイプが定義されているますが、実際に使うのはそれほど多くありません。
# 
# <div class="table_center_80">
# 
# |表記1 |表記2|表記3|データ型 |説明  |
# |---|---|---|---|---|
# |np.bool |-| ?|bool  |真偽値  |
# |np.int8 | int8 | i1  |int8  |8ビット符号付き整数 |
# |np.int16 | int16 | i2  |int16  |16ビット符号付き整数 |
# |np.int32 | int32 | i4  |int32  |32ビット符号付き整数 |
# |np.int64 | int64 | i8  |int64  |64ビット符号付き整数 |
# |np.uint8 | uint8 | u1  |uint8  |8ビット符号なし整数 |
# |np.uint16 | uint16 | u2  |uint16  |16ビット符号なし整数 |
# |np.uint32 | uint32 | u4  |uint32  |32ビット符号なし整数 |
# |np.uint64 | uint64 | u8  |uint64  |64ビット符号なし整数 |
# |np.float16 | float16 | f2  |float16  |半精度浮動小数点型 |
# |np.float32 | float32 | f4  |float32  |単精度浮動小数点型 |
# |np.float64 | float64 | f8  |float64  |倍精度浮動小数点型 |
# |np.float128 | float128 | f16  |float128  |4倍精度浮動小数点型 |
# 
# </div>
# 
# 表記1、表記2、表記3は定義の方法としては同じです。

# In[5]:


a = np.array([i for i in range(5)  ], dtype=np.int8)
b = np.array([i for i in range(5)  ], dtype='int8')
c = np.array([i for i in range(5)  ], dtype='i1')

print(a.dtype)
print(b.dtype)
print(c.dtype)

d = np.array(True, dtype='?')
e = np.array(True, dtype=np.bool)

print(d.dtype)
print(e.dtype)


# ## axis
# numpyは高階のテンソルを利用する事ができ、平均や合計値などの統計情報を計算する際、どの方向に計算するか指定することが出来ます。その方向を指定する際、axisをいうオプションを利用します。
# 
# ### axisの方向
# ![png](base_nb_files_local/axis.png)
# 
# ### 例
# 言葉で説明するより実際に計算をさせてみた方が早いと思います。

# In[6]:


a = np.arange(10)

print('\n####### ベクトルの場合 #######')
print('\na : ')
print(a)
print('\nnp.mean(a) : ')
print(np.mean(a))
print('\nnp.mean(a, axis=0) : ')
print(np.mean(a, axis=0))

print('\n####### 行列の場合 #######')
a = np.arange(10).reshape(2,5)
print('\na : ')
print(a)
print('\nnp.mean(a) : ')
print(np.mean(a))
print('\nnp.mean(a, axis=0) : ')
print(np.mean(a, axis=0))
print('\nnp.mean(a, axis=1) : ')
print(np.mean(a, axis=1))

print('\n####### 3階のテンソルの場合 #######')
a = np.arange(24).reshape(2,3,4)
print('\na : ')
print(a)
print('\nnp.mean(a) : ')
print(np.mean(a))
print('\nnp.mean(a, axis=0) : ')
print(np.mean(a, axis=0))
print('\nnp.mean(a, axis=1) : ')
print(np.mean(a, axis=1))
print('\nnp.mean(a, axis=2) : ')
print(np.mean(a, axis=2))


# ## ブロードキャスト
# numpyは行列やベクトルとスカラー量の演算がされたとき、行列やベクトルのすべての要素に対してスカラー量の演算が実行されます。最初慣れないと勘違いしてしまうので、押さえておきましょう。スカラー量である$a$がベクトル$b$の全成分に対して演算されていることがわかります。

# In[7]:


a = 10
b = np.array([1, 2])

print('a     : ',a)
print('b     : ',b)
print('a + b : ',a + b)
print('a * b : ',a * b)
print('b / a : ',b / a)


# ## スライシング
# スライシングはndarray形式で定義された変数から、特定の数値をスライスして取り出すための手法です。とても便利なので、ぜひとも覚えておきたいです。

# In[8]:


a = np.arange(12).reshape(-1,3)

print('a : \n{}'.format(a))
print()
print('a.shape : ',a.shape)
print()
print('a[0,1]    : ', a[0,1], '## row=1, col=1の要素') 
print()
print('a[2,2]    : ', a[2,2], '## row=2, col=2の要素')
print()
print('a[1]      : ', a[1], '## row=1の要素')
print()
print('a[-1]     : ', a[-1], '## 最後の行の要素')
print()
print('2行目から3行目、1列目から2列目までの要素')
print('a[1:3,0:2]  : \n{}'.format(a[1:3,0:2]))
print()
print('すべての列、1列目から1列おきのすべての要素')
print('a[:,::2]  : \n{}'.format(a[:,::2]))
print()
print('1行目から1行おきのすべての要素')
print('a[::2]    : \n{}'.format(a[::2]))
print()
print('2行目から1行おきのすべての要素')
print('a[1::2]   : \n{}'.format(a[1::2]))
print()


# ## all, any, where
# 
# - all:要素のすべてがtrueならtrueを返す
# - any:要素の少なくても一つがtrueならtrueを返す

# In[9]:


a = np.array([[0,1],[1,1]])

print(a.all())
print(a.any())


# whereで条件を満たす要素のインデックスを返します。

# In[10]:


a = np.array([[0,2],[1,1]])

print(np.where(a>1)) ## 1より大きい2のインデックスである(0,1)を返す


# (0,1)がwhere条件に当てはまるインデックスとなります。

# ### whereの三項演算子
# whereを利用すると三項演算子の利用に利用できます。最初の条件が満たされていれば、第二引数を、満たされていなければ、第三引数の要素を取ります。この形のwhereは頻繁に利用します。

# In[11]:


a = np.array([2 *i +1 for i in range(6)]).reshape(2,3)
print('a : ', a)
print('6より大きい要素はそのままで、小さければ0とする')
np.where(a>6,a,0)


# In[12]:


a = np.array([2 *i +1 for i in range(6)]).reshape(2,3)
b = np.zeros((2,3))

print(a)
print(b)
print('aの要素が3で割り切れれば、該当するbの値を、そうでなければaの値を返す')
np.where(a%3==0, b, a)


# ## 基本定数

# ### 自然対数の底

# In[13]:


np.e


# ### 円周率

# In[14]:


np.pi


# ## 基本的な四則演算

# ### np.add(x,y)
# 要素ごとの足し算です。一般的なベクトルの加法です。

# In[15]:


a = np.array([1.,2.])
b = np.array([4.,3.])
np.add(a,b)


# ### np.reciprocal(x)
# 要素ごとの逆数になります。

# In[16]:


b = np.array([4.,3.])
np.reciprocal(b)


# この関数について面白い事に気づきました。python3系では、整数型の割り算であっても小数点以下まで計算してくれます。python2系では、整数部分だけ表示されます。しかし、逆数を計算するこの関数で整数型の逆数を計算すると、整数部分しか表示してくれません。データ型を浮動小数型である事を明示するとちゃんと小数点以下まで計算してくれます。

# In[17]:


# print(1/8) # => 0.125が返る@python3系
# print(1/8) # => 0が返る@python2系
print(np.reciprocal(8))
print(np.reciprocal(8, dtype='float16'))
print(np.reciprocal(8.))


# ### np.multiply(x,y)
# 要素ごとのかけ算です。アダマール積といわれています。ベクトルの内積とは異なります。

# In[18]:


a = np.array([1.,2.])
b = np.array([4.,3.])
np.multiply(a,b)


# ### np.divide(x,y)
# 要素ごとの割り算の商を求めます。

# In[19]:


a = np.array([1.,2.])
b = np.array([4.,3.])
np.divide(b,a)


# ### np.mod(x,y)
# 要素ごとの割り算のあまりを求めます。

# In[20]:


a = np.array([3.,2.])
b = np.array([11.,3.])
print(np.mod(b,a))


# ### np.divmod(x,y)
# 要素ごとの割り算の商とあまりを同時に求めます。

# In[21]:


a = np.array([3.,2.])
b = np.array([11.,3.])
print(np.divmod(b,a))


# ### np.power(x,y)
# 累乗の計算です。ベクトルを指定するとベクトル同士の指数の計算になります。
# 
# #### $2^3=8$

# In[22]:


np.power(2,3)


# #### $4^1$ と $3^2$

# In[23]:


a = np.array([1.,2.])
b = np.array([4.,3.])
np.power(b,a)


# ### np.subtract(x,y)
# 要素ごとの引き算です。

# In[24]:


a = np.array([1.,2.])
b = np.array([4.,3.])
np.subtract(b,a)

