
# coding: utf-8

# ## pandasとデータ分析
# pandasはデータ分析では必ず利用する重要なツールです。この使い方を知るか知らないか、もしくは、やりたいことをグーグル検索しなくてもすぐに手を動かせるかどうかは、エンジニアとしての力量に直結します。ここでは、具体的なデータを元に私の経験から重要と思われるメソッドや使い方を説明します。他に重要な使い方に遭遇したらどんどん追記していきます。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/pandas/pandas_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/pandas/pandas_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)


# ### importとバージョン確認

# In[4]:


import pandas as pd

print('pandas version :', pd.__version__)


# ## 基本操作
# 
# ### データの読み込みと表示
# 
# 利用させてもらうデータは[danielさんのgithub](https://github.com/chendaniely/pandas_for_everyone)になります。pandasの使い方の本を書いておられる有名な方のリポジトリです。[Pythonデータ分析／機械学習のための基本コーディング！ pandasライブラリ活用入門](https://www.amazon.co.jp/dp/B07NZP6V29/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)です。僕も持っています。とても勉強になると思います。
# 
# データはエボラ出血の発生数(Case)と死者数(Death)だと思います。
# 
# `read_csv`を利用して、CSVを読み込み、先頭の5行目を表示してみます。

# In[5]:


import pandas as pd

df = pd.read_csv('./country_timeseries.csv', sep=',')
df.head()


# 末尾の5データを表示します。

# In[6]:


df.tail()


# ### データの確認

# #### データの型などの情報を取得

# In[7]:


df.info()


# #### 大きさ（行数と列数）の確認

# In[8]:


df.shape


# #### インデックスの確認

# In[9]:


df.index


# #### カラム名の確認

# In[10]:


df.columns


# #### 任意の列名のデータの取得
# 
# カラム名を指定して、任意のカラムだけ表示させます。

# In[11]:


df[['Cases_UnitedStates','Deaths_UnitedStates']].head()


# #### 行数や列数を指定してデータを取得

# In[12]:


df.iloc[[6,7],[0,3]]


# #### ある条件を満たしたデータを取得

# In[13]:


df[df['Deaths_Liberia'] > 3000][['Deaths_Liberia']]


# #### 統計量の取得
# 
# describe()を利用して、列ごとの統計量を取得することが出来ます。ぱっと見、概要を得たいときに有力です。

# In[14]:


df.describe()


# In[ ]:





# ## インデックスをdatetime型に変更
# 
# インデックスをDateに変更し、上書きします。

# In[15]:


df.set_index('Date', inplace=True)
df.index


# In[16]:


df.head()


# ついでにDateというインデックス名も変更します。rename関数を利用します。

# In[26]:


df.rename(index={'Date':'YYYYMMDD'}, inplace=True)


# In[27]:


df.columns
# df.sort_values(by="YYYYMMDD", ascending=True).head()


# インデックスでソートします。ただ、日付が文字列のオブジェクトになっているので、目論見通りのソートになっていません。

# In[30]:


df.sort_index(ascending=True).head()


# インデックスをdatetime型に変更します。

# In[31]:


df.index


# In[33]:


df.index = pd.to_datetime(df.index, format='%m/%d/%Y')
df.index


# となり、dtype='object'からobject='datetime64'とdatetime型に変更されていることが分かります。そこでソートしてみます。

# In[35]:


df.sort_index(ascending=True).head(10)


# In[36]:


df.sort_index(ascending=True).tail(10)


# となり、想定通りのソートになっている事が分かります。
# 
# また、datetime型がインデックスに設定されたので、日付を扱いのが容易になっています。
# 例えば、2015年のデータを取得するのに、以下の様になります。

# In[41]:


df['2015']


# In[46]:


df['2014-12'].sort_index(ascending=True)


# In[38]:


df.index.year


# In[25]:


df.index


# ## queryとwhereの使い方 (ソートも)

# ## 列名やインデックス名の変更

# ## nullの使い方

# ## get_dummiesの使い方

# ## 頻出のコマンド一覧
# 概要として、よく利用するコマンドを以下に載せます。
# 
# #### 
# ```python
# df.query()
# ```
# 
# #### 
# ```python
# df.unique()
# ```
# 
# #### 
# ```python
# df.drop_duplicates()
# ```
# 
# #### 
# ```python
# df.describe()
# ```
# 
# #### 
# ```python
# df.set_index()
# ```
# 
# #### 
# ```python
# df.rename()
# ```
# 
# #### 
# ```python
# df.apply()
# ```
# 
# #### 
# ```python
# pd.cut()
# ```
# 
# #### 
# ```python
# df.isnull()
# ```
# 
# #### 
# ```python
# df.any()
# ```
# 
# #### 
# ```python
# df.fillna()
# ```
# 
# #### 
# ```python
# df.dropna()
# ```
# 
# #### 
# ```python
# df.replace()
# ```
# 
# #### 
# ```python
# df.mask()
# ```
# 
# #### 
# ```python
# df.drop()
# ```
# 
# #### 
# ```python
# df.value_counts()
# ```
# 
# #### 
# ```python
# df.groupby()
# ```
# 
# #### 
# ```python
# df.diff()
# ```
# 
# #### 
# ```python
# df.rolling()
# ```
# 
# #### 
# ```python
# df.pct_change()
# ```
# 
# #### 
# ```python
# df.plot()
# ```
# 
# #### 
# ```python
# df.pivot()
# ```
# 
# #### 
# ```python
# pd.get_dummies()
# ```
# 
# #### 
# ```python
# df.to_csv()
# ```
# 
# #### 
# ```python
# pd.options.display.max_columns = None
# ```
# 

# ## よく使う関数
# 
# 最後のまとめとして、良く使う関数をまとめておきます。
# 
# #### インデックスの変更(既存のカラム名に変更)
# 
# ```python
# df.set_index('xxxx')
# ```
# 
# #### カラム名の変更
# 
# ```python
# df.rename(columns={'before': 'after'}, inplace=True)
# ```
# 
# #### あるカラムでソートする
# 
# ```python
# df.sort_values(by='xxx', ascending=True)
# ```
# 
# #### インデックスでソートする
# 
# ```python
# df.sort_index()
# ```
# 
# #### datetime型の型変換
# ```python
# df.to_datetime()
# ```
# 
# #### NaNのカラムごとの個数
# ```python
# df.isnull().sum()
# ```
# 
# 
# 

# ## 参考文献
# - [チートシート](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
# - [read_csvの全引数について解説してくれてます](https://own-search-and-study.xyz/2015/09/03/pandas%E3%81%AEread_csv%E3%81%AE%E5%85%A8%E5%BC%95%E6%95%B0%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99/)
