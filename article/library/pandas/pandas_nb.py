#!/usr/bin/env python
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


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

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


# #### カラムの削除
# Deaths_Guineaというカラムを削除しています。

# In[14]:


df.drop(['Deaths_Guinea'], axis=1, inplace=True)
df.columns


# #### 統計量の取得
# 
# describe()を利用して、列ごとの統計量を取得することが出来ます。ぱっと見、概要を得たいときに有力です。

# In[15]:


df.describe()


# また、value_countsメソッドを利用して、値の頻度を簡単に求める事ができます。今回用いたデータが連続量のため、少々わかりにくいですが、0.0のデータ数が15である事がわかります。その他のデータ数はすべて1個である事がわかります。

# In[16]:


df['Deaths_Liberia'].value_counts()


# ## インデックスをdatetime型に変更
# 
# インデックスをDateに変更し、上書きします。時系列データの場合、インデックスを日付にすると解析しやすいことが多いです。ただ、単純に文字列としてインデックスするよりも、pandaに標準で備わっているdatetime型に変換すると集計処理などが便利になります。
# 
# Dateというインデックス名をYYYYMMDDに変更します。。rename関数を利用します。

# In[17]:


df.rename(columns={'Date':'YYYYMMDD'}, inplace=True)
df.set_index('YYYYMMDD', inplace=True)
df.index


# In[18]:


df.head()


# In[19]:


df.columns


# インデックスでソートします。ただ、日付が文字列のオブジェクトになっているので、目論見通りのソートになっていません。

# In[20]:


df.sort_index(ascending=True).head()


# インデックスをdatetime型に変更します。

# In[21]:


df.index


# In[22]:


df.index = pd.to_datetime(df.index, format='%m/%d/%Y')
df.index


# となり、dtype='object'からobject='datetime64'とdatetime型に変更されていることが分かります。そこでソートしてみます。

# In[23]:


df.sort_index(ascending=True, inplace=True)
df.head(10)


# In[24]:


df.tail(10)


# となり、想定通りのソートになっている事が分かります。
# 
# また、datetime型がインデックスに設定されたので、日付を扱いのが容易になっています。
# 例えば、2015年のデータを取得するのに、以下の様になります。

# In[25]:


df['2015']


# In[26]:


df['2014-12'].sort_index(ascending=True)


# さらに、平均や合計値などの統計値を、年や月単位で簡単に取得することができます。

# In[27]:


df.resample('Y').mean()


# In[28]:


df.resample('M').mean()


# In[29]:


df.resample('Y').sum()


# In[30]:


df.resample('M').sum()


# とても便利です。さらに、datetime型のもう一つの利点として、`.year`や`.month`などのメソッドを利用して、年月日を取得することが出来ます。

# In[31]:


df.index.year


# In[32]:


df.index.month


# In[33]:


df.index.day


# ## cut処理（ヒストグラムの作成）
# データの解析をしていると、データを特定の条件の下分割して、集計したいという場面がよくあります。例えば、季節ごとに集計したい場合などがあると思います。ちょっと月と季節が合っていませんが、季節でラベリングする例です。

# In[34]:


labels = ['春', '夏', '秋', '冬']
df['season'] = pd.cut(list(df.index.month),  bins=[0,3,6,9,12], labels=labels, right=True)
df[['season']][5:10]


# In[35]:


df[['season']][73:78]


# ## query, where, maskの使い方 (ソートも)
# numpyと同じように、queryやwhereなども使うことが出来ます。使い方は直感的にnumpyと同じなので、すぐに使えると思います。感染者と死者数でクエリを実行してみます。
# 
# queryは抽出したい条件式を指定します。

# In[36]:


df[['Deaths_Liberia','Cases_Liberia']].query('Deaths_Liberia > 100 and Cases_Liberia > 7000')


# whereも条件を指定すると、条件を満たすデータはそのまま、見たさないデータはNaNが格納されたデータを返します。

# In[37]:


df[['Deaths_Liberia']].where(df['Deaths_Liberia'] > 1000)


# NaNではなく、別の数字を入れることも可能です。この辺はnumpyと同じでとても助かります。

# In[38]:


df[['Deaths_Liberia']].where(df['Deaths_Liberia'] > 3000,0)


# maskメソッドはwhereと逆で、条件を満たすものを第二引数に書き換えます。

# In[39]:


df[['Deaths_Liberia']].mask(df['Deaths_Liberia'] > 3000, 0)


# ## nullの使い方
# 
# データにはしばしばNullが含まれるので、正しいデータ分析のためにはNullがどの程度含まれていて、それがどの程度解析に影響を及ぼすのか確認する必要があります。
# 
# `isnull`によって、Nullの部分をFalseにしたテーブルを作成する事が出来ます。

# In[40]:


df.isnull()


# また、sumメソッドを利用すると、各カラムごとにNullの個数をカウントする事が出来ます。

# In[41]:


df.isnull().sum()


# 同様に、meanメソッドで平均を出すことが出来ます。

# In[42]:


df.isnull().mean()


# Nullのデータを書き換えます。`fillna`というメソッドを利用します。

# In[43]:


df.fillna(value={'Cases_Liberia': 0.0, 'Deaths_Liberia': 0.0}, inplace=True)
df.isnull().sum()


# これで確かにCases_LiberiaとDeath_Liberiaのnullの数が0になりました。

# また、ある列にNullがある行を削除することが出来ます。dropnaを適用した前後のデータ数を比較してみるとわかります。削除前は、

# In[44]:


df.shape


# となります。削除後は以下の通りで、確かに削除されていることがわかります。

# In[45]:


df.dropna(subset=['Cases_Nigeria'], axis=0).shape


# In[46]:


df.dropna(subset=['Cases_Nigeria'], axis=0).isnull().sum()


# In[47]:


df.dropna(subset=['Cases_Nigeria'], axis=0).head()


# ## 列名やインデックス名の変更
# 上で既に出てきていますが、列名やインデックスの名前を変更したい場合はよくあります。renameメソッドを使います。

# In[48]:


df.rename(columns={'before': 'after'}, inplace=True)
df.rename(index={'before': 'after'}, inplace=True)


# ## SQL likeなメソッド
# SQLおなじみのgroupbyがpandasで利用できます。こちらは個人的にはよく利用しますね。

# In[49]:


df.groupby(['season'])['season'].count()


# ## CSVへ出力
# 
# メモリに格納されているすべてのデータを出力します。

# In[50]:


df.to_csv('./out.csv')


# In[51]:


get_ipython().system('head  -n 10 out.csv')


# In[52]:


df.to_csv('./out.csv', columns=['Deaths_Liberia'])


# In[53]:


get_ipython().system('head  -n 10 out.csv')


# ヘッダーとインデックスを記述しないように出来ます。

# In[54]:


df.to_csv('./out.csv', header=False, index=False)


# In[55]:


get_ipython().system('head  -n 10 out.csv')


# ### その他追記

# #### 型変換
# 型を指定して上書きします。一括の変換の表記方法です。inplaceがなく、少し時間を使ってしまい、メモしておきます。

# In[56]:


test = pd.DataFrame({'max':[1], 'min':[2], 'mean':[1.5]})

test.info()


# In[57]:


test = test.astype({'max': float, 'min': float, 'mean': float})

test.info()


# ## よく使う関数
# 
# 最後のまとめとして、良く使う関数をまとめておきます。個人的なsnipetみたいなものです。
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
# df.rename(index={'before': 'after'}, inplace=True)
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
# - [データ分析で頻出のPandas基本操作](https://qiita.com/ysdyt/items/9ccca82fc5b504e7913a)
# 実務レベルで重要な部分を丁寧に書かれています。とても参考になります。
