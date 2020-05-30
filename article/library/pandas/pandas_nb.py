
# coding: utf-8

# ## pandasとデータ分析
# pandasはデータ分析では必ず利用する重要なツールです。この使い方を知るか知らないか、もしくは、やりたいことをグーグル検索しなくてもすぐに手を動かせるかどうかは、エンジニアとしての力量に直結します。ここでは、具体的なデータを元に私の経験から重要と思われるメソッドや使い方を説明します。他に重要な使い方に遭遇したらどんどん追記していきます。
# 
# また、jupyter形式のファイルは[github](https://github.com/hiroshi0530/wa/blob/master/src/pandas/pandas_nb.ipynb)に置いておきます。
# 
# ## 頻出のコマンド一覧
# 概要として、よく利用するコマンドを以下に載せます。
# 概要として、よく利用するコマンドを以下に載せます。
# 
# ### ファイルのIO
# 
# #### CSVファイルの読み込み
# ```python
# df.read_csv()
# ```
# 
# #### EXCELファイルの読み込み
# ```python
# df.read_excel()
# ```
# 
# #### 先頭の5行を表示
# ```python
# df.head()
# ```
# 
# #### 最後の5行を表示
# ```python
# df.tail()
# ```
# 
# #### インデックスの確認
# ```python
# df.index
# ```
# 
# #### サイズの確認
# ```python
# df.shape
# ```
# 
# #### カラム名の確認
# ```python
# df.columns
# ```
# 
# #### データ形式の確認
# ```python
# df.dtypes
# ```
# 
# #### 
# ```python
# df.loc[]
# ```
# 
# #### 
# ```python
# df.iloc[]
# ```
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
# df.sort_values()
# ```
# 
# #### 
# ```python
# df.to_datetime()
# ```
# 
# #### 
# ```python
# df.sort_index()
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
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# #### 
# ```python
# 
# ```
# 
# 
# 
# ## 具体例
# 
# 以下実際のデータを用いて、上記のコマンドの利用例を説明します。
# 
# ### 環境
# 最初に、私の実行環境のOSとterminalの環境です。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('uname -a | awk \'{c="";for(i=3;i<=NF;i++) c=c $i" "; print c}\'')


# ### importとバージョン確認

# In[3]:


import pandas as pd

pd.__version__


# ### データの読み込み
# データの例として、Googleのtensorflowのページでも利用されている[Auto MPG](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/) のデータセットを利用します。wgetでデータをダウンロードします。-O オプションで上書きします。

# In[4]:


get_ipython().system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data -O ./auto-mpg.data   ')


# データ属性は[本家のホームページ](https://archive.ics.uci.edu/ml/datasets/auto+mpg)によると、
# 
# 1. mpg: continuous
# 2. cylinders: multi-valued discrete
# 3. displacement: continuous
# 4. horsepower: continuous
# 5. weight: continuous
# 6. acceleration: continuous
# 7. model year: multi-valued discrete
# 8. origin: multi-valued discrete
# 9. car name: string (unique for each instance)
# 
# となっています。詳細は[本家のホームページ](https://archive.ics.uci.edu/ml/datasets/auto+mpg)を参照してください。
# 
# データの概要を見てみます。

# In[5]:


get_ipython().system('head -n 5 auto-mpg.data')


# 9個のカラムがあります。また、データの区切り形式を確認するため、タブを可視化するコマンドを実行します。catのtオプションになります。私の環境はmacOSですので、linux環境の方はmanで調べてください。

# In[6]:


get_ipython().system('head -n 5 auto-mpg.data | cat -evt')


# これより、最後のカラムの前にタブがあるのがわかります。少々わかりにくいですが、^I がタブの目印になります。
# 
# これだと、区切り文字が空白とタブが混在しているので、タブを空白に置換します。出来れば、sedでタブを置換したいのですが、sedの挙動がmacOSとlinuxで異なるので、やや冗長ですが、一度中間ファイルを作成します。実際のタブの置換はtrを利用します。

# In[7]:


get_ipython().system("cat auto-mpg.data | tr '\\t' ' ' >> temp.data")
get_ipython().system('mv temp.data auto-mpg.data && rm -f temp.data')
get_ipython().system('head -n 5 auto-mpg.data | cat -evt')


# 最後のコマンドでタブの有無を確認すると、確かにタブが消えています。ここでようやく準備が整いました。このファイルをpandasを用いて読み込みます。その際、column名を指定します。
# 
# ここまでの流れは面倒と感じるかもしれませんが、データ分析の仕事をしているとデータ分析コンテストのように整然としたデータがそろっていることの方が珍しいです。データを整える前処理も重要な仕事です。それらにはlinuxのコマンドを使いこなすことが重要です。

# In[8]:


column_names = ['mpg','cylinders','displacement','horsepower','weight',
                'acceleration', 'model year', 'origin', 'car name'] 

df = pd.read_csv('./auto-mpg.data', 
                 names=column_names,
                 sep=' ',
                 skipinitialspace=True)


# In[9]:


df.head()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import seaborn as sns

iris = sns.load_dataset("iris")
sns.pairplot(iris)


# In[11]:


sns.pairplot(iris, hue="species")


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
