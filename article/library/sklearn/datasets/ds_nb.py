
# coding: utf-8

# ## sickit-learn データセットの使い方
# scikit-learnは機械学習、データ分析には必須のライブラリです。ここではデフォルトでscikit-learnに付随されているデータセットの使い方をメモしておきます。
# 
# ### sickit-learn 目次
# 
# 1. [公式データセット](/article/library/sklearn/datasets/) <= 本節
# 2. [データの作成](/article/library/sklearn/makedatas/)
# 3. [線形回帰](/article/library/sklearn/linear_regression/)
# 4. [ロジスティック回帰](/article/library/sklearn/logistic_regression/)
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sklearn/datasets/ds_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sklearn/datasets/ds_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import sklearn

sklearn.__version__


# データ表示用にpandasもimportしておきます。

# In[4]:


import pandas as pd

pd.__version__


# 画像表示用にmatplotlibもimportします。画像はwebでの見栄えを考慮して、svgで保存する事とします。

# In[5]:


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib.pyplot as plt


# ## 概要
# scikit-learnは機械学習に必要なデータセットを用意してくれています。ここでは公式サイトにそってサンプルデータの概要を説明します。
# 
# 1. toy dataset
# 2. 実際のデータセット
# 
# ## toy datasets
# 
# toyというのは、おそらく簡易的なデータで、実際の機械学習のモデル生成には不十分な量という意味だと思います。
# 
# ### boston住宅価格のデータ
# 
# - target: 住宅価格
# - 回帰問題

# In[6]:


from sklearn.datasets import load_boston

boston = load_boston()


# 最初なので少し丁寧にデータを見ていきます。

# In[7]:


type(boston)


# データタイプはsklearn.utils.Bunch型だとわかります。

# In[8]:


dir(boston)


# DESCR, data, feature_names, filename, targetのプロパティを持つ事がわかります
# 一つ一つの属性値を見ていきます。DESCRは、データに関する説明、filenameはデータのファイルの絶対パスなので省略します。

# #### boston.data
# 実際に格納されているデータです。分析対象とする各特徴量が格納されています。説明変数とも言うようです。

# In[9]:


boston.data


# #### boston.feature_names
# 各特徴量の名前です。

# In[10]:


boston.feature_names


# #### boston.target
# 予測するターゲットの値です。公式サイトによるとbostonの場合は価格の中央値（Median Value）となります。

# In[11]:


boston.target


# pandasで読み込みます。

# In[12]:


df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df['MV'] = pd.DataFrame(data=boston.target)

df.shape


# In[13]:


df.head()


# となり、データ数が506個である事がわかります。また、各特徴量の統計量は以下の通りです。

# In[14]:


df.describe()


# ### アヤメのデータ
# 
# - target: アヤメの種類
# - 分類問題

# In[15]:


from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))
print(dir(iris))


# In[16]:


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['IRIS'] = pd.DataFrame(data=iris.target)
df.shape


# In[17]:


df.head()


# 最初の5個のデータだと0しかないので、ランダムサンプリングしてみると以下のようになります。

# In[18]:


df.sample(frac=1, random_state=0).reset_index().head()


# 各特徴量は以下の通りです。日本語に訳しましたが、あまりぴんと来ませんね。
# 
# <div class="table_center_45">
# 
# |英語名 |日本名|
# |:---:|:---:|
# |sepal length |がく片の長さ  |
# |sepal width | がく片の幅 |
# |petal length | 花びらの長さ |
# |petal width  | 花びらの幅 |
# 
# </div>
# 
# また、IRISというターゲットの値は0,1,2となっており、それらは`iris.target_names`で確認する事ができます。

# In[19]:


iris.target_names


# このリストのインデックスと対応しており、表にすると以下の様になります。
# 
# <div class="table_center_30">
# 
# |index |IRIS|
# |:---:|:---:|
# |0 |setosa  |
# |1 | versicolor |
# |2 | virginica |
# 
# </div>

# ### 糖尿病患者のデータ
# 
# - target: 基準時から糖尿病の状態
# - 回帰問題

# In[20]:


from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print(type(diabetes))
print(dir(diabetes))
print(diabetes.feature_names)
print(diabetes.data.shape)


# In[21]:


df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['QM'] = diabetes.target # QM : quantitative measure 
df.head()


# ### 手書きデータ
# 
# - target:0~9までの数字
# - 分類問題
# 
# データは`digits.images`と`digits.data`の中に入っていますが、imagesは二次元配列でdataは8x8の一次元配列で格納されています。

# In[22]:


from sklearn.datasets import load_digits

digits = load_digits()

print(type(digits))
print(dir(digits))
print(digits.data.shape)
print(digits.images.shape)
print(digits.target_names)


# 一番最初に格納されているデータは以下の様になっています。

# In[23]:


print(digits.images[0])


# `digits.images[0]`を画像化してみます。

# In[24]:


plt.imshow(digits.images[0], cmap='gray')
plt.grid(False)
plt.colorbar()


# 何となく0に見えますね。色合いを変えてみます。

# In[25]:


plt.imshow(digits.images[0])
plt.grid(False)
plt.colorbar()


# グレースケールより見やすいでしょうか？変わらないですかね･･･
# もちろん、これらのデータに対して、正解のデータが与えられています。

# In[26]:


print(digits.target[0])


# ### 生理学的データと運動能力のデータ
# 
# - target: 生理学的データ（体重、ウェスト、脈拍） (日本語訳の不正確かもしれません)
# - 回帰問題
# 
# 運動能力から体重やウェストなどの身体的特徴を求める問題
# 

# In[27]:


from sklearn.datasets import load_linnerud

linnerud = load_linnerud()

print(type(linnerud))
print(dir(linnerud))


# In[28]:


df1 = pd.DataFrame(data=linnerud.data, columns=linnerud.feature_names)
df2 = pd.DataFrame(data=linnerud.target, columns=linnerud.target_names)


# In[29]:


df1.head()


# In[30]:


df2.head()


# ### ワインのデータ
# 
# - target: ワインの種類
# - 分類問題

# In[31]:


from sklearn.datasets import load_wine

wine = load_wine()

print(type(wine))
print(dir(wine))
print(wine.feature_names)

print(wine.target)
print(wine.target_names)


# pandasで読み込んでみます。ターゲットの名前をWINEとして、`wine.target`を追加します。

# In[32]:


df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['WINE'] = pd.DataFrame(data=wine.target)
df.head()


# 先頭から5個のサンプリングだとWINEの列がすべて0になってしまったので、ランダムサンプリングしてみます。

# In[33]:


df.sample(frac=1, random_state=0).reset_index().head()


# ### 乳がんのデータ
# 
# - target: がんの良性/悪性
# - 分類問題

# In[40]:


from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()

print(type(bc))
print(dir(bc))
print(bc.feature_names)
print(bc.target_names)


# 属性がかなり多いです。悪性が良性かの分類問題です。pandasで読み込んでみます。

# In[41]:


df = pd.DataFrame(data=bc.data, columns=bc.feature_names)
df['MorB'] = pd.DataFrame(data=bc.target) # MorB means maligant or benign
df.head()


# ランダムサンプリングしてみます。

# In[42]:


df.sample(frac=1, random_state=0).reset_index().head()


# 良性の陰性の結果がMorBに見て取れます。

# ## 参考資料
# - [scikit-learn 公式ページ](https://scikit-learn.org/stable/datasets/index.html)
