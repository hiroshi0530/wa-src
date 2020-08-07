
# coding: utf-8

# ## 第2章 小売店のデータでデータ加工を行う10本ノック
# 
# 第２章では小売店の売り上げデータの解析、予測になります。汚いデータをいかに加工して予測のモデルを構築をして行くかという演習になっています。
# 
# こういう実務的な問題を演習として用意してくれているので、とてもありがたいです。
# 
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/02/02_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/02/02_nb.ipynb)
# 
# ### 筆者の環境

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
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)


# ## 解答

# ### ノック11 : データを読み込んでみよう

# In[4]:


uriage_data = pd.read_csv('uriage.csv')
uriage_data.head()


# 途中で半角スペースが入っていたり、大文字、小文字の差異などいい感じに表記が揺れています。一応カラム名を確認しておきます。

# In[13]:


uriage_data.columns


# In[5]:


kokyaku_data = pd.read_excel('kokyaku_daicho.xlsx')
kokyaku_data.head()


# In[16]:


kokyaku_data.columns


# ### ノック12 : データの揺れを見てみよう

# In[6]:


uriage_data['item_name'].head()


# In[7]:


uriage_data['item_price'].head()


# かなりデータが揺れがあるのがわかります。

# ### ノック13 : データに揺れがあるまま集計してみよう 
# 
# 日付がobject型なのでdatetime型に変換します。

# In[8]:


uriage_data['purchase_date'] = pd.to_datetime(uriage_data['purchase_date'])


# In[22]:


uriage_data[['purchase_date']].head()


# daetime型に変換されています。
# 
# 月ごとの集計値を計算します。

# In[10]:


uriage_data['purchase_month'] = uriage_data['purchase_date'].dt.strftime('%Y%m')
res = uriage_data.pivot_table(index='purchase_month', columns='item_name', aggfunc='size', fill_value=0)
res


# In[11]:


res.shape


# 商品数が99個になっています。
# 次に価格の集計についても見てみます。

# In[12]:


res = uriage_data.pivot_table(index='purchase_month', columns='item_name', values='item_price', aggfunc='sum', fill_value=0)
res


# こちらも全く意味をなしてないことがわかります。

# ### ノック14 : 
# 
# まずは商品名の揺れを補正していくようです。今回抽出された商品の一覧です。

# In[24]:


pd.unique(uriage_data['item_name'])


# 商品数は９９個です。

# In[25]:


len(pd.unique(uriage_data['item_name']))


# スペースの有無、半角全角統一をします。文字列を扱うメソッド`str`を利用します。

# In[32]:


uriage_data['item_name'] = uriage_data['item_name'].str.upper()
uriage_data['item_name'] = uriage_data['item_name'].str.replace("　","")
uriage_data['item_name'] = uriage_data['item_name'].str.replace(" ","")
uriage_data.sort_values(by=['item_name'], ascending=True).head(3)


# In[30]:


uriage_data['item_name'].unique()


# In[31]:


len(uriage_data['item_name'].unique())


# となり、商品名の揺れは解消されました。

# ### ノック15 :  金額欠損値の補完をしよう
# 
# 金額にどれぐらいの欠損値（Null)があるか確認しています。

# In[36]:


uriage_data.isnull().sum()


# In[37]:


uriage_data.shape


# 行数が2999で、387行が欠損値で約１２％が欠落していることになります。

# 教科書では、

# In[38]:


uriage_data.isnull().any(axis=0)


# とデータの欠損の有無を確認しています。

# In[40]:


uriage_data.isnull().head()


# anyメソッド、引数のオプションにaxisを指定することで、一つでもNullがあればTrueを返し、Nullの有無を確認出来ます。

# In[41]:


uriage_data.isnull().any(axis=0)


# In[45]:


flg_is_null = uriage_data['item_price'].isnull()


# In[46]:


uriage_data.loc[flg_is_null, 'item_name'].unique()


# In[48]:


flg_is_null = uriage_data['item_price'].isnull()

for trg in list(uriage_data.loc[flg_is_null, 'item_name'].unique()):
  price = uriage_data.loc[(~flg_is_null) & (uriage_data['item_name'] == trg), 'item_price'].max()
  uriage_data['item_price'].loc[(flg_is_null) & (uriage_data['item_name'] == trg)] = price
  
uriage_data.head()


# In[51]:


uriage_data['item_price'].isnull().any(axis=0)
uriage_data.isnull().any(axis=0)


# ### ノック16 : 顧客名の揺れを補正しよう
# 
# 次は文字列の揺れの修正になります。

# In[52]:


kokyaku_data['顧客名'].head()


# こちらもスペースが有無に違いがある事尾がわかります。

# In[53]:


uriage_data['customer_name'].head()


# こちらはスペースがありません。こういった名前の表式の差異は、名前が一意キーとなっている場合ｌ、ジョインできないなどの大きいな問題になります。

# こちらは、スペースを削除する方向でデータの補正を行います。

# In[54]:


kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace(" ","")
kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace("　","")
kokyaku_data['顧客名'].head()


# 事務レベルでは名前の補正は本当にやっかいです。このほかにも、同姓同名の処理や、漢字が同じなのに読み方が異なる場合など、大変です。

# ### ノック17 : 日付の揺れを補正しよう
# 
# 次に日付の揺れを補正します。教科書の場合は、日付の表記が異なっていたり、エクセルの設定により日付が数字になっていたりするので、それを補正します。
# 
# まずは数値として取り込まれているデータを取得します。

# In[58]:


flg_is_series = kokyaku_data['登録日'].astype('str').str.isdigit()

flg_is_series.sum()


# ### ノック18 : 顧客名をキーに二つのデータを結合しよう

# ### ノック19 : クレンジングしたデータをダンプしよう

# ### ノック20 : データを集計しよう

# ## 関連記事
# - [第1章 ウェブからの注文数を分析する10本ノック](/ml/data100/01/)
# - [第2章 小売店のデータでデータ加工を行う10本ノック](/ml/data100/02/)
# - [第3章 顧客の全体像を把握する10本ノック](/ml/data100/03/)
# - [第4章 顧客の行動を予測する10本ノック](/ml/data100/04/)
# - [第5章 顧客の退会を予測する10本ノック](/ml/data100/05/)
# - [第6章 物流の最適ルートをコンサルティングする10本ノック](/ml/data100/06/)
# - [第7章 ロジスティクスネットワークの最適設計を行う10本ノック](/ml/data100/07/)
# - [第8章 数値シミュレーションで消費者行動を予測する10本ノック](/ml/data100/08/)
# - [第9章 潜在顧客を把握するための画像認識10本ノック](/ml/data100/09/)
# - [第10章 アンケート分析を行うための自然言語処理10本ノック](/ml/data100/10/)
