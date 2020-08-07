#!/usr/bin/env python
# coding: utf-8

# ## 第2章 小売店のデータでデータ加工を行う10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 第２章では小売店の売り上げデータの解析、予測になります。汚いデータをいかに加工して予測のモデルを構築をして行くかという演習になっています。
# 
# こういう実務的な問題を演習として用意してくれているので、とてもありがたいです。
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


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

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

# In[5]:


uriage_data.columns


# In[6]:


kokyaku_data = pd.read_excel('kokyaku_daicho.xlsx')
kokyaku_data.head()


# In[7]:


kokyaku_data.columns


# ### ノック12 : データの揺れを見てみよう

# In[8]:


uriage_data['item_name'].head()


# In[9]:


uriage_data['item_price'].head()


# かなりデータが揺れがあるのがわかります。

# ### ノック13 : データに揺れがあるまま集計してみよう 
# 
# 日付がobject型なのでdatetime型に変換します。

# In[10]:


uriage_data['purchase_date'] = pd.to_datetime(uriage_data['purchase_date'])


# In[11]:


uriage_data[['purchase_date']].head()


# daetime型に変換されています。
# 
# 月ごとの集計値を計算します。

# In[12]:


uriage_data['purchase_month'] = uriage_data['purchase_date'].dt.strftime('%Y%m')
res = uriage_data.pivot_table(index='purchase_month', columns='item_name', aggfunc='size', fill_value=0)
res


# In[13]:


res.shape


# 商品数が99個になっています。
# 次に価格の集計についても見てみます。

# In[14]:


res = uriage_data.pivot_table(index='purchase_month', columns='item_name', values='item_price', aggfunc='sum', fill_value=0)
res


# こちらも全く意味をなしてないことがわかります。

# ### ノック14 : 商品名の揺れを補正しよう
# 
# まずは商品名の揺れを補正していくようです。今回抽出された商品の一覧です。

# In[15]:


pd.unique(uriage_data['item_name'])


# 商品数は９９個です。

# In[16]:


len(pd.unique(uriage_data['item_name']))


# スペースの有無、半角全角統一をします。文字列を扱うメソッド`str`を利用します。

# In[17]:


uriage_data['item_name'] = uriage_data['item_name'].str.upper()
uriage_data['item_name'] = uriage_data['item_name'].str.replace("　","")
uriage_data['item_name'] = uriage_data['item_name'].str.replace(" ","")
uriage_data.sort_values(by=['item_name'], ascending=True).head(3)


# In[18]:


uriage_data['item_name'].unique()


# In[19]:


len(uriage_data['item_name'].unique())


# となり、商品名の揺れは解消されました。

# ### ノック15 :  金額欠損値の補完をしよう
# 
# 金額にどれぐらいの欠損値（Null)があるか確認します。

# In[20]:


uriage_data.isnull().sum()


# In[21]:


uriage_data.shape


# 行数が2999で、387行が欠損値で約１２％が欠落していることになります。

# 教科書では、

# In[22]:


uriage_data.isnull().any(axis=0)


# とデータの欠損の有無を確認しています。

# In[23]:


uriage_data.isnull().head()


# anyメソッド、引数のオプションにaxisを指定することで、一つでもNullがあればTrueを返し、Nullの有無を確認出来ます。

# In[24]:


uriage_data.isnull().any(axis=0)


# このNull値に対して、補完をします。少々複雑ですが、教科書では以下の通りように補完しています。それぞれのitem_nameに対して、Nullではない値の最大値を持ってきて、補完しています。

# In[25]:


flg_is_null = uriage_data['item_price'].isnull()

for trg in list(uriage_data.loc[flg_is_null, 'item_name'].unique()):
  price = uriage_data.loc[(~flg_is_null) & (uriage_data['item_name'] == trg), 'item_price'].max()
  uriage_data['item_price'].loc[(flg_is_null) & (uriage_data['item_name'] == trg)] = price
  
uriage_data.head()


# panasのversionによって、上記の様なワーニングが出ますが、本筋ではないので無視します。興味があったら、pandasのコピーとビューについて調べてみてください。

# ### ノック16 : 顧客名の揺れを補正しよう
# 
# 次は文字列の揺れの修正になります。

# In[26]:


kokyaku_data['顧客名'].head()


# こちらもスペースが有無に違いがある事尾がわかります。

# In[27]:


uriage_data['customer_name'].head()


# こちらはスペースがありません。こういった名前の表式の差異は、名前が一意キーとなっている場合ｌ、ジョインできないなどの大きいな問題になります。

# こちらは、スペースを削除する方向でデータの補正を行います。

# In[28]:


kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace(" ","")
kokyaku_data['顧客名'] = kokyaku_data['顧客名'].str.replace("　","")
kokyaku_data['顧客名'].head()


# 事務レベルでは名前の補正は本当にやっかいです。このほかにも、同姓同名の処理や、漢字が同じなのに読み方が異なる場合など、大変です。

# ### ノック17 : 日付の揺れを補正しよう
# 
# 次に日付の揺れを補正します。教科書の場合は、日付の表記が異なっていたり、エクセルの設定により日付が数字になっていたりするので、それを補正します。
# 
# まずは数値として取り込まれているデータを取得します。

# In[29]:


flg_is_series = kokyaku_data['登録日'].astype('str').str.isdigit()
flg_is_series.head()


# In[30]:


flg_is_series.sum()


# isdigitがtrueである数=日付が数値として読み込まれている数が２２個である事がわかります。内容を見てみると、42xxxという数値が含まれています。

# In[31]:


kokyaku_data['登録日']


# エクセルでの日付が数値になっているのは1900年1月１日からの日数ですので、to_timedeltaメソッドでその日数をdatetime型に変換し、1900年1月１日をto_datetime型に変換し、加算することで現在の日付をdatetime型で取得します。

# In[32]:


fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_series, '登録日'].astype('float'), unit='D') + pd.to_datetime('1900/01/01')
fromSerial


# 次に、数値以外の日付をそのままdatetime型で取得します。

# In[33]:


fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_series,'登録日'])
fromString


# あとはこの二つを連結すれば良いです。

# In[34]:


kokyaku_data['登録日'] = pd.concat([fromSerial, fromString])
kokyaku_data[['登録日']].head()


# 登録年月を取得し、月ごとの登録者数を算出してみます。教科書ではgroupbyを利用してます。また、datetime型をobject型に変換するdtを利用します。

# In[35]:


kokyaku_data['登録年月'] = kokyaku_data['登録日'].dt.strftime('%Y%m')
kokyaku_data['登録年月']


# In[36]:


res = kokyaku_data.groupby('登録年月').count()
res


# 教科書をコピーしているだけだとつまらないので、個人的によく利用するresampleメソッドを利用して、groupbyと同じ処理をしてみます。resampleメソッドでは2017年8月の集計がちゃんと０というように出力されます。インデックスを登録日に変更する必要があります。

# In[37]:


kokyaku_data.set_index('登録日').resample('M').count()[['顧客名']].head(10)


# 最後にエクセル由来に数値型が残っていないことを確認します。登録日のカラムはdatetime型なので一度文字列にしてから、isdigitメソッドを利用します。

# In[38]:


kokyaku_data['登録日'].astype('str').str.isdigit().sum()


# となり、ちゃんとすべての数値がdatetime型に変換されていることがわかりました。

# ### ノック18 : 顧客名をキーに二つのデータを結合しよう

# これまでの売上履歴と顧客台帳をマージします。顧客名を一意キーとして結合します。pandasのmergeメソッドを利用します。

# In[39]:


join_data = pd.merge(uriage_data, kokyaku_data, left_on = 'customer_name', right_on = '顧客名', how='left')
join_data.head()


# 当然ながら顧客名とcustomer_nameが重複するので、customer_nameを削除します。

# In[40]:


join_data = join_data.drop('customer_name', axis=1)
join_data.head()


# このようなデータ加工をクレンジングと言うそうです。知りませんでした。。

# ### ノック19 : クレンジングしたデータをダンプしよう
# 
# 最後にCSVで出力するのですが、必要なカラムだけを出力します。

# In[41]:


dump_data = join_data[['purchase_date', 'purchase_month', 'item_name', 'item_price', '顧客名', 'かな', '地域', 'メールアドレス', '登録日']]
dump_data.head()


# to_csvメソッドを利用してCSVで保存します。

# In[42]:


dump_data.to_csv('dump_data.csv', index=False)


# 確認してみます。

# In[43]:


get_ipython().system('head -n 5 dump_data.csv')


# ### ノック20 : データを集計しよう
# 
# ノック１９で加工したデータを利用して、再度集計処理をしてみます。

# In[44]:


import_data = pd.read_csv('dump_data.csv')
import_data.head()


# purchase_monthに対して商品ごとの集計をします。

# In[45]:


byItem = import_data.pivot_table(index='purchase_month', columns='item_name', aggfunc='size', fill_value=0)
byItem.head()


# とても綺麗なテーブルになっていると思います。次に、売上金額です。

# In[46]:


byPrice = import_data.pivot_table(index='purchase_month', columns='item_name',values='item_price', aggfunc='sum', fill_value=0)
byPrice.head()


# 次に顧客ごとの販売数です。

# In[47]:


Customer = import_data.pivot_table(index='purchase_month', columns='顧客名', aggfunc='size', fill_value=0)
Customer.head()


# 次に地域ごとにおける販売数です。

# In[48]:


Region = import_data.pivot_table(index='purchase_month', columns='地域', aggfunc='size', fill_value=0)
Region.head()


# とても便利で、実用的です。最後に集計期間で購入していない顧客の洗い出しをしています。purchase_dataがNullのユーザーを抽出しています。

# In[49]:


away_data = pd.merge(uriage_data, kokyaku_data, left_on='customer_name', right_on='顧客名', how='right')
away_data[away_data['purchase_date'].isnull()][['顧客名', '登録日']]


# とても有意義な演習でした。実際の現場はもっとデータが汚いかと思いますが、演習の題材としては非常に勉強になるものでした。教科書の著者に感謝申し上げます。

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
