#!/usr/bin/env python
# coding: utf-8

# ## 第3章 顧客の全体像を把握する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 本演習で利用しているデータは本サイトからは利用できません。ぜひとも「Python実践データ分析１００本ノック」を購入し、本に沿ってダウンロードして自分の手でコーディングしてみてください。（私は決して回し者ではないので安心してください笑）
# 
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/03/03_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/03/03_nb.ipynb)
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

# ### ノック 21 : データを読み込んで把握しよう
# 
# 利用履歴、顧客情報、会員区分、キャンディーズの各テーブルを読み込みます。

# In[4]:


uselog = pd.read_csv('use_log.csv') # 利用履歴
customer = pd.read_csv('customer_master.csv') # 顧客情報
class_master = pd.read_csv('class_master.csv') # 会員区分
campaign_master = pd.read_csv('campaign_master.csv') # キャンペーン区分

uselog.head()


# In[5]:


uselog.shape


# In[6]:


customer.head()


# In[7]:


customer.shape


# In[8]:


class_master.head()


# In[9]:


class_master.shape


# In[10]:


campaign_master.head()


# In[11]:


campaign_master.shape


# ### ノック 22 : 顧客データを整形しよう
# 
# 顧客テーブルに会員区分とキャンペーン区分を追加します。

# In[12]:


customer_join = pd.merge(customer, class_master, on='class', how='left')
customer_join = pd.merge(customer_join, campaign_master, on='campaign_id', how='left')
customer_join.head()


# ### ノック 23 : 顧客データの基礎集計をしよう 
# 
# 会員区分やキャンペーン区分ごとの人数や、入会、退会の期間、男女比率などを集計してみます。

# In[13]:


customer_join.groupby('class_name').count()[['customer_id']]


# In[14]:


customer_join.groupby('campaign_name').count()[['customer_id']]


# In[15]:


customer_join.groupby('gender').count()[['customer_id']]


# In[16]:


customer_join.groupby('is_deleted').count()[['customer_id']]


# In[17]:


customer_join['start_date'] = pd.to_datetime(customer_join['start_date'])
customer_start = customer_join.loc[customer_join['start_date'] > pd.to_datetime('20180401')]
customer_start.head()
customer_start.shape


# ### ノック 24 : 最新顧客データの基礎集計をしてみよう
# 
# 新規顧客の条件抽出をdatetime型を用いて抽出しています。
# 
# ※NaTはdatetime型のNullです。

# In[18]:


customer_join['end_date'] = pd.to_datetime(customer_join['end_date'])
customer_newer = customer_join.loc[(customer_join['end_date'] >= pd.to_datetime('20190331')) | (customer_join['end_date'].isna())]
customer_newer['end_date'].unique()


# In[19]:


customer_newer.shape


# In[20]:


customer_newer.groupby('class_name').count()['customer_id']


# In[21]:


customer_newer.groupby('campaign_name').count()['customer_id']


# In[22]:


customer_newer.groupby('gender').count()['customer_id']


# ### locの復習
# 
# 少しここでlocの使い方を復習して追います。locは行と列に対して、条件を指定して表示させる事が出来ます。
# 
# 最初にデータを用意しておきます。

# In[23]:


_ = np.arange(12).reshape(3,4)
df = pd.DataFrame(_, columns=['a', 'b', 'c', 'd'])
df

行に対して、indexが2で割ったあまりが0の行だけ抽出するように条件を設定します。
# In[24]:


df.loc[df.index % 2 == 0]


# 当然ですが、否定演算子も指定出来ます。

# In[25]:


df.loc[~(df.index % 2 == 0)]


# and や or 演算も可能です。

# In[26]:


df.loc[(df.index % 2 == 0) & (df.index % 3 == 0)]


# In[27]:


df.loc[(df.index % 2 == 0) | (df.index % 3 == 1)]


# このように条件で抽出したものをもう一度結合したいときは、concatメソッドを利用します。

# In[28]:


pd.concat([df.loc[df.index % 2 == 0], df.loc[df.index % 2 != 0]])


# ### ノック 25 : 利用履歴データを集計しよう
# 
# 次に利用履歴です。こちらは時系列解析になります。

# In[29]:


uselog['usedate'] = pd.to_datetime(uselog['usedate'])
uselog['年月'] = uselog['usedate'].dt.strftime('%Y%m')
uselog_months = uselog.groupby(['年月', 'customer_id'], as_index=False).count()
uselog_months.head()


# In[30]:


uselog_months.rename(columns={'log_id':'count'}, inplace=True)


# In[31]:


uselog_months.head()


# In[32]:


uselog_months.drop('usedate', axis=1).head()


# In[33]:


uselog_customer = uselog_months.groupby('customer_id').agg(['mean', 'median', 'max', 'min'])['count']
uselog_customer = uselog_customer.reset_index(drop=False)
uselog_customer.head()


# groupbyからのaggメソッドによる統計値の出力になっています。勉強になります。

# ### ノック 26 : 利用履歴データから定期利用フラグを作成しよう

# In[34]:


uselog['weekday'] = uselog['usedate'].dt.weekday
uselog_weekday = uselog.groupby(['customer_id', '年月', 'weekday'], as_index = False).count()[['customer_id', '年月', 'weekday', 'log_id']]
uselog_weekday.rename(columns={'log_id':'count'}, inplace=True)
uselog_weekday.head()


# In[35]:


uselog_weekday = uselog_weekday.groupby('customer_id', as_index=False).max()[['customer_id', 'count']]
uselog_weekday['routine_flg'] = 0
uselog_weekday['routine_flg'] = uselog_weekday['routine_flg'].where(uselog_weekday['count'] < 4,1)
uselog_weekday.head()


# ### ノック 27 : 顧客データと利用履歴データを結合しよう
# 
# これまでのデータを結合するだけです。

# In[36]:


customer_join = pd.merge(customer_join, uselog_customer, on='customer_id', how='left')
customer_join = pd.merge(customer_join, uselog_weekday[['customer_id', 'routine_flg']], on='customer_id', how='left')
customer_join.head()


# ### ノック 28 : 会員期間を計算しよう
# 
# 日付の比較のためにrelativedeltaを利用します。退会の日付データない場合は2019年4月30日の日付で埋めます。

# In[37]:


from dateutil.relativedelta import relativedelta
customer_join['calc_data'] = customer_join['end_date']
customer_join['calc_data'] = customer_join['end_date'].fillna(pd.to_datetime('20190430'))
customer_join['membership_period'] = 0

for i in range(len(customer_join)):
  delta = relativedelta(customer_join['calc_data'].iloc[i], customer_join['start_date'].iloc[i])
  customer_join['membership_period'].iloc[i] = delta.years * 12 + delta.months
  
customer_join.head()


# ### ノック 29 : 顧客行動の各種統計量を把握しよう

# In[38]:


customer_join[['mean', 'median', 'max', 'min']].describe()


# In[39]:


customer_join.groupby('routine_flg').count()['customer_id']


# In[40]:


plt.hist(customer_join['membership_period'])
plt.show()


# ### ノック 30 : 退会ユーザーと継続ユーザーの違いを把握しよう

# In[41]:


customer_end = customer_join.loc[customer_join['is_deleted'] == 1]
customer_end.describe()


# In[42]:


customer_end = customer_join.loc[customer_join['is_deleted'] == 0]
customer_end.describe()


# In[43]:


customer_join.to_csv('customer_join.csv', index=False)


# In[44]:


get_ipython().system('head -n 5 customer_join.csv')


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
