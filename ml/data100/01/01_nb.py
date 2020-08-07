#!/usr/bin/env python
# coding: utf-8

# ## 第1章 ウェブからの注文数を分析する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/01/01_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/01/01_nb.ipynb)
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

# ### 共通部分

# In[4]:


import pandas as pd


# ### ノック 1 : データを読み込んでみよう

# In[5]:


get_ipython().system('ls -a | grep csv')


# In[6]:


get_ipython().system('head -n 5 customer_master.csv')


# In[7]:


master = pd.read_csv('customer_master.csv')
master.head()


# In[8]:


item = pd.read_csv('item_master.csv')
item.head()


# In[9]:


transaction_1 = pd.read_csv('transaction_1.csv')
transaction_1.head()


# In[10]:


transaction_2 = pd.read_csv('transaction_2.csv')
transaction_2.head()


# In[11]:


transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
transaction_detail_1.head()


# ### ノック 2 : データを結合（ユニオン）してみよう
# データを縦方向に連結します。

# In[12]:


transaction_1 = pd.read_csv('transaction_1.csv')
transaction_1.tail()


# In[13]:


transaction_2 = pd.read_csv('transaction_2.csv')
transaction_2.head()


# transaction_1のtailとtrasaction_2が続いてる様に見えます。
# データサイズを確認します

# In[14]:


print(transaction_1.shape)
print(transaction_2.shape)


# In[15]:


transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)
transaction.head()


# In[16]:


transaction.tail()


# In[17]:


print(transaction.shape)


# transaction_detailも結合します。

# In[18]:


transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
transaction_detail_1.head()


# In[19]:


transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')
transaction_detail_2.head()


# In[20]:


transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index=True)
transaction_detail.head()


# In[21]:


transaction_detail.shape


# ### ノック 3 : 売上データ同士を結合してみよう

# 横方向の結合（ジョイン）します。

# In[22]:


join_data = pd.merge(transaction_detail, transaction[['transaction_id', 'payment_date', 'customer_id']], on='transaction_id', how='left')
join_data.head()


# In[23]:


join_data.shape


# ### ノック 4 : マスターデータを結合してみよう

# In[24]:


join_data = pd.merge(join_data, master, on='customer_id', how='left')
join_data = pd.merge(join_data, item, on='item_id', how='left')
join_data.head()


# ### ノック 5 : 必要なデータ列を作ろう

# In[25]:


join_data['price'] = join_data['quantity'] * join_data['item_price']
join_data.head()


# ### ノック 6 : データ検算をしよう

# In[26]:


join_data['price'].sum() == transaction['price'].sum()


# ### ノック 7 : 各種統計量を把握しよう

# In[27]:


join_data.isnull().sum()


# In[28]:


join_data.describe()


# ### ノック 8 : 月別でデータを集計してみよう

# In[29]:


join_data.dtypes


# payment_dataがobject型となっているので、datetime型に変更して、年ごとや月ごとの集計をしやすくします。

# In[30]:


join_data['payment_date'] = pd.to_datetime(join_data['payment_date'])
join_data['payment_month'] = join_data['payment_date'].dt.strftime('%Y%m')
join_data[['payment_date', 'payment_month']].head()


# In[31]:


join_data.groupby('payment_month').sum()['price']


# ### ノック 9 : 月別、商品別でデータを集計してみよう

# In[32]:


join_data.groupby(['payment_month', 'item_name']).sum()[['price','quantity']]


# pivot_tableを利用して、見やすくします。

# In[33]:


pd.pivot_table(join_data, index='item_name', columns='payment_month', values=['price','quantity'], aggfunc='sum')    


# ### ノック 10 : 商品別の売上推移を可視化してみよう

# In[34]:


graph_data = pd.pivot_table(join_data, index='payment_month', columns='item_name', values='price', aggfunc='sum')   
graph_data.head()


# In[35]:


plt.plot(list(graph_data.index), graph_data['PC-A'], label='PC-A')    
plt.plot(list(graph_data.index), graph_data['PC-B'], label='PC-B')    
plt.plot(list(graph_data.index), graph_data['PC-C'], label='PC-C')    
plt.plot(list(graph_data.index), graph_data['PC-D'], label='PC-D')    
plt.plot(list(graph_data.index), graph_data['PC-E'], label='PC-E')    

plt.grid()
plt.legend()
plt.show()


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
