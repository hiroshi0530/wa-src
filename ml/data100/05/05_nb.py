
# coding: utf-8

# ## 第5章 顧客の退会を予測する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 本演習で利用しているデータは本サイトからは利用できません。ぜひとも「Python実践データ分析１００本ノック」を購入し、本に沿ってダウンロードして自分の手でコーディングしてみてください。（私は決して回し者ではないので安心してください笑）
# 
# 前章（4章）では、クラスタリングと線形回帰を実行してみました。今回は決定木のようです。データ分析や予測において最初に使われるのがXGBoostやLightGBM、ランダムフォレストであり、それらの手法の基礎となっているのが決定木です。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[1]:


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

# ### ノック : 41 データを読み込んで利用で他を整形しよう

# In[97]:


customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')


# In[3]:


customer.head()


# In[4]:


uselog_months.head()


# In[18]:


year_months = list(uselog_months['年月'].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
  tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]].copy()
  tmp.rename(columns={'count': 'count_0'}, inplace=True)
  tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i - 1]].copy()
  del tmp_before['年月']
  tmp_before.rename(columns={'count': 'count_1'}, inplace=True)
  tmp = pd.merge(tmp, tmp_before, on='customer_id', how='left')
  uselog = pd.concat([uselog, tmp], ignore_index=True)


# In[8]:


uselog.head()


# ### ノック : 42 大会前日の大会顧客データを作成しよう

# In[135]:


from dateutil.relativedelta import relativedelta
exit_customer = customer.loc[customer['is_deleted'] == 1].copy()

exit_customer['exit_date'] = None
exit_customer['end_date'] = pd.to_datetime(exit_customer['end_date'].copy())

for i in range(len(exit_customer)):
  col_id_exit_date = exit_customer.columns.get_loc('exit_date')
  col_id_end_date = exit_customer.columns.get_loc('end_date')
  # print(col_idx)
  # print(exit_customer.iloc[[i], [col_id_end_date]])
  # print(type(exit_customer.iloc[[i], [col_id_end_date]]))
  # print(exit_customer.iloc[i, col_id_end_date])
  # exit_customer.iloc[[i], [col_id_exit_date]] = exit_customer.iloc[[i], [col_id_end_date]] - relativedelta(months=1)
  exit_customer.iloc[i, col_id_exit_date] = exit_customer.iloc[i, col_id_end_date] - relativedelta(months=1)
  # print(type(exit_customer.iloc[[i], [col_id_end_date]]))
  # print(type(exit_customer.iloc[i, col_id_end_date]))
  
# for i in range(len(exit_customer)):
# print(exit_customer['exit_date'])
# print(relativedelta(months=1) + exit_customer['end_date'].iloc[0])
# print(exit_customer['exit_date'])
# print(type(exit_customer['exit_date']))
# exit_customer['exit_date'].loc[0] = 2
# # print(exit_customer['exit_date'].loc[0])
# print(exit_customer['exit_date'].iloc[0])
# print(exit_customer['end_date'].iloc[0])
# # exit_customer['end_date'].iloc[0] = 100
# print(exit_customer.columns)
# print(type(exit_customer))
# print()
# print()
# print()
# # exit_customer.iloc[[1, 2], ['exit_date']] = 40
# print(exit_customer.iloc[[1, 2], [0]])
# 
# 
# col_idx = exit_customer.columns.get_loc('exit_date')
# 
# print(col_idx)
# print(col_idx)
# print(col_idx)
# 
# exit_customer.iloc[[1, 2], [col_idx]] = 100

# exit_customer['exit_date'][0] = 1
# exit_customer['exit_date'].iloc[0] = exit_customer['end_date'].iloc[0] - relativedelta(months=1)

# exit_customer[0,'exit_date'] = 1


exit_customer['年月'] = exit_customer['exit_date'].dt.strftime('%Y%m')
uselog['年月'] = uselog['年月'].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=['customer_id', '年月'], how='left')


# In[136]:


len(uselog)


# In[137]:


exit_uselog.head()


# In[141]:


exit_uselog = exit_uselog.dropna(subset=['name'])
print(len(exit_uselog))
print(len(exit_uselog['customer_id'].unique()))
exit_uselog.head()


# ### ノック : 43 継続顧客のデータを作成しよう

# In[14]:


conti_customer = customer.loc[customer['is_deleted'] == 0]
conti_uselog = pd.merge(uselog, conti_customer, on=['customer_id'], how='left')


# In[15]:


conti_uselog.head()


# In[16]:


conti_uselog = conti_uselog


# In[17]:


conti_uselog.head()


# ### ノック : 44 予測する月の在籍期間を作成しよう

# In[19]:


predict_data['period'] = 0
predict_data['now_date'] = pd.to_datetime(predict_data['年月'], format="%Y%m")

for i in range(5):
  print(i, ' : ABC')


# ### ノック : 45 欠損値を除去しよう

# ### ノック : 46 文字列型の変数を処理できるように整形しよう

# ### ノック : 47 決定木を用いて大会予測モデルを作成してみよう

# ### ノック : 48 予測モデルの評価を行い、モデルのチューニングをしてみよう

# ### ノック : 49 モデルに寄与している変数を確認しよう

# ### ノック : 50 顧客の退会を予測しよう

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
