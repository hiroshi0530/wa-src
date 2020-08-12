#!/usr/bin/env python
# coding: utf-8

# ## 第5章 顧客の退会を予測する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 本演習で利用しているデータは本サイトからは利用できません。ぜひとも「Python実践データ分析１００本ノック」を購入し、本に沿ってダウンロードして自分の手でコーディングしてみてください。（私は決して回し者ではないので安心してください笑）
# 
# 前章（4章）では、クラスタリングと線形回帰をして見ました。今回は決定木のようです。データ分析や予測において最初に使われるのがほとんど決定木を基本とする、XGBoostやLightGBM、ランダムフォレストの元となっています。
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

# ### ノック : 41 データを読み込んで利用で他を整形しよう

# In[4]:


customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')


# In[7]:


customer.head()


# In[8]:


uselog_months.head()


# In[9]:


year_months = list(uselog_months['年月'].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
  tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]]
  tmp.rename(columns={'count': 'count_0'}, inplace=True)
  tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i-1]]
  del tmp_before['年月']
  tmp_before.rename(columns={'count': 'count_1'}, inplace=True)
  tmp = pd.merge(tmp, tmp_before, on='customer_id', how='left')
  uselog = pd.concat([uselog, tmp], ignore_index=True)


# In[12]:


uselog.head()


# In[13]:


uselog.shape


# ### ノック : 42 大会前日の大会顧客データを作成しよう

# In[24]:


from dateutil.relativedelta import relativedelta
exit_customer = customer.loc[customer['is_deleted'] == 1]

exit_customer.head()


# In[32]:


exit_customer['exit_date'] = None
exit_customer['end_date'] = pd.to_datetime(exit_customer['end_date'])

for i in range(len(exit_customer)):
  exit_customer['exit_date'].iloc[i] = exit_customer['end_date'].iloc[i] - relativedelta(months=1)

exit_customer['年月'] = exit_customer['exit_date'].dt.strftime('%Y%m')
uselog['年月'] = uselog['年月'] .astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=['customer_id', '年月'], how='left')


# In[33]:


len(uselog)


# In[34]:


exit_uselog.head()


# ### ノック : 43 継続顧客のデータを作成しよう

# In[ ]:





# ### ノック : 44 予測する月の在籍期間を作成しよう

# In[ ]:





# ### ノック : 45 欠損値を除去しよう

# In[ ]:





# ### ノック : 46 文字列型の変数を処理できるように整形しよう

# In[ ]:





# ### ノック : 47 決定木を用いて大会予測モデルを作成してみよう

# In[ ]:





# ### ノック : 48 予測モデルの評価を行い、モデルのチューニングをしてみよう

# In[ ]:





# ### ノック : 49 モデルに寄与している変数を確認しよう

# In[ ]:





# ### ノック : 50 顧客の退会を予測しよう

# In[ ]:





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
