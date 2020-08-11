#!/usr/bin/env python
# coding: utf-8

# ## 第4章 顧客の行動を予測する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 本演習で利用しているデータは本サイトからは利用できません。ぜひとも「Python実践データ分析１００本ノック」を購入し、本に沿ってダウンロードして自分の手でコーディングしてみてください。（私は決して回し者ではないので安心してください笑）
# 
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# 同じくスポーツジムのデータの例を用いて、クラスタリングや線形回帰など基本的な機械学習手法を実行します。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/04/04_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/04/04_nb.ipynb)
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

# ### ノック 31 : データを読み込んで確認しよう

# In[4]:


uselog = pd.read_csv("use_log.csv")
uselog.head()


# In[5]:


customer = pd.read_csv('customer_join.csv')
customer.head()


# 欠損値の個数を確認します。

# In[6]:


uselog.isnull().sum()


# In[7]:


customer.isnull().sum()


# ### ノック 32 : クラスタリングで顧客をグループ化しよう 
# 
# k-meansという手法を用いてクラスタリングを行います。説明変数として、平均や中央値、最大最小値などを利用します。最初にk-meansに利用する変数を持つテーブルを作成します。

# In[8]:


customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]
customer_clustering.head()


# 実際にk-meanを利用してクラスタリングを実行します。その前に、整数型では標準化の際にwarningが出るので、float型に変換します。

# In[9]:


customer_clustering.info()


# In[10]:


customer_clustering = customer_clustering.astype({'max': float, 'min': float, 'membership_period': float})


# In[11]:


customer_clustering.info()


# となり、型変換が出来ました。

# In[12]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)

customer_clustering_sc


# #### sklearnのstandardscalerについて
# 
# データの標準化です。与えられた数値のデータを平均0、標準偏差1のガウス分布に変換します。
# 
# $ \displaystyle
# x = \frac{x - x_{mean}}{\sigma}
# $
# 
# という変換をしてくれます。

# In[13]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data = np.arange(9).reshape(3,3).astype('float64')
print('before : \n',data)

data = sc.fit_transform(data)
print('after : \n', data)


# 次に実際にクラスタリングしてみます。クラス多数を4に設定します。各クラスタリングに対して0から3のラベルが付与されています。

# In[14]:


kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(customer_clustering_sc)
customer_clustering['cluster'] = clusters.labels_

customer_clustering['cluster'].unique()


# In[15]:


customer_clustering.head()


# ### ノック 33 :クラスタリングの結果を分析しよう 

# In[16]:


customer_clustering.columns = ['月内平均値', '月内中央値', '月内最大値', '月内最小値', '会員期間', 'cluster']

customer_clustering.groupby('cluster').count()


# In[17]:


customer_clustering.groupby('cluster').mean()


# ### ノック 34 : クラスタリングの結果を可視化してみよう
# 
# 主成分分析により時限削減を行い、可視化してみます。主成分分析（PCA)や次元削減につては機械学習の教科書を参考にしてください。

# In[18]:


from sklearn.decomposition import PCA
X = customer_clustering_sc
pca = PCA(n_components=2)
pca.fit(X)

x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)

pca_df['cluster'] = customer_clustering['cluster']


# In[19]:


# 主成分空間上の行列
x_pca


# In[20]:


# pcs_dfのカラム
pca_df.columns


# In[21]:


for i in customer_clustering['cluster'].unique():
  tmp = pca_df.loc[pca_df['cluster'] == i]
  plt.scatter(tmp[0], tmp[1])

plt.grid()
plt.show()


# 綺麗に4次元のデータを2次元に圧縮することが出来ました。素晴らしい。

# ### ノック 35 : クラスタリングの結果を基に退会顧客の傾向を把握しよう
# 
# 4つに分けたクラスターの継続顧客と退会顧客の集計を行います。
# 
# 退会顧客を特定するためにid_deleted列をcustomer/clusteringに追加し、cluster、is_deleted毎に集計を行います。

# In[22]:


customer_clustering = pd.concat([customer_clustering, customer], axis=1)


# In[23]:


customer_clustering.groupby(['cluster', 'is_deleted'], as_index=False).count()[['cluster', 'is_deleted', 'customer_id']]


# In[24]:


customer_clustering.groupby(['cluster', 'routine_flg'], as_index=False).count()[['cluster', 'routine_flg', 'customer_id']]


# ### ノック 36 : 翌日の利用回数予測を行うための準備をしよう
# 
# 教師あり学習の線形回帰の演習になります。

# In[28]:


uselog['usedate'] = pd.to_datetime(uselog['usedate'])
uselog['年月'] = uselog['usedate'].dt.strftime('%Y%m')
uselog_months = uselog.groupby(['年月', 'customer_id'], as_index=False).count()

uselog_months.rename(columns={'log_id':'count'}, inplace=True)

uselog_months.drop(['usedate'], axis=1, inplace=True)
uselog_months.head()


# In[32]:


year_months = list(uselog_months['年月'].unique())
predict_data = pd.DataFrame()

# 2018年10月から2019年3月までの半年間
for i in range(6, len(year_months)):
  tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]]
  tmp.rename(columns={'count': 'count_pred'}, inplace=True)

  for j in range(1,7):
    tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i-j]]
    
    del tmp_before['年月']
    tmp_before.rename(columns={'count': 'count_{}'.format(j-1)}, inplace=True)
    
    tmp = pd.merge(tmp, tmp_before, on='customer_id', how='left')
  
  predict_data = pd.concat([predict_data, tmp], ignore_index=True)  
  
predict_data.head()


# In[33]:


predict_data = predict_data.dropna()
predict_data = predict_data.reset_index(drop=True)

predict_data.head()


# ### ノック 37 : 特徴となる変数を付与しよう

# In[35]:


predict_data = pd.merge(predict_data, customer[['customer_id', 'start_date']], on='customer_id', how='left')
predict_data.head()


# In[36]:


predict_data['now_date'] = pd.to_datetime(predict_data['年月'], format='%Y%m')
predict_data['start_date'] = pd.to_datetime(predict_data['start_date'])

from dateutil.relativedelta import relativedelta

predict_data['period'] = None

for i in range(len(predict_data)):
  delta = relativedelta(predict_data['now_date'][i], predict_data['start_date'][i])
  predict_data['period'][i] = delta.years * 12 + delta.months
  
predict_data.head()


# ### ノック 38 : 来月の利用回数予測モデルを構築しよう

# In[41]:


predict_data = predict_data.loc[predict_data['start_date'] >= pd.to_datetime('20180401')]

from sklearn import linear_model
import sklearn.model_selection

model = linear_model.LinearRegression()

X = predict_data[['count_0', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'period']]
y = predict_data['count_pred']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model.fit(X_train, y_train)


# In[42]:


model.score(X_train, y_train)


# In[43]:


model.score(X_test, y_test)


# ### ノック 39 : モデルに寄与している変数を確認しよう

# In[44]:


coef = pd.DataFrame({'feature_names':X.columns, 'coefficient': model.coef_})
coef


# 直近（count_0)の係数が最大となっています。直近の利用回数が高ければ、次の月の利用する傾向があることがわかります。

# ### ノック 40 : 来月の利用回数を予測しよう

# In[45]:


x1 = [3, 4, 4, 6, 8, 7, 8]
x2 = [2, 2, 3, 3, 4, 6, 8]
x_pred = [x1, x2]


# In[47]:


model.predict(x_pred)


# 最後にCSVのデータを保存します。

# In[48]:


uselog_months.to_csv('use_og_months.csv', index=False)


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
