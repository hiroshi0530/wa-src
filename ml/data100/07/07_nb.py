
# coding: utf-8

# ## 第7章 ロジスティクスネットワークの最適設計を行う10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 輸送問題の最適化という事で、pulpとortoolpyを利用します。
# 私も今回初めて利用するので勉強させていただきます。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/07/07_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/07/07_nb.ipynb)
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

# ### ノック 61 : 輸送最適化問題を解いてみよう
# 
# 実際にpulpとortoolpyを読み込んで最適化問題を解きます。

# In[4]:


from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min, addvars, addvals

df_tc = pd.read_csv('trans_cost.csv', index_col='工場')
df_tc


# In[5]:


df_demand = pd.read_csv('demand.csv')
df_demand


# In[6]:


df_supply = pd.read_csv('supply.csv')
df_supply


# In[7]:


# 初期設定
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)

# index, columnsの数に対して直積を作成します
pr = list(product(range(nw), range(nf)))

print(nw)
print(nf)
print(pr)


# In[8]:


# 数理モデルの作成
m1 = model_min()
v1 = {(i,j): LpVariable('v%d_%d'%(i,j), lowBound=0) for i,j in pr}


# In[13]:


m1 += lpSum(df_tc.iloc[i][j] * v1[i,j] for i,j in pr)
for i in range(nw):
  m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]

for j in range(nf):
  m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]

m1.solve()


# ### ノック 62 : 最適輸送ルートをネットワークで確認しよう

# ### ノック 63 : 最適輸送ルートが制約条件内に収まっているか確認してみよう

# ### ノック 64 : 生産計画に関するデータを読み込んでみよう

# ### ノック 65 : 利益を計算する関数を作って見よう

# ### ノック 66 : 生産最適化問題を問いてみよう

# ### ノック 67 : 最適生産計画が制約条件内に長待て散るどうかを確認しよう

# ### ノック 68 : ロジスティックネットワーク設計問題を解いてみよう

# ### ノック 69 : 最適ネットワークにおける輸送コストとその内訳を計算しよう

# ### ノック 70 : 最適ネットワークにおける生産コストとその内訳を計算しよう

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
