#!/usr/bin/env python
# coding: utf-8

# ## 第10章 アンケート分析を行うための自然言語処理10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# アンケート処理の演習になります。こちらも前の章よりはやりやすかったです。しかしとても勉強になるので、ぜひとも自分のものにしたいです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/10/10_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/10/10_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[ ]:


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

# ### ノック 91 : データを読み込んで把握しよう

# In[ ]:





# ### ノック 92 : 不要な文字を除去してみよう

# In[ ]:





# ### ノック 93 : 文字列をカウントしてヒストグラムを表示してみよう

# In[ ]:





# ### ノック 94 : 形態素解析で文書を解析してみよう

# In[ ]:





# ### ノック 95 : 形態素解析で文章から「動詞・名詞」を抽出してみよう

# In[ ]:





# ### ノック 96 : 形態素解析で抽出した頻出する名詞を確認してみよう

# In[ ]:





# ### ノック 97 : 関係のない単語を除去してみよう

# In[ ]:





# ### ノック 98 : 顧客満足度と頻出単語の関係を見てみよう

# In[ ]:





# ### ノック 99 : アンケート毎の特徴を表現してみよう

# In[ ]:





# ### ノック 100 : 類似アンケートを探してみよう

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
