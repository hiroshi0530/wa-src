#!/usr/bin/env python
# coding: utf-8

# ## 第9章 潜在顧客を把握するための画像処理10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 私は画像やテキストの処理は割と経験あるので、前章のネットワークの可視化などよりかなりスムーズに演習することが出来ました。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/09/09_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/09/09_nb.ipynb)
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

# ### ノック 81 : 画像データを読み込んでみよう 

# In[ ]:





# ### ノック 82 : 映像データを読み込んでみよう

# In[ ]:





# ### ノック 83 : 映像を画像に分割し、保存してみよう

# In[ ]:





# ### ノック 84 : 画像内のどこに人がいるのか検出しよう

# In[ ]:





# ### ノック 85 : 画像内の人の顔を検出してみよう

# In[ ]:





# ### ノック 86 : 画像内の人がどこに顔を向けているのかを検出してみよう

# In[ ]:





# ### ノック 87 : 検出した情報を統合し、タイムラプスを作ってみよう

# In[ ]:





# ### ノック 88 : 全体像をグラフにして可視化してみよう

# In[ ]:





# ### ノック 89 : 人通りの変化をグラフで確認しよう

# In[ ]:





# ### ノック 90 : 移動平均を計算することでノイズの影響を除去しよう

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
