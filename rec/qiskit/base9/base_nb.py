#!/usr/bin/env python
# coding: utf-8

# ## HHLアルゴリズム
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-applications/hhl_tutorial.html
# 
# 私の拙いブログでqiskitがRec（推薦システム）のカテゴライズしいたのは、すべてHHLを理解するためでした。現在、推薦システムに興味があり、開発などを行っていますが、そこで重要なのが連立一次方程式を解く事です。連立一次方程式は、数理モデルをコンピュータを利用してとく場合に高い確率で利用されますが、推薦システムもUser-Item行列から如何にしてユーザーエンゲージメントの高い特徴量を抽出出来るかという事が重要になってきます。
# 
# よって、量子コンピュータを利用して高速に連立一次方程式を解く事を目標に量子アルゴリズムの復習を開始したわけですが、ようやく目的までたどり着きました。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### 筆者の環境

# In[2]:


get_ipython().system('sw_vers')


# In[3]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import qiskit
import json

import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.visualization import plot_histogram

dict(qiskit.__qiskit_version__)


# In[ ]:





# ## 理論部分の理解
# 
# HHLの仮定
# - ローディングを実施する効果的なオラクルが存在
# - ハミルトニアンシミュレーションと解の関数の計算が可能
# - $A$がエルミート行列
# - $\mathcal{O}\left(\log (N) s^{2} \kappa^{2} / \epsilon\right)$で計算可能
# - 古典アルゴリズムが完全解を返すが、HHLは解となるベクトルを与える関数を近似するだけ

# In[ ]:





# $\tilde{\theta}$ は $2^{n} \theta$ に対するバイナリー近似を示しており、 $n$ の添え文字は $n$ デ ィジットに切り捨てられていることを示しています。

# $$
# \operatorname{QPE}\left(U,|0\rangle_{n}|\psi\rangle_{m}\right)=|\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
# $$

# In[ ]:





# In[ ]:




