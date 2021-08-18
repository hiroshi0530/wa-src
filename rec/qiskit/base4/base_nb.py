#!/usr/bin/env python
# coding: utf-8

# ## ドイチ-ジョサのアルゴリズム
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-algorithms/deutsch-jozsa.html
# 
# 今回は、ドイチ-ジョサのアルゴリズムを数式を追って理解を深めようと思います。
# 
# 私の勉強不足ですが、学生時代、量子情報を勉強した際はドイチ-ジョサのアルゴリズムというのは知りませんでした。ショアのアルゴリズムなどと比較すると有名ではないのかもしれませんが、qiskitのサイトに寄ると、古典アルゴリズムよりも優れたパフォーマンスを持つと発表された最初の量子アルゴリズムという事で、理解は必須のような気がしています。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)
# 
# ### 筆者の環境

# In[36]:


get_ipython().system('sw_vers')


# In[37]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[18]:


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


# In[19]:


import qiskit
import json

dict(qiskit.__qiskit_version__)


# In[25]:


from qiskit import IBMQ, Aer, execute
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import array_to_latex


# ## 問題設定
# 
# 
