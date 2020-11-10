
# coding: utf-8

# ## blueqat tutorial 100番代
# 
# 今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
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


# In[3]:


from blueqat import __version__
print('blueqat version : ', __version__)


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[4]:


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


# In[5]:


from blueqat import Circuit


# ### 古典ベートと量子ゲートの比較
# 
# #### NOTゲート VS Xゲート
# 
# #### Xゲート
# 
# #### Yゲート
# 
# #### Zゲート
# 
# #### アダマールゲート
# 
# #### 位相ゲート
# 
# #### CNOTゲート

# ### 量子もつれ

# ### 1量子ビットの計算

# In[6]:


for i in range(5):
  print(Circuit().h[0].m[:].run(shots=100))


# ### 2量子ビットの計算

# In[9]:


Circuit().cx[0,1].m[:].run(shots=100)


# In[10]:


Circuit().x[0].cx[0,1].m[:].run(shots=100)

