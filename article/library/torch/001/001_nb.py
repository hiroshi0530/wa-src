#!/usr/bin/env python
# coding: utf-8

# ## pytorch で配列を反転させる
# 
# 最近NLP関連でpytorchを触る機会が増え、個人的に覚えておきたいことをメモしておきます。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/torch/001/001_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/torch/001/001_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[45]:


get_ipython().system('sw_vers')


# In[46]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('torch version :', torch.__version__)


# ## numpyによる反転

# In[48]:


a = np.array([i for i in range(10)])
a


# In[49]:


a[::-1]


# In[50]:


a = np.array([[i * j for i in range(10)] for j in range(10)])
a


# In[51]:


a[:,::-1]


# ## pytrochによる反転
# 
# pytorchは`a[::-1]`のような反転は出来ないので、別の方法で反転させる必要がある。

# ### 1次元tensorの反転

# In[52]:


a = torch.tensor(range(12))
a


# In[53]:


torch.flip(a, dims=[0])


# ### 2次元tensorの反転

# In[58]:


a = a.reshape(3,4)
a


# X軸で反転させる。

# In[59]:


torch.flip(a, dims=[0])


# Y軸で反転させる。

# In[60]:


torch.flip(a, dims=[1])


# 左右で反転させる。

# In[61]:


torch.fliplr(a)


# たまに忘れていちいち調べることになるので、覚えておく。
