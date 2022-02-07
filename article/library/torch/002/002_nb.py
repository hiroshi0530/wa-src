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
import torch

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('torch version :', torch.__version__)


# In[16]:


a = torch.tensor(range(12)).reshape(3,4).to(torch.float)
a


# In[17]:


b = torch.tensor([i + 0.5 for i in range(4)]).reshape(1,-1).to(torch.float)
b


# In[21]:


torch.cdist(a, b, p=2)


# In[20]:


np.sqrt((4*3.5**2))


# In[ ]:




