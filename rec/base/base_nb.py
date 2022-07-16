#!/usr/bin/env python
# coding: utf-8

# ## Marketing と Recommender Systems と Information Retrival
# 
# 主に推薦システムの理解に必要な線型代数の知識をまとめていこうと思います。
# 推薦システムで利用される、user-itemの行列（嗜好行列）に対して、しばしば低ランク近似が成立する事を前提に議論が進められることがあります。
# 

# ## 記事案
# 
# ### 
# 
# ### Recommender Systems
# 1. コンテンツベースフィルタリング
# 2. 

# 

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import json
import math
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from tabulate import tabulate

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import matplotlib

print('matplotlib  : {}'.format(matplotlib.__version__))
print('networkdx   : {}'.format(nx.__version__))
print('numpy       : {}'.format(np.__version__))
print('torch       : {}'.format(torch.__version__))


# In[39]:


get_ipython().system('ls -alh ../dataset/ml-100k/')


# In[40]:


import pandas as pd

df = pd.read_csv('../dataset/ml-100k/ml-100k.inter', sep='\t')
df


# In[49]:


# plt.figure(figsize=(18, 6)).patch.set_facecolor('white')
# plt.style.use('ggplot')
df.groupby('user_id:token').count().sort_values(by='item_id:token', ascending=False).reset_index() .plot.scatter(x='user_id:token', y='item_id:token')


# In[48]:


df.groupby('user_id:token').count().sort_values(by='item_id:token', ascending=False).reset_index() ['item_id:token'].plot()


# In[ ]:





# 

# ## まとめ
# 
# - aa

# ## 参考文献
# 
# - [1][現代線形代数](https://www.amazon.co.jp/%E7%8F%BE%E4%BB%A3%E7%B7%9A%E5%BD%A2%E4%BB%A3%E6%95%B0-%E2%80%95%E5%88%86%E8%A7%A3%E5%AE%9A%E7%90%86%E3%82%92%E4%B8%AD%E5%BF%83%E3%81%A8%E3%81%97%E3%81%A6%E2%80%95-%E6%B1%A0%E8%BE%BA-%E5%85%AB%E6%B4%B2%E5%BD%A6/dp/4320018818)
