#!/usr/bin/env python
# coding: utf-8

# ## scipy tips
# aaaaa-
# 
# ### scipy 目次
# 
# 1. [公式データセット](/article/library/sklearn/datasets/) <= 本節
# 2. [データの作成](/article/library/sklearn/makedatas/)
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/sklearn/datasets/ds_nb.ipynb)
# 
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy

matplotlib.__version__
scipy.__version__


# ## 正規分布

# In[4]:


from scipy.stats import norm

x = norm.rvs(size=1000)


# In[5]:


plt.grid()
plt.hist(x, bins=20)


# In[ ]:




