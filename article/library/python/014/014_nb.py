#!/usr/bin/env python
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/014/014_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/014/014_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import japanize_matplotlib
import snap


# In[18]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[19]:


df_train.head()


# In[20]:


df_test.head()


# ## asdfghjk

# In[ ]:





# ## 参考記事

# In[ ]:




