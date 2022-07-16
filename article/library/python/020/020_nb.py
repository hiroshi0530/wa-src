#!/usr/bin/env python
# coding: utf-8

# ## pandasでgroupby後にfilterをかける
# 
# pandas利用中にgroupby後にある条件を適用したい場面に遭遇した。
# 調べてみると、`groupby.filter(lambda x: x)`でfilter関数を適用できる事がわかった。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/020/020_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/020/020_nb.ipynb)
# 
# ### 実行環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## 適当なDataFrameを作成

# In[1]:


import pandas as pd

df = pd.DataFrame({
    'item': ['apple', 'apple', 'apple', 'orange', 'melon', 'apple', 'orange'],
    'sales': [1, 2, 1, 2, 3, 1, 1],
})
df


# このDataFrameから、例えば、itemのカウントが2個以上の項目だけgroupbyしたいという機会があった。
# これは以下の様に、groupbyの後にfilterとlambdaを利用する事で実現できる。

# In[4]:


df.groupby('item').filter(lambda x: x['sales'].count() >= 2)


# In[6]:


df.groupby('item').filter(lambda x: x['sales'].max() >= 3)


# In[8]:


df.groupby('item').filter(lambda x: x['sales'].min() <= 2)


# 今までは余計なDataFrameを作成していたので、今後はワンライナーで実行するようにする。
