#!/usr/bin/env python
# coding: utf-8

# ## 辞書型の引数をアンパックして代入する
# 
# 多くの引数を一度に代入するとき、辞書型の引数をアンパックして代入するととても便利である。
# 忘れないようにメモしておく。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/018/018_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/018/018_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## 三つの引数を持つ関数を用意

# In[6]:


def test(a,b,c):
  print('a : ', a)
  print('b : ', b)
  print('c : ', c)


# ### リスト型をアンパックして代入する

# In[9]:


arg = [
   '123',
   '456',
   '789',
]

test(*arg)


# ### 辞書型をアンパックして代入する

# In[10]:


arg = {
  'a': '123',
  'b': '456',
  'c': '789',
}

test(**arg)


# 簡単例だが後から遡れるようにメモ。
