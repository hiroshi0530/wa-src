
# coding: utf-8

# ## sys.getsizeof(a)
# 
# pythonではメモリの普段意識する場面はないかもしれませんが、オブジェクト単位でメモリの使用量を確認する方法です。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/002/002_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/002/002_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[1]:


get_ipython().system('python -V')


# In[12]:


import sys

a = [i ** 2 for i in range(10000)]

print('(先頭の10個を表示) a :',a[0:10])
print('aの使用メモリ        :', sys.getsizeof(a))

