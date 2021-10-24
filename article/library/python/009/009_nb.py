#!/usr/bin/env python
# coding: utf-8

# ## Python のGC（ガベージコレクション）と参照カウンタ
# 
# pythonを利用する上で、便利な表記などの個人的なメモである。基本的な部分は触れておらず、対象も自分が便利だなと思ったものに限定している。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/009/009_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/009/009_nb.ipynb)
# 
# ### 筆者の環境

# In[24]:


get_ipython().system('sw_vers')


# In[25]:


get_ipython().system('python -V')


# ## GCとgetrefcount
# 
# GCの条件について色々する機会があったのでメモとして残しおく。
# 
# pythonは基本的にはガベージコレクションが採用されており、CやC++のように明示的にメモリを解放しなくても不要になったオブジェクトに関しては自動的に解放されるようになっている。それを実現している参照カウンタである。すべてのオブジェクトには、参照カウンタが内蔵されており、ある別のオブジェクトからそのオブジェクトへ参照があったら、参照カウンタをインクリメントする。また、その参照が削除されたらデクリメントを行い、カウンタが0になったらそのオブジェクトを解放する。
# 
# 普段あまり参照カウンタには注意を払ったことがなかったのですが、今回この数字を元に色々解析したので、その概要だけ記録しておく。

# In[42]:


import sys

class A():
  def __init__(self, a, b):
    self.a = a
    self.b = b


# オブジェクトが作成され、さらに`sys.getrefcount`から参照されるので、参照カウンタは2になる。

# In[43]:


a = A('a', 1) 
sys.getrefcount(a)


# bからも参照されるので3になる。

# In[44]:


b = a
sys.getrefcount(a)


# bを削除すると2になる。

# In[45]:


b = None
sys.getrefcount(a)


# インスタンス変数へ参照しても、インクリメントされない。

# In[46]:


c = a.b
sys.getrefcount(a)


# ### GC
# 
# 通常、GCは`del`を行った後、`gc.collet()`を明示的に行う。
# ただ、`del`後もメモリからその値が削除されるわけではなく、あくまでもpythonから参照できなくなるだけである。

# In[30]:


import gc

b = A('b', 2) 
del b

print(gc.get_stats()[2])
gc.collect()
print(gc.get_stats()[2])


# gc後は参照カウンタは取得できない。

# In[31]:


sys.getrefcount(b)


# ## 参考記事
# 
# - http://docs.daemon.ac/python/Python-Docs-2.4/ext/refcounts.html
# - https://emptypage.jp/notes/py-__del__-and-refcycle.html
