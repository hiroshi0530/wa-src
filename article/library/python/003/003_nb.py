
# coding: utf-8

# ## id() メモリアドレスの取得
# 
# 普段あまり意識しないメモリアドレスですが、メモリの量が限られた環境の中ではどのオブジェクトがどれぐらいメモリを使用しているか知っているとハードウェアに優しいシステムを開発することが出来ます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/003/003_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/003/003_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


a = [i ** 2 for i in range(100)]

id(a)


# オブジェクトが同一であるかどうかチェックする際に利用します。

# In[4]:


a = [1,2,3]
b = a

print('a :',id(a))
print('b :',id(b))


# ### 値が同じだと異なる変数の宣言も同じアドレスが割り当てられる

# In[5]:


a = 1
b = 1

print('a =',a)
print('b =',b)
print('同じアドレス')
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))
print()

a = 1
b = 2

print('a =',a)
print('b =',b)
print('異なるアドレス')
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))


# ### 代入すると新しいオブジェクトが出来る

# In[6]:


a = 1

print('a =',a)
print('a :',id(a))
print()

# aに異なる値を代入
a = 2

print('a =',a)
print('a :',id(a))


# ### 配列の場合は参照渡しで、参照元が変更されると参照先も変更される

# In[7]:


a = [1,2,3]
b = a

print('a =',a)
print('b =',b)
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))
print()

a[0] = 5

print('参照元も変更')
print('a =',a)
print('b =',b)
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))


# ### copy()を利用し、明示的に別のオブジェクトを作成する

# In[8]:


from copy import copy
a = [1,2,3]
b = copy(a)

print('a =',a)
print('b =',b)
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))
print()

a[0] = 5

print('a =',a)
print('b =',b)
print('a :',id(a))
print('b :',id(b))
print(id(a) == id(b))

