#!/usr/bin/env python
# coding: utf-8

# ## re 正規表現
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# 正規表現は書くことたくさんありそうですが、思い出したり、新しい使い方に遭遇したりしたら随時更新します。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/004/004_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/004/004_nb.ipynb)
# 
# ### 環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[88]:


import re
a = ''


# ## re.compile()

# ## re.sub()
# 正規表現で文字列を置換します。個人的には最も良く利用します。
# 
# ```text
# re.sub(正規表現パターン, 置換文字列, 置換対象文字列） 
# ```

# In[127]:


pat = r'0(8|9)0-[0-9]{4}-[0-9]{4}'
repl = r'0X0-YYYY-ZZZZ'
obj = r'080-1234-5678'

re.sub(pat, repl, obj)


# 置換する範囲。

# In[128]:


pat = r'0(8|9)0-[0-9]{4}-[0-9]{4}'
obj = r"""
080-1234-5678
090-8765-4321
"""

print(re.sub(pat,r'0X0-YYYY-ZZZZ', obj))


# ### 後方参照

# In[121]:


pat = r'0(8|9)0-([0-9]{4})-([0-9]{4})'
obj = r'080-1234-5678'

re.sub(pat,r'0\g<1>0-\3-\2', obj)


# 数字が連続し、分離する場合。

# In[ ]:


pat = r'0(8|9)0-([0-9]{4})-([0-9]{4})'
obj = r'080-1234-5678'

re.sub(pat,r'0\g<1>0-\3-\2', obj)


# ### 改行

# In[105]:


pat = '^0(8|9)0-[0-9]{4}-[0-9]{4}'
obj = """080-1234-5678
090-1234-4567
"""

re.sub(pat,'^0X0-YYYY-ZZZZ', obj, flags=re.MULTILINE)


# ## re.subn()

# ## re.split()
# 正規表現で文字列を分割します。

# ## re.match()

# ## re.search

# ## re.findall()
# 正規表現に一致するすべてのパターンをリストで取得します。

# ### 文字列の検索

# In[36]:


import re

pat = r'aaat(.*)tb([a-z]*)b'
pat = r'b{2,}'
target = 'aaatestbbbcccbbbbb'

result_match = re.match(pat, target)
result_search = re.search(pat, target)
result_findall = re.findall(pat, target)

print('### group ###')
if result_match:
  print('group :', result_match.group())
  print('span  :', result_match.span())
  print('start :', result_match.start())
  print('end   :', result_match.end())
else:
  print('matcn None')
print()

print('### search ###')
if result_search:
  print('group :', result_search.group())
  print('span  :', result_search.span())
  print('start :', result_search.start())
  print('end   :', result_search.end())
else:
  print('search None')
print()


print('### findall ###')
print(re.findall(pat, target))
print()

print('### finditer ###')
print(re.finditer(pat, target))
print()


# In[87]:


# re.findall('abc(...)*def', 'sssabcsabcssdefsssdefsssssssssssssssssssdefs')
re.findall('abc(...)*def', 'abcsabcssdefsssdef')
a = re.finditer('abc(...)*def', 'abctttsssdefqqqdef')

for i in a:
  print(i.group())
  print(i.span())


# In[38]:


re.search('(abc(...)*def)', 'sssabcsabcssdefsssdefsssssssssssssssssssdefs').group()


# In[48]:


text = "東京都港区虎ノ門 123-4567, 東京都渋谷区渋谷 765-4321"
import re
postCode_list = re.findall(r'(([0-9]{3})-[0-9]{4})' , text)
# postCode_list = re.findall(r'[0-9]{3}-[0-9]{4}' , text)
print (postCode_list)


# In[ ]:





# ## re.findall()
# 正規表現に一致するすべてのパターンをリストで取得します。

# ## re.finditer()
# 正規表現に一致するすべてのパターンをイテレータで取得します。

# In[ ]:




