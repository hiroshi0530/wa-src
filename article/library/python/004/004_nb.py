#!/usr/bin/env python
# coding: utf-8

# ## re 正規表現
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# 正規表現は書くことたくさんありそうですが、思い出したり、新しい使い方に遭遇したりしたら随時更新します。
# 
# 若い頃はいろいろな正規表現パターンを覚えようと思って頑張りましたが、今は必要な時だけ調べて利用するようにしています。全部覚えて使いこなす（正規表現マスター）になるのはしんどいですね。
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


# ## reの読み込み
# 
# 正規表現は通常、解析したい文字列と解析の元となる正規表現のパターンの二つの文字列を必要とします。
# 
# 元となる正規表現のパターンにはrをつけると、バックスラッシュが必要な文字もそのまま表現できるので、デフォルトでつけておいた方が良いようです。
# 
# 読み込みは以下の通りです。

# In[3]:


import re


# ## re.compile()
# 
# 正規表現による検索や置換をする方法は、二通りあります。一つが、`re.complie()`を用いて、正規表現パターンをあらかじめコンパイルしておいて、必要な時にそのオブジェクトを利用する方法。もう一つが、必要な時にコンパイルの処理を行い、それと同時にそのオブジェクトを利用する方法です。何度もその検索パターンを利用する場合は、`re.compile()`を利用した方が良いかと思います。
# 
# 二つの利用方法の例を示します。
# 
# ### コンパイルを利用しない場合

# In[4]:


obj = r'asdfghjkl'
target = r'gh'

ret = re.findall(target, obj)

if ret:
  print('findall の結果 : ', ret)


# ### コンパイルを利用する場合

# In[5]:


target = r'asdfghjkl'
pat = r'gh'

pat_target = re.compile(pat)

ret = pat_target.findall(pat)

if ret:
  print('findall の結果 : ', ret)


# ## 文字列の検索
# 
# 検索関数は4つの方法があります。私個人としてはfindallを利用するのが最も多いです。
# 
# - re.match : ターゲットの文字列の先頭が正規表現とマッチするか
# - re;.search : ターゲットの文字列が正規表現とマッチするか
# - re.findall : ターゲットの文字列で、正規表現とマッチする部分をリスト化して返す 
# - re.finditer :ターゲットの文字列で、正規表現とマッチする部分をイテレータとして返す
# 
# ### re.match

# In[6]:


import re

pat = r'aaat(.*)tb([a-z]*)b'
target = 'aaatestbbbcccbbbbb'

result_match = re.match(pat, target)

print("### 文字列 ###")
print('pat    : ', pat)
print('target : ', target)
print()

print('### group ###')
if result_match:
  print('group :', result_match.group())
  print('span  :', result_match.span())
  print('start :', result_match.start())
  print('end   :', result_match.end())
else:
  print('matcn None')


# ### re.search

# In[7]:


pat = r'aaat(.*)tb([a-z]*)b'
target = 'aaatestbbbcccbbbbb'

result_search = re.search(pat, target)

print("### 文字列 ###")
print('pat    : ', pat)
print('target : ', target)
print()

print('### search ###')
if result_search:
  print('group :', result_search.group())
  print('span  :', result_search.span())
  print('start :', result_search.start())
  print('end   :', result_search.end())
else:
  print('search None')


# In[8]:


result_search = re.search('(abc(...)*def)', 'sssabcsabcssdefsssdefsssssssssssssssssssdefs')

print("### 文字列 ###")
print('pat    : ', pat)
print('target : ', target)
print()

print('### search ###')
if result_search:
  print('group :', result_search.group())
  print('span  :', result_search.span())
  print('start :', result_search.start())
  print('end   :', result_search.end())
else:
  print('search None')


# ### re.findall

# In[9]:


pat = r'aaat(.*)tb([a-z]*)b'
target = 'aaatestbbbcccbbbbb'

result_findall = re.findall(pat, target)

print("### 文字列 ###")
print('pat    : ', pat)
print('target : ', target)
print()

print('### findall ###')
print(re.findall(pat, target))


# ### re.finditer
# 
# 一致するイテレータを返します。

# In[10]:


pat = r'aaat(.*)tb([a-z]*)b'
target = 'aaatestbbbcccbbbbb'

result_finditer = re.finditer(pat, target)

print("### 文字列 ###")
print('pat    : ', pat)
print('target : ', target)
print()

print('### finditer ###')
print(re.finditer(pat, target))


# ## 文字列の置換
# 
# ### re.sub()
# 正規表現で文字列を置換します。個人的には最も良く利用します。
# 
# ```text
# re.sub(正規表現パターン, 置換文字列, 置換対象文字列） 
# ```

# In[11]:


pat = r'0(8|9)0-[0-9]{4}-[0-9]{4}'
repl = '0X0-YYYY-ZZZZ'
obj = '080-1234-5678'

re.sub(pat, repl, obj)


# In[12]:


pat = r'0(8|9)0-[0-9]{4}-[0-9]{4}'
obj = """
080-1234-5678
090-8765-4321
"""

print(re.sub(pat,r'0X0-YYYY-ZZZZ', obj))


# #### 後方参照

# In[13]:


pat = r'0(8|9)0-([0-9]{4})-([0-9]{4})'
obj = '080-1234-5678'

re.sub(pat,r'0\g<1>0-\3-\2', obj)


# 数字が連続し、分離する場合。

# In[14]:


pat = r'0(8|9)0-([0-9]{4})-([0-9]{4})'
obj = '080-1234-5678'

re.sub(pat,r'0\g<1>0-\3-\2', obj)


# #### 改行

# In[15]:


pat = r'^0(8|9)0-[0-9]{4}-[0-9]{4}'
obj = """080-1234-5678
090-1234-4567
"""

re.sub(pat,'^0X0-YYYY-ZZZZ', obj, flags=re.MULTILINE)


# ### re.subn()
# 
# 準備中

# ### re.split()
# 正規表現で文字列を分割します。
