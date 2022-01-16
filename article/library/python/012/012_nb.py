#!/usr/bin/env python
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/template/template_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/template/template_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import japanize_matplotlib


# ## pandasでカラムを追加したときの注意点
# 
# あるnan入りのDataFrameに対して、それをdropした後、Series型のカラムを追加しようとしたとき、想定通りに動かず、数時間潰してしまった。

# In[5]:


a = pd.DataFrame({
  'a': [1,2,3],
  'b': [1,np.nan,3],
  'c': [1,2,3],
})

a = a.dropna(subset=['b'])
a['d'] = pd.Series([1,33])
a


# $d$の２番目のデータに３３が入っている想定だったが、nanが入っている。
# 
# インデックスをリセットしなければならないという事に気付くのに、数時間かかった。

# In[8]:



a = pd.DataFrame({
  'a': [1,2,3],
  'b': [1,np.nan,3],
  'c': [1,2,3],
})

a = a.dropna(subset=['b']).reset_index()
a['d'] = pd.Series([1,33])
a


# となり、想定通りに新しいカラムを追加することができた。

# 実験的に最初の$a$に適当な数字でインデックスをつけて見ると、全く今まで想定しなかったカラムの追加がされていた。

# In[9]:


a = pd.DataFrame({
  'a': [1,2,3],
  'b': [1,np.nan,3],
  'c': [1,2,3],
},index=[12,24,36])

# a = a.dropna(subset=['b']).reset_index()
a = a.dropna(subset=['b'])
a['d'] = pd.Series([1,33])
a


# 以下の様に新規のカラムを追加する際、暗黙的にインデックスが０から張られてしまうのだと思う。
# 
# ```python
# a['d'] = pd.Series([1,33])
# ```
# 
# いままでこの事を意識することなくカラムを追加してきたが、特に問題なかったように記憶している。単純にdropnaなどをせず、インデックスが0から順番に整数で貼られていたからだと思われる。
# 
# pandasに詳しい人なら当たり前の事かもしれないが、かなり時間を取られたので、メモ。
