#!/usr/bin/env python
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/010/010_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/010/010_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## pandasでgroupbyした後に高速で集計結果を取り出す
# 
# 以前、とあるpandasのスペシャリストからgroupbyした後、結果(DataFrame型)を高速で取り出す方法を教えてもらったのでメモしておく。
# 
# 基本的なライブラリ読み込み。

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import time
import json

import matplotlib.pyplot as plt
import numpy as np


# In[59]:


count = 8
df_list = []

for k in range(count):
  num = 10 ** k
  a1 = [{'a': i, 'b':i ** 2, 'c': i ** 3 % 9973} for i in range(num)]
  df_list.append(pd.DataFrame(a1))


# こちらが教えてもらったgroupyとforを組み合わせて、高速で取り出す方法。

# In[60]:


time_list = []
for i, df in enumerate(df_list):
  start_time = time.time()
  for c, _ in df.groupby('c'):
    pass
  time_list.append(time.time() - start_time)


# 恥ずかしながら、こちらがこれまで自分が使っていた手法。

# In[62]:


time_list_02 = []
c_list = df_list[count - 1]['c'].unique().tolist()

for i, df in enumerate(df_list):
  for c in c_list:
    df[df['c'] == c].count()
  time_list_02.append(time.time() - start_time)


# ### 結果比較
# 
# プロットして比較してみる。$x$軸はDataFrameの行数の指数、$y$軸は対数になっている事に注意。

# In[65]:


import matplotlib.pyplot as plt

plt.yscale('log')
plt.plot(range(count), time_list, label='modified')
plt.plot(range(count), time_list_02, label='previous')
plt.grid()
plt.legend()
plt.show()


# さらに大量なデータの場合でも1桁程度は高速が出来ているのを確認している。
# 普段何気なく使っているpandasも奥が深い。
# これを教えてくれた方に感謝しないと！本当に助かりました！
