#!/usr/bin/env python
# coding: utf-8

# ## pandasのSettingWithCopyWarningについて
# 
# pandasを利用していると、SettingWithCopyWarningが出ることがあります。基本的には、参照渡しに起因する部分が原因で、DataFrameをcopy()メソッドによって、別のメモリに独立して作成すれば問題ないのですが、今回copy()を利用してもワーニングが解決出来ませんでした。
# 
# なぜこうなるかのは不明で、おそらくcopy()を利用した場合のワーニングは無視しても問題ないと思いますが、一応解決案をメモしておきます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/07/07_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/07/07_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pandas as pd
import numpy as np

# 6 x 2のDataFrameを作成します
df = pd.DataFrame(np.arange(12).reshape(6, 2), columns=['c0', 'c1'])

df


# locで条件に合う部分だけを抽出する形でオブジェクトを作成し、カラムを指定してから、ilocを用いて上書きしようとするとSettingWithCopyWarningが出現します。
# 
# これはよく見られるワーニングです。

# In[4]:


df_1 = df[['c0']].loc[df['c0'] % 3 == 0]

df_1['c1'] = None 
df_1['c1'].iloc[0] = 12
df_1.head()


# 通常であれば、copy()メソッドを利用し、参照渡しではなく、別途メモリ上にオブジェクトを作成すればワーニングは消えます。しかし、この場合は消えません。

# In[5]:


df_2 = df[['c0']].loc[df['c0'] % 3 == 0].copy()

df_2['c1'] = None 
df_2['c1'].iloc[0] = 12
df_2.head()


# ### 解決案
# 
# ilocで直接、行番号と列番号をしてすれば良いようです。
# そのために、わざわざ columns.get_locメソッドを利用して、カラムのインデックス番号を取得する必要があります。

# In[6]:


df_3 = df[['c0']].loc[df['c0'] % 3 == 0]

df_3['c1'] = None 
idx = df_3.columns.get_loc('c1')

df_3.iloc[0, idx] = 12
df_3.head()


# ### まとめ
# 
# 以上SettingWithCopyWarningの特殊な回避方法の紹介でした。ただ、私の感覚ですが、copy()メソッドを利用していれば問題ないでしょうし、pythonやpandasのバージョンによって挙動は変わると思います。
# 
# 何かしらの参考になれば幸いです。
