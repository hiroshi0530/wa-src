#!/usr/bin/env python
# coding: utf-8

# ## jupyter notebook のカーネルが読み込めない
# 
# jupyter notebookでは仮想環境を簡単に変更できますが、今回うまく環境が切り替えることが出来なかったので、メモしておく。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/015/015_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/015/015_nb.ipynb)
# 
# ### 筆者の環境

# In[9]:


get_ipython().system('sw_vers')


# In[10]:


get_ipython().system('python -V')


# ## カーネル確認コマンド
# 
# 通常、jupyter notebook で実際のpythonの実行環境を表示を確認したい時は以下のコマンドを実行する。
# だが、今回ちゃんと所望のpythonのバージョンが指定されていたが、実際にはうまくいかないことがあった。

# In[2]:


import sys
sys.executable


# ## 課題
# 
# lgbmをjupyter notebook上で設定したが、実際に実行するとその環境のpythonが実行されていないことが分かった。

# ## 設定しているkernelの一覧
# 
# 利用しているカーネル一覧は以下のコマンドで確認できる。

# In[4]:


get_ipython().system('jupyter kernelspec list')


# ## カーネルの設定ファイル
# 
# `~/Library/Jupyter/kernels/`にそれぞれのカーネルの設定ファイルが保存されている。
# この中でlgbm2をカーネルとして設定している。

# In[7]:


get_ipython().system('ls -alh /Users/hiroshi.wayama/Library/Jupyter/kernels/')


# この中でlgbm2をカーネルとして設定しているがうまくいかない。そのファイルを開いてみる。

# In[6]:


get_ipython().system('cat /Users/hiroshi.wayama/Library/Jupyter/kernels/lgbm/kernel.json')


# pythonへのパスがデフォルトの`/Users/hiroshi.wayama/anaconda3/bin/python`となっており、想定していた`/Users/hiroshi.wayama/anaconda3/envs/lgbm/bin/python`ではないことが分かる。
# うまくいっているlgbm2を開いてみる。想定通り設定されていることが分かる。

# In[6]:


get_ipython().system('cat /Users/hiroshi.wayama/Library/Jupyter/kernels/lgbm2/kernel.json')


# パスの部分を利用したい環境のpythonパスに書き換えるとうまくいった。
# あまりこういう所で時間を取られたくないので覚えておく。
