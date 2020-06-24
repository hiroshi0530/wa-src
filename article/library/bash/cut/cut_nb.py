#!/usr/bin/env python
# coding: utf-8

# ##  cut
# テキストファイルを読み取り、特定の文字列で分割します。タブで分割されたファイルなどから特定の列の値を抽出します。
# 
# ```text
# NAME
#      cut -- cut out selected portions of each line of a file
# 
# SYNOPSIS
#      cut -b list [-n] [file ...]
#      cut -c list [file ...]
#      cut -f list [-d delim] [-s] [file ...]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cut/cut_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('bash --version')


# ## 使用例

# ### 代表的なオプション
# - d : 区切り文字を指定します
# - f : 区切りから何文字目の文字を抽出するか選択します

# テスト用のファイルを作成します。タブ区切りで作成します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\tb\\nc\\td" > temp1.txt\ncat -t temp1.txt')


# 1列目を表示します。

# In[4]:


get_ipython().run_cell_magic('bash', '', 'cut -f 1 temp1.txt')


# 2列目を表示します。

# In[5]:


get_ipython().run_cell_magic('bash', '', 'cut -f 2 temp1.txt')


# スペースで区切られたファイルを作成します。

# In[6]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a b\\nc d" > temp2.txt\ncat temp2.txt')


# In[7]:


get_ipython().run_cell_magic('bash', '', "cut -d ' ' -f 1 temp2.txt")


# In[1]:


get_ipython().run_cell_magic('bash', '', "cut -d ' ' -f 2 temp2.txt")

