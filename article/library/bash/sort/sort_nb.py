#!/usr/bin/env python
# coding: utf-8

# ## sort
# ファイルを読み込み、降順、昇順に並び替えをします。
# 
# ```text
# NAME
#      sort -- sort or merge records (lines) of text and binary files
# 
# SYNOPSIS
#      sort [-bcCdfghiRMmnrsuVz] [-k field1[,field2]] [-S memsize] [-T dir]
#           [-t char] [-o output] [file ...]
#      sort --help
#      sort --version
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/sort/sort_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。

# In[1]:


get_ipython().system('sw_vers')


# In[1]:


get_ipython().system('bash --version')


# ## 使用例

# In[11]:


get_ipython().run_cell_magic('bash', '', 'echo -e "b\\nc\\na\\nz\\ny" > temp1.txt\ncat temp1.txt\nsort -r temp1.txt > temp2.txt\necho -e "\\nsorted"\ncat temp2.txt')

