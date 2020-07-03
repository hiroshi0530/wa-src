#!/usr/bin/env python
# coding: utf-8

# ## split
# ファイルの分割を行います。
# 
# ```bash
# NAME
#      split -- split a file into pieces
# 
# SYNOPSIS
#      split [-a suffix_length] [-b byte_count[k|m]] [-l line_count]
#            [-p pattern] [file [name]]
# 
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)
# 
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# linux環境では-nオプションで分割数が指定出来るのですが、manを見てもわかるとおり、FreeBSD（macOSの元となるOS）ではそういうオプションはありません。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。
# 

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('bash --version')


# ## 使用例
# 
# 通常、以下の様な代表的なオプションと共に利用します。
# 
# ### 代表的なオプション
# - b : 分割するバイト数
# - l : 分割する行数
# - a : prefixに利用する文字数
# 

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo -e "1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10" > temp\ncat temp\n\nsplit -l 2 -a 3 temp prefix_\necho -e "\\n<file list>"\nls | grep -v split\n\necho -e "\\n<prefix_aaa file content>"\ncat prefix_aaa')

