#!/usr/bin/env python
# coding: utf-8

# ## tail
# ファイルの末尾を表示します。headオプションの逆です。
# 
# ```bash
# TAIL(1)                   BSD General Commands Manual 
# 
# NAME
#      tail -- display the last part of a file
# 
# SYNOPSIS
#      tail [-F | -f | -r] [-q] [-b number | -c number | -n number] [file ...]
# 
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/tail/tail_nb.ipynb)
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

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo "ファイルの準備"\necho -e "1\\n2 \\n3 \\n4 \\n5 \\n6" > temp\ncat temp\n\necho -e "\\n<ファイルの末尾3行を表示>"\ntail -n 3 temp')


# ## 代表的なオプション
# - n : 表示する行数
