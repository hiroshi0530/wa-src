#!/usr/bin/env python
# coding: utf-8

# ## tar
# ファイルを結合し、アーカイブを作成します。また、アーカイブからファイルを展開します。
# 
# ```bash
# BSDTAR(1)                 BSD General Commands Manual 
# 
# NAME
#      tar -- manipulate tape archives
# 
# SYNOPSIS
#      tar [bundled-flags <args>] [<file> | <pattern> ...]
#      tar {-c} [options] [files | directories]
#      tar {-r | -u} -f archive-file [options] [files | directories]
#      tar {-t | -x} [options] [patterns]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/tar/tar_nb.ipynb)
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
# 
# ### ファイルの作成

# In[10]:


get_ipython().run_cell_magic('bash', '', 'echo "ファイルの準備"\necho "1234567890" > temp1\necho "0987654321" > temp2\n\ntar -czf temp.tgz temp1 temp2\necho -e "\\n<ls>"\nls | grep temp')


# ### ファイルの解凍

# In[13]:


get_ipython().run_cell_magic('bash', '', 'tar -xzf temp.tgz')


# ## 代表的なオプション
# 長年、4つのコマンドを組み合わせた以下の二つのコマンドを利用しています。
# vはverboseで必要な場合利用します。
# 今までこの二つで困ったことはありません。
# 
# ### ファイルの解凍
# - xzvf
# 
# ### ファイルの作成
# - czvf
