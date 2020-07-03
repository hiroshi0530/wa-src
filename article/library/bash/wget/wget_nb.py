#!/usr/bin/env python
# coding: utf-8

# ## wget
# 指定されたURLからファイルをダウンロードします。
# 
# ```bash
# WGET(1)                            GNU Wget          
# 
# NAME
#        Wget  The noninteractive network downloader.
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/wget/wget_nb.ipynb)
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
# 普段は -O オプションを利用してファイルをローカルに保存します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'wget https://ja.wayama.io/index.html -O test.html')


# ファイルがあるかどうか確認します。ちゃんとダウンロードされています。

# In[4]:


get_ipython().system('ls | grep test')


# ## 代表的なオプション
# オプションは色々ありますが、とりあえず-Oだけ覚えています。
# - O : ファイル名を指定します。
