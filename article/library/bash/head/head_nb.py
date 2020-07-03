
# coding: utf-8

# ## head
# ファイルの先頭、先頭から指定された行を表示します。
# 
# ```bash
# NAME
#      head -- display first lines of a file
# 
# SYNOPSIS
#      head [-n count | -c bytes] [file ...]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/head/head_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/head/head_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。

# In[13]:


get_ipython().system('sw_vers')


# In[3]:


get_ipython().system('bash --version')


# ## 使用例
# 
# オプションなしのデフォルト設定では最初の10行を表示します。

# In[16]:


get_ipython().run_cell_magic('bash', '', 'echo "テスト用のテキストの作成"\necho -e "1 \\n2 \\n3 \\n4 \\n5 \\n6 \\n7 \\n8 \\n9 \\n10 \\n11 \\n12" > temp\nhead temp\necho "先頭の10行しか表示されない"')


# ## 代表的なオプション
# - c : 先頭から指定したバイト数を表示
# - n : 先頭から指定した行数を表示⇒最も良く使うオプション

# ### n オプション

# In[12]:


get_ipython().run_cell_magic('bash', '', 'echo -e "test \\ntest \\ntest \\ntest \\ntest \\ntest \\n" > test\necho "先頭3行目を表示する"\nhead -n 3 test')

