
# coding: utf-8

# ## ls
# ファイルやディレクトリを表示します。
# 以下の様にオプションがたくさんあるので、覚えるのは無理です。
# 自分がよく使うものを使えこなせれば良いかと思います。
# 
# ```bash
# NAME
#      ls -- list directory contents
# 
# SYNOPSIS
#      ls [-ABCFGHLOPRSTUW@abcdefghiklmnopqrstuwx1] [file ...
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/ls/ls_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/ls/ls_nb.ipynb)
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
# オプションなしです。直下のディレクトリにあるファイルを表示します。

# In[2]:


get_ipython().run_cell_magic('bash', '', 'ls')


# 最もよく使うのが`ls -al`です。エイリアス設定をしています。リスト形式で、ドットから始まる隠しファイルも表示してくれます。

# In[3]:


ls -al


# ## 代表的なオプション
# - a = all
# - l == list
