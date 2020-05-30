
# coding: utf-8

# ## echo
# 文字列を標準出力に表示します。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/article/library/bash/echo/echo_nb.ipynb)
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


get_ipython().run_cell_magic('bash', '', 'echo "echo is basic option."')


# ## 代表的なオプション
# - n : 改行コードを付与しない
# - e : エスケープ文字を有効にして表示する

# ### n オプション

# In[4]:


get_ipython().run_cell_magic('bash', '', 'echo -n "linux"\necho " Linux"')


# ### e オプション

# In[5]:


get_ipython().run_cell_magic('bash', '', 'echo -e "this is a pen.\\nthis is a pen."')

