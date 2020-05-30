
# coding: utf-8

# ## paste
# テキストファイルを列方向に結合します。catの列板です。
# 
# ```text
# NAME
#      paste -- merge corresponding or subsequent lines of files
# 
# SYNOPSIS
#      paste [-s] [-d list] file ...
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/article/library/bash/paste/paste_nb.ipynb)
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

# 結合する二つのファイルを作成します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\nb\\nc\\nd\\ne\\nf" > temp1.txt\necho -e " 1\\n 2\\n 3\\n 4\\n 5\\n 6" > temp2.txt')


# In[4]:


get_ipython().run_cell_magic('bash', '', 'paste temp1.txt temp2.txt')


# ３つ以上のファイルも連結できます。

# In[5]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\nb\\nc\\nd\\ne\\nf" > temp3.txt\necho -e " 1\\n 2\\n 3\\n 4\\n 5\\n 6" > temp4.txt\npaste temp1.txt temp2.txt temp3.txt temp4.txt')


# ### 代表的なオプション
# - d : 結合文字を指定します。デフォルトはタブです。
# - s : 行と列を反転させます。 

# In[6]:


get_ipython().run_cell_magic('bash', '', 'paste -d _ temp1.txt temp2.txt')


# In[7]:


get_ipython().run_cell_magic('bash', '', 'paste -s temp1.txt temp2.txt')

