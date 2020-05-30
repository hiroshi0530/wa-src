
# coding: utf-8

# ## cat
# ファイルを連結します。ファイルの中身を表示するのによく使用します。
# 
# ```bash
# NAME
#      cat -- concatenate and print files
# 
# SYNOPSIS
#      cat [-benstuv] [file ...]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/article/library/bash/cat/cat_nb.ipynb)
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
# ```bash
# cat file
# cat file1 file2
# ```

# ファイルを作成します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo "a b c" > temp.txt \ncat temp.txt')


# ### 代表的なオプション
# - t : タブを明示的に表示します(^Iと表示されます)
# - e : 改行コードを明示的に表示します（＄と表示されます）

# In[4]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\tb\\tc" > temp2.txt \ncat -t temp2.txt')


# In[1]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\tb\\tc" > temp3.txt \ncat -e temp3.txt')


# ## 参考記事
# 
# - [echo](/article/library/bash/echo/)
