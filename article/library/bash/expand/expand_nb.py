#!/usr/bin/env python
# coding: utf-8

# ## expand
# ファイルを読み込み、タブを半角スペースに変換します。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/expand/expand_nb.ipynb)
# 
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
# expnad <option> filen_ame
# ```

# ## 代表的なオプション
# - [-t N] : 変換するタブの数を指定します。 

# ### tオプション

# タブを使ったファイルを作成します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo -e "1\\t2\\t3" > temp.txt  # echo -e でタブをファイルに挿入します。')


# In[4]:


get_ipython().run_cell_magic('bash', '', 'cat -t temp.txt # cat -t でタブを可視化します。')


# In[5]:


get_ipython().run_cell_magic('bash', '', 'expand -t 5 temp.txt')


# ## 参考記事
# 
# - [cat](/article/library/bash/cat/)
# - [echo](/article/library/bash/echo/)
