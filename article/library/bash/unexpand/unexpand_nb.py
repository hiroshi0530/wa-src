
# coding: utf-8

# ## unexpand
# 与えられたファイルのスペースをタブに変換します。結果を標準出力に表示します。
# 
# ```bash
# NAME
#      expand, unexpand -- expand tabs to spaces, and vice versa
# 
# SYNOPSIS
#      expand [-t tab1,tab2,...,tabn] [file ...]
#      unexpand [-a | -t tab1,tab2,...,tabn] [file ...]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/unexpand/unexpand_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/unexpand/unexpand_nb.ipynb)
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


get_ipython().run_cell_magic('bash', '', 'echo -e "a   b\\nc   d" > temp1.txt\ncat temp1.txt')


# In[4]:


get_ipython().run_cell_magic('bash', '', 'unexpand -a -t 3 temp1.txt > temp2.txt\ncat -te temp2.txt')

