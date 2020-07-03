
# coding: utf-8

# ## uniq
# 重複している行を削除します。
# 
# ```bash
# UNIQ(1)                   BSD General Commands Manual                  UNIQ(1)
# 
# NAME
#      uniq -- report or filter out repeated lines in a file
# 
# SYNOPSIS
#      uniq [-c | -d | -u] [-i] [-f num] [-s chars] [input_file [output_file]]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/uniq/uniq_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/uniq/uniq_nb.ipynb)
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
# データ分析の仕事をしているとよく利用します。重複している無駄なものはゴミになる可能性が高いですから。通常は、
# 
# ```bash
# uniq <in file> <out file>
# ```
# 
# で重複している行を削除した結果を別ファイルに出力させます。

# In[13]:


get_ipython().run_cell_magic('bash', '', 'echo "ファイルの準備"\necho -e "123\\n123\\n123" > temp\necho -e "<before>"\ncat temp\n\nuniq temp temp2\necho -e "\\n<after>"\ncat temp2')

