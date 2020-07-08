#!/usr/bin/env python
# coding: utf-8

# ## at 
# 時間を指定してジョブを実行する
# 
# ```bash
# AT(1)                     BSD General Commands Manual  
# 
# NAME
#      at, batch, atq, atrm -- queue, examine, or delete jobs for later execution
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/at/at_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/at/at_nb.ipynb)
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
# 通常私が利用するときは、fオプションとtオプションを利用します。
# 
# 以下の様なbashファイルを用意します。

# In[13]:


get_ipython().run_cell_magic('bash', '', "\ncat << 'EOF' > temp.sh\n#!/bin/bash\n\necho `date +%Y%m%d`\n\nEOF\n\nchmod +x temp.sh")


# In[14]:


get_ipython().system('ls -al')


# In[15]:


get_ipython().system('./temp.sh')


# このファイルを2020年7月8日18:00に実行させるには次のようなコマンドを実行します。タイマー的な使い方が出来るので、一時的に使いたいのであれば、CRONを設定するより簡単です。

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'at -f temp.sh.sh -t 202007081800')


# ## 代表的なオプション
# - f : ファイルを指定します
# - t : 時間のフォーマットを定義します (YYYYmmddHHMM)
