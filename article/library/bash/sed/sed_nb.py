
# coding: utf-8

# ## sed
# 文字列を置換します。
# とても便利なコマンドで、様々な所で利用します。
# 文字列の置換だけでなく、新たな行を追加したり、削除したり出来ます。
# 
# ```bash
# NAME
#      sed -- stream editor
# 
# SYNOPSIS
#      sed [-Ealn] command [file ...]
#      sed [-Ealn] [-e command] [-f command_file] [-i extension] [file ...]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/sed/sed_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/sed/sed_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。

# In[2]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('bash --version')


# ## 使用例
# 基本的には
# 
# ```bash
# sed s/before/after/g <FILE>
# ```
# 
# でファイル内のbeforeという文字列をafterに置換します。
# 
# ```bash
# sed -ei s/before/after/g <FILE>
# ```
# というeiオプションでファイルを上書きします。MACでなければiオプションで上書きできます。最後のgはファイル内のすべてbeforeに対して置換を行います。なければ最初に一致する最初のbeforeだけ置換します。

# In[16]:


get_ipython().run_cell_magic('bash', '', 'echo "ファイルの準備をします。"\necho "=== example : before ===" > temp\ncat temp\nsed -ie "s/before/after/g" temp\ncat temp')


# ## 代表的なオプション
# - e
# - i
