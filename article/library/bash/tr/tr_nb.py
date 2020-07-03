
# coding: utf-8

# ## tr
# 文字列を削除したり置換したりします。置換前後の文字列は1：1となるので、文字列の長さは同じにする必要があります。標準入力になります。
# 
# ```bash
# TR(1)                     BSD General Commands Manual 
# 
# NAME
#      tr -- translate characters
# 
# SYNOPSIS
#      tr [-Ccsu] string1 string2
#      tr [-Ccu] -d string1
#      tr [-Ccu] -s string1
#      tr [-Ccu] -ds string1 string2
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/tr/tr_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/tr/tr_nb.ipynb)
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
# abc 123という文字列を作り、それを標準入力から読み込み、abcをefgに置換します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo "abc 123" > temp\ncat temp | tr abc efg')


# dオプションを用いて文字列を削除します。

# In[4]:


get_ipython().run_cell_magic('bash', '', 'echo "abc 123" > temp2\ncat temp2 | tr -d 123')


# 大体上二つの利用方法が主かと思います。利用する頻度は中ぐらいです。僕はsedの方が好きなので、そちらを使う頻度が高いです。

# ## 代表的なオプション
# - d : 削除する
