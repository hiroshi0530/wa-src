#!/usr/bin/env python
# coding: utf-8

# ## xargs
# 標準入力からリストを読み込んで、逐次コマンドを実行します。
# とても便利なコマンドで、ファイルの操作などをCUIで実行するには必須のコマンドだと思います。使いこなせばかなり作業効率が上がると思います。
# 
# ```bash
# XARGS(1)                  BSD General Commands Manual 
# 
# NAME
#      xargs -- construct argument list(s) and execute utility
# 
# SYNOPSIS
#      xargs [-0opt] [-E eofstr] [-I replstr [-R replacements]] [-J replstr]
#            [-L number] [-n number [-x]] [-P maxprocs] [-s size]
#            [utility [argument ...]]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/xargs/xargs_nb.ipynb)
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
# とても便利なコマンド、様々な利用所があります。
# 以下に私がよく利用するコマンドの例を示します。
# 
# ### lsを利用してファイル名を一括して変更する
# 
# ```bash
# ls | grep XXXX | xargs -I{} cp {} {}.bk
# ```
# 
# XXXXが含まれるファイルを検索し、そのファイルのバックアップを取る。I{}でパイプで渡す前の引数を取得することができます。

# In[2]:


get_ipython().run_cell_magic('bash', '', 'echo "tempという文字列がつくファイルを3つ作成します。"\necho "temp1" > temp1\necho "temp2" > temp2\necho "temp3" > temp3\n\nls | grep temp')


# これら3つのファイルに対してバックアップファイルを作成します。

# In[5]:


get_ipython().run_cell_magic('bash', '', 'ls | grep temp | xargs -I{} cp {} {}.bk\n\necho "<file>"\nls | grep temp')


# となり、一括でバックアップが作成されます。
# 
# ### 特定の文字列を含むファイルを一括削除
# 
# abcという文字列を含むファイルをすべて削除します。とても便利です。

# In[7]:


get_ipython().run_cell_magic('bash', '', 'echo "abcという名前がつくファイルを5つ作成"\necho "abc" > abc1\necho "abc" > abc2\necho "abc" > abc3\necho "abc" > abc4\necho "abc" > abc5\n\nls | grep abc')


# In[10]:


get_ipython().system('ls | grep abc | xargs -I{} rm {}')
get_ipython().system('ls | grep abc')


# となり、abcという文字列がつく5つのファイルは削除されました。
# 
# ### ファイルの中にXXXが含まれるファイル名の再帰検索し、すべてYYYに置換
# 
# ```bash
# grep XXX -rl . | xargs sed -i.bk -e 's/XXX/YYY/g'
# ```
# 
# ### ファイル名にXXXが含まれるファイル名の再帰検索し、すべてYYYに置換
# 
# ```bash
# find ./ -type f | sed 'p;s/aaa/bbb/' | xargs -n2 mv
# ```
# 
# 以上のように、とても使い勝手良いコマンドですので、ぜひ自分の用途に応じて使いこなして作業効率を上げていきたいです。

# ## 代表的なオプション
# `xargs -I{}` でパイプ前の結果を引数として取得できるので重宝しています。
# - I{}
