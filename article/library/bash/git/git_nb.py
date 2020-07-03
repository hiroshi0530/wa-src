
# coding: utf-8

# ## git
# ソースコードを管理するgitのコマンドです。GUIからでも便利ですが、コマンドを覚えるとさらに便利で、早いです。
# 
# gitについてはネット上で使い方がたくさん紹介されているので、ここでは自分が使うコマンドを中心に書いていこうと思います。
# 
# ```bash
# NAME
#        git - the stupid content tracker
# 
# SYNOPSIS
#        git [--version] [--help] [-C <path>] [-c <name>=<value>]
#            [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
#            [-p|--paginate|-P|--no-pager] [--no-replace-objects] [--bare]
#            [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
#            [--super-prefix=<path>]
#            <command> [<args>]
# ```
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/git/git_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/git/git_nb.ipynb)
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

# よく利用しているコマンドの組み合わせと共に紹介します。
# 
# ### [git add] ファイルの差分をインデックスに追加する

# In[7]:


get_ipython().system('git add -h')


# ### [git status] 現在の状態を確認する 
# 
# ```bash
# git status
# ```

# ## 備忘録
# よく使うが、ついつい忘れてしまうコマンドです。コードレビューの時など良く利用します。

# ### ブランチ間のdiff
# これでデフォルト設定しているvimdiffでdiffをチェックできます。.gitconfigでdiffのデフォルトをvimdiffに設定する必要があります。

# In[ ]:


git difftool branchA branchB


# ### コミット間の差分（ファイル名のみ）
# コミットIDを指定して、ファイルの差分をチェック

# In[ ]:


git diff --stat shaA shaB


# ### コミット間の差分（各ファイル）
# vimdiffで差分が見れるのは本当に便利

# In[ ]:


git difftool -y shaA shaB -- <FILE NAME>

