
# coding: utf-8

# ## cat
# ファイルを連結します。ファイルの中身を表示するのによく使用します。
# ヒアドキュメントなど、複数行にわたるファイルを作成するのに利用します。
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
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)
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
# ### ファイルの表示、連結
# 
# ```bash
# cat file
# cat file1 file2
# cat file1 file2 > file3
# ```

# ファイルを作成し、その中身を表示します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo "a b c" > temp.txt \ncat temp.txt')


# ファイルを二つ作成し連結します。

# In[3]:


get_ipython().run_cell_magic('bash', '', 'echo "e f g" > temp1.txt\necho "h i j" > temp2.txt\ncat temp1.txt temp2.txt > temp3.txt')


# temp3.txtが作成され、その中でtemp1.txtとtemp2.txtが連結されていることがわかります。

# In[6]:


get_ipython().run_cell_magic('bash', '', 'cat temp3.txt')


# ### ヒアドキュメントの作成
# 
# スクリプトの中で複数行にわたるファイルを作成する際によく利用します。
# EOFの表記は何でも良いです。ファイルを作成する際にはリダイレクトさせます。

# In[8]:


get_ipython().run_cell_magic('bash', '', '\ncat << EOF > temp10.txt\na b c\ne f g\nh i j\nEOF')


# In[9]:


cat temp10.txt


# ただ、これだとコマンドをそのままを入れ込むことが出来ません。コマンドの結果や、変数などが展開されて表記されます。

# In[23]:


get_ipython().run_cell_magic('bash', '', '\ncat << EOF > temp11.sh\n#!/bin/bash\n\nuser="test"\n\necho ${user}\n\nEOF')

ここであえて変数を展開させたくない場合や、コマンドそのものの表記を残したい場合は、EOFをシングルクオテーションマークで囲みます。
# In[27]:


get_ipython().run_cell_magic('bash', '', 'cat << \'EOF\' > temp12.sh\n#!/bin/bash\n\nuser="test"\n\necho ${user}\n\nEOF\n\ncat temp12.sh')


# となり、ちゃんとファイルの中に`${user}`が展開されずにファイルの中に記載されていることがわかります。

# ### 代表的なオプション
# - t : タブを明示的に表示します(^Iと表示されます)
# - e : 改行コードを明示的に表示します（＄と表示されます）

# In[4]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\tb\\tc" > temp2.txt \ncat -t temp2.txt')


# In[1]:


get_ipython().run_cell_magic('bash', '', 'echo -e "a\\tb\\tc" > temp3.txt \ncat -e temp3.txt')

