
# coding: utf-8

# ## seq
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/article/library/bash/seq/seq_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# 実際に動かす際は先頭の！や先頭行の%%bashは無視してください。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('bash --version')


# ### 通常の使い方

# In[3]:


get_ipython().system('for i in `seq 10`; do echo $i; done')


# seqを利用しなくても、{..}でいけます。0埋めなど知る必要がなければこれで十分です。

# In[4]:


get_ipython().system('for i in {0..10}; do echo $i; done')


# {..}は配列を作る演算子です。

# In[5]:


get_ipython().system('echo {1..10}')


# ### 連番の0埋め

# In[6]:


get_ipython().system('for i in `seq -w 3 10`; do echo $i; done')


# ### 埋める0の量を変更 

# In[7]:


get_ipython().system('for i in `seq -f %03g 3 10`; do echo $i; done')

