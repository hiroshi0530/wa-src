#!/usr/bin/env python
# coding: utf-8

# ## pandasのreindexとdate_rangeを利用して、時系列データの欠損を埋める
# 
# ECサイトの売上のデータ解析などをしていると、休日のデータが欠損している場合がある。
# 解析時には日付が欠損していると不便なことがあるので、0などのある値で埋めるために、pandasのreindexとdate_rangeを利用する。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/022/022_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/022/022_nb.ipynb)
# 
# ### 実行環境

# In[16]:


get_ipython().system('sw_vers')


# In[17]:


get_ipython().system('python -V')


# 時系列データでデータの穴抜けがあるDataFrameを用意する。

# In[18]:


import pandas as pd

df = pd.DataFrame({
    'sales': [i + 1 for i in range(5)],
    'date': pd.to_datetime(['2022-07-01', '2022-07-02', '2022-07-05', '2022-07-06', '2022-07-09'])
})
df


# In[ ]:


土日が休みや定休日があるお店だとよく見られるデータである。
時系列データで日付に穴があると、解析時に不便な場合があるので、これを埋める事が今回の目的である。


# ## date_range
# 
# pandasにはdate_rangeという連続的な日次のデータを作成してくれる関数がある。
# startとendを設定し、frequencyを指定するだけである。
# freqに`60min`を設定すると1時間毎に、`240min`を指定すると4時間毎のdatetime型のlistを作ることができる。

# In[27]:


pd.date_range('2022-07-01', '2022-07-02', freq='60min')


# In[28]:


pd.date_range('2022-07-01', '2022-07-02', freq='240min')


# ## reindex
# 
# date_rangeとreindexを利用して、欠損データの穴埋めをする。
# reindexは設定されいるindexに値があるときはその値が割り振られ、値がない場合はNaNが割り振られる。
# ただ、穴埋めするデータも`fill_value`で指定することができる。今回は0で埋める。

# In[29]:


start_time = df['date'].tolist()[0]
end_time = df['date'].tolist()[-1]

time_list = pd.date_range(start_time, end_time, freq='1d')
df.set_index('date').reindex(time_list, fill_value=0)


# 7/3や7/4などの欠損データが0で穴埋めされている。
# 
# 意味がないが、reindexの引数のlistに応じてデータを並び替えることができるので、`time_list[::-1]`とすると、順序を逆にすることができる。

# In[32]:


df.set_index('date').reindex(time_list[::-1], fill_value=0)


# date_rangeもreindexも使用頻度は高くないので、忘れないようにする。
