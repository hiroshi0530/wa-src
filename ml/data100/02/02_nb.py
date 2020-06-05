#!/usr/bin/env python
# coding: utf-8

# ## 第2章 小売店のデータでデータ加工を行う10本ノック
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/02/02_nb.ipynb)
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pandas as pd

pd.__version__


# In[4]:


import matplotlib

matplotlib.__version__


# matplotlibを呼び込み、保存する画像をsvgに設定します。

# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# ## 解答

# ### ノック11 : データを読み込んでみよう

# In[6]:


uriage_data = pd.read_csv('uriage.csv')
uriage_data.head()


# In[7]:


kokyaku_data = pd.read_excel('kokyaku_daicho.xlsx')
kokyaku_data.head()


# ### ノック12 : データの揺れを見てみよう

# In[8]:


uriage_data['item_name'].head()


# In[9]:


uriage_data['item_price'].head()


# かなりデータが揺れがあるのがわかります。

# ### ノック13 : データに揺れがあるまま集計してみよう 

# In[10]:


uriage_data['purchase_date'] = pd.to_datetime(uriage_data['purchase_date'])


# In[17]:


uriage_data['purchase_date']


# datetime型に変換されています。

# In[18]:


uriage_data['purchase_month'] = uriage_data['purchase_date'].dt.strftime('%Y%m')
res = uriage_data.pivot_table(index='purchase_month', columns='item_name', aggfunc='size', fill_value=0)
res


# In[20]:


res.shape


# 商品数が99個になっています。
# 次に価格についても見てみます。

# In[21]:


res = uriage_data.pivot_table(index='purchase_month', columns='item_name', values='item_price', aggfunc='sum', fill_value=0)
res


# ### ノック14 : 

# In[ ]:





# ### ノック15 : 

# In[ ]:





# ### ノック16 : 

# In[ ]:





# ### ノック17 : 

# In[ ]:





# ### ノック18 : 

# In[ ]:





# ### ノック19 : 

# In[ ]:





# ### ノック20 : 

# In[ ]:





# ## 関連記事
# - [第1章 ウェブからの注文数を分析する10本ノック](/ml/data100/01/)
# - [第2章 小売店のデータでデータ加工を行う10本ノック](/ml/data100/02/)
# - [第3章 顧客の全体像を把握する10本ノック](/ml/data100/03/)
# - [第4章 顧客の行動を予測する10本ノック](/ml/data100/04/)
# - [第5章 顧客の退会を予測する10本ノック](/ml/data100/05/)
# - [第6章 物流の最適ルートをコンサルティングする10本ノック](/ml/data100/06/)
# - [第7章 ロジスティクスネットワークの最適設計を行う10本ノック](/ml/data100/07/)
# - [第8章 数値シミュレーションで消費者行動を予測する10本ノック](/ml/data100/08/)
# - [第9章 潜在顧客を把握するための画像認識10本ノック](/ml/data100/09/)
# - [第10章 アンケート分析を行うための自然言語処理10本ノック](/ml/data100/10/)
