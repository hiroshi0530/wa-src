
# coding: utf-8

# ## 第10章 アンケート分析を行うための自然言語処理10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# アンケート処理の演習になります。こちらも前の章よりはやりやすかったです。しかしとても勉強になるので、ぜひとも自分のものにしたいです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/10/10_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/10/10_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)


# ## 解答

# ### ノック 91 : データを読み込んで把握しよう

# In[4]:


survey = pd.read_csv('survey.csv')
survey.head()


# In[5]:


len(survey)


# In[6]:


survey.isna().sum()


# 実際のデータではユーザーがコメントしてくれない場合は多分にあります。

# In[7]:


survey = survey.dropna()
survey.isna().sum()


# In[8]:


len(survey)


# ### ノック 92 : 不要な文字を除去してみよう

# In[9]:


survey['comment'] = survey['comment'].str.replace('AA', '')
survey['comment'].head()


# 正規表現で括弧付きのパターンを削除します。

# In[10]:


survey['comment'] = survey['comment'].str.replace("\(.+?\)", "" ,regex=True)
survey['comment'].head()


# 大文字の括弧も削除の対象にします。

# In[11]:


survey['comment'] = survey['comment'].str.replace("\（.+?\）", "" ,regex=True)
survey['comment'].head()


# ### ノック 93 : 文字列をカウントしてヒストグラムを表示してみよう

# In[12]:


survey['length'] = survey['comment'].str.len()
survey.head()


# 分布を見たいのでヒストグラム表示してみます。

# In[13]:


plt.grid()
plt.hist(survey['length'], bins=10)
plt.show()


# ### ノック 94 : 形態素解析で文書を解析してみよう

# In[14]:


import MeCab
tagger = MeCab.Tagger()
text = 'すもももももももものうち'
words = tagger.parse(text)
words


# 単語ごとに分割し、特定の品詞の単語のみを取得します。

# In[15]:


words = tagger.parse(text).splitlines()
words_arr = []
for i in words:
  if i == 'EOS': 
    continue
  word_tmp = i.split()[0]
  words_arr.append(word_tmp)
words_arr


# ### ノック 95 : 形態素解析で文章から「動詞・名詞」を抽出してみよう

# In[16]:


text = 'すもももももももものうち'
words = tagger.parse(text).splitlines()
words_arr = []
parts = ['名詞', '動詞']

for i in words:
  if i == 'EOS' or i == '':
    continue
  
  word_tmp = i.split()[0]
  part = i.split()[1].split(',')[0]
  
  if not (part in parts): 
    continue
  words_arr.append(word_tmp)
words_arr


# ### ノック 96 : 形態素解析で抽出した頻出する名詞を確認してみよう

# In[17]:


all_words = []
parts = ['名詞']

for n in range(len(survey)):
  text = survey['comment'].iloc[n]
  words = tagger.parse(text).splitlines()
  
  words_arr = []
  
  for i in words:
    
    if i == 'EOS' or i == '':
      continue
    
    word_tmp = i.split()[0]
    part = i.split()[1].split(',')[0]
    
    if not (part in parts): 
      continue
    words_arr.append(word_tmp)
  
  all_words.extend(words_arr)
  
all_words[0:10]


# In[18]:


all_words_df = pd.DataFrame({'words': all_words, "count": len(all_words) * [1]})
all_words_df.head()


# In[19]:


all_words_df = all_words_df.groupby('words').sum()
all_words_df.head()


# In[20]:


all_words_df.sort_values('count', ascending=False).head()


# すべての単語に1を与え、groupbyで合計値を取ってソートします。

# ### ノック 97 : 関係のない単語を除去してみよう
# 
# ストップワードを設定します。関係の単語で除去すべき単語という意味です。

# In[21]:


stop_words = ['の']

all_words = []
parts = ['名詞']

for n in range(len(survey)):
  text = survey['comment'].iloc[n]
  words = tagger.parse(text).splitlines()
  
  words_arr = []
  
  for i in words:
    if i == 'EOS' or i == '':
      continue
    word_tmp = i.split()[0]
    
    part = i.split()[1].split(",")[0]
    if not (part in parts):
      continue
    
    if word_tmp in stop_words:
      continue
    
    words_arr.append(word_tmp)
  
  all_words.extend(words_arr)

all_words[0:10]


# In[22]:


all_words_df = pd.DataFrame({'words': all_words, "count": len(all_words) * [1]})
all_words_df = all_words_df.groupby('words').sum()
# print(all_words_df)
all_words_df.sort_values('count', ascending=False).head()


# 「の」が削除され、「公園」が繰り上がっています。

# ### ノック 98 : 顧客満足度と頻出単語の関係を見てみよう

# In[23]:


stop_words = ['の']

all_words = []
satisfaction = []

parts = ['名詞']

for n in range(len(survey)):
  text = survey['comment'].iloc[n]
  words = tagger.parse(text).splitlines()
  
  words_arr = []
  
  for i in words:
    if i == 'EOS' or i == '':
      continue
    word_tmp = i.split()[0]
    
    part = i.split()[1].split(",")[0]
    if not (part in parts):
      continue
    
    if word_tmp in stop_words:
      continue
    
    words_arr.append(word_tmp)
    satisfaction.append(survey['satisfaction'].iloc[n])
  all_words.extend(words_arr)

all_words_df = pd.DataFrame({"words": all_words, "satisfaction": satisfaction, "count": len(all_words) * [1]}) 
  
all_words_df.head()


# In[24]:


words_satisfaction = all_words_df.groupby('words').mean()['satisfaction']
words_count = all_words_df.groupby('words').sum()['count']

words_df = pd.concat([words_satisfaction, words_count], axis=1)
words_df.head()


# In[25]:


words_df = words_df.loc[words_df['count'] >= 3]
words_df.sort_values('satisfaction', ascending=False).head()


# In[26]:


words_df.sort_values('satisfaction').head()


# ### ノック 99 : アンケート毎の特徴を表現してみよう
# 類似した文章の検索になります。

# In[27]:


parts = ['名詞']
all_words_df = pd.DataFrame()
satisfaction = []

for n in range(len(survey)):
  text = survey['comment'].iloc[n]
  words = tagger.parse(text).splitlines()
  words_df = pd.DataFrame()
  
  for i in words:
    if i == 'EOS' or i == '':
      continue
    word_tmp = i.split()[0]
    
    part = i.split()[1].split(",")[0]
    if not (part in parts):
      continue
    
    if word_tmp in stop_words:
      continue
    
    words_df[word_tmp] = [1]
  
  all_words_df = pd.concat([all_words_df, words_df], ignore_index=True, sort=False)
all_words_df.head()


# fillnaでNaN部分を0に補完します。

# In[28]:


all_words_df = all_words_df.fillna(0)
all_words_df.head()


# ### ノック 100 : 類似アンケートを探してみよう
# 
# cos類似度を使ってターゲットと似たような文章を検索します。

# In[29]:


print(survey['comment'].iloc[2])


# In[30]:


target_text = all_words_df.iloc[2]
target_text.head()


# コサイン類似度で文章の類似度を定量化します。

# In[31]:


cos_sim = []
for i in range(len(all_words_df)):
  cos_text = all_words_df.iloc[i]
  cos = np.dot(target_text, cos_text) / (np.linalg.norm(target_text) * np.linalg.norm(cos_text))
  
  cos_sim.append(cos)

all_words_df['cos_sim'] = cos_sim
all_words_df.sort_values('cos_sim', ascending=False).head()
  


# 上位のコメント表示してみます。上記を見ると、25,15,33,50となっています。

# In[32]:


print(survey['comment'].iloc[2])
print(survey['comment'].iloc[24])
print(survey['comment'].iloc[15])
print(survey['comment'].iloc[33])


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
