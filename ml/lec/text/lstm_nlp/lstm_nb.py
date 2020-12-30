#!/usr/bin/env python
# coding: utf-8

# ## kerasとLSTMを用いた文章の生成
# 
# LSTMを用いて文章を生成することが出来ます。文章を時系列データとして訓練データとして学習し、文章を入力し、次の文字列を予測するようなっモデルを生成します。今回は前回青空文庫からダウンロードした、宮沢賢治の銀河鉄道の夜を学習データとして採用し、LSTMによって、宮沢賢治風の文章を作成してみようと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_nlp/lstm_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_nlp/lstm_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

import tensorflow as tf
from tensorflow import keras
import gensim
import gensim

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)
print('gensim version : ', gensim.__version__)


# ## テキストファイルの前処理
# 
# 題材として、宮沢賢治の銀河鉄道の夜を利用します。既に著作権フリーなので、自由に利用できます。ちなみに、宮沢賢治は同郷で高校の先輩ですが、日本語が全く出来ない私は一度も読んだことはないです。ですので、LSTMによる文章が自然なものなのか、宮沢賢治風なのか、不明です。。
# 
# 銀河鉄道の夜は以前、word2vecを利用した[分散表現の作成の記事](/ml/lec/text/w2v/)で利用しました。
# テキストの前処理などは重複する部分があるかと思います。
# 
# まずはテキストの中身を見てみます。

# In[4]:


get_ipython().run_cell_magic('bash', '', 'cat ginga.txt | head -n 25')


# In[5]:


get_ipython().run_cell_magic('bash', '', 'cat ginga.txt | tail -n 25')


# となり、ファイルの先頭と、末尾に参考情報が載っているほかは、ちゃんとテキストとしてデータが取れている模様です。
# 先ず、この辺の前処理を行います。

# In[6]:


import re

with open('ginga.txt', mode='r') as f:
  all_sentence = f.read()


# In[7]:


all_sentence = all_sentence.replace(" ", "").replace("　","").replace("\n","").replace("|","")


# 《》で囲まれたルビの部分を削除します。正規表現を利用します。

# In[8]:


all_sentence = re.sub("《[^》]+》", "", all_sentence)


# ----------の部分で分割を行い、2番目の要素を取得します。

# In[9]:


all_sentence = re.split("\-{8,}", all_sentence)[2]
all_sentence[:100]


# となり、不要な部分を削除し、必要な部分をall_sentenceに格納しました。

# ## one hot vectorの作成
# 
# 文章を学習させるには、日本語の文字1文字1文字をベクトルとして表現する必要があります。前回やったとおりword2vecを用いてベクトル表現を得る方法もありますが、ここでは、それぞれの文字に対して、`[0,0,1,0,0]`などのone-hot-vectorを付与します。ですので、ベクトルの次元数としては、文字数分だけあり、学習にかなりの時間を要します。
# 
# まず、銀河鉄道の夜で利用されている文字をすべて取り出します。

# In[10]:


all_chars = sorted(list(set(all_sentence)))
all_chars[:10]


# 次に、文字に対して数字を対応させます。上記の`all_chars`に格納された順番の数字を付与します。

# In[11]:


char_num_dic = dict((c, i) for i, c in enumerate(all_chars))
num_char_dic = dict((i, c) for i, c in enumerate(all_chars))


# 後の処理を簡単にするために、文字列を受け取って、対応する数字のリストを返す関数を作成します。

# In[12]:


def get_scalar_list(char_list):
  return [char_num_dic[c] for c in char_list]


# この関数を利用し、予想に利用する文字列と予想する文字を数字のリストに変換します。
# 
# また、LSTMで予測するのに必要な時系列データの数を100とします。
# 100個の文字列から、次の1文字を予測するモデルを作成します。

# In[13]:


NUM_LSTM = 100

train_chars_list = []
predict_char_list = []
for c in range(0, len(all_sentence) - NUM_LSTM):
  train_chars_list.append(get_scalar_list(all_sentence[c: c + NUM_LSTM]))
  predict_char_list.append(char_num_dic[all_sentence[c + NUM_LSTM]])


# In[14]:


print(train_chars_list[0])


# In[15]:


print(predict_char_list[0])


# train_chars[0]からpredict_char[0]を予測するようなモデルを作成します。
# 
# これらの数字をone hot vectorで表現します。
# 
# 表現するベクトルのサイズは`len(all_chars)`となります。また、kerasに投入することを前提に、入力するテンソルの形状として
# 
# `(サンプル数、予測に利用する時系列データの数、one-hot-vectorの次元)`となります。

# In[16]:


# xを入力するデータ
# yを正解データ
# one-hot-vectorを入力するため、最初にゼロベクトルを作成します。

x = np.zeros((len(train_chars_list), NUM_LSTM, len(all_chars)), dtype=np.bool)
y = np.zeros((len(predict_char_list), len(all_chars)), dtype=np.bool)


# 必要な部分だけ1に修正します。

# In[18]:


# 入力データに割り当てられた数字の要素を1に設定します。
for i in range(len(train_chars_list)):
  for j in range(NUM_LSTM):
    x[i, j, train_chars_list[i][j]] = 1

# 正解データに割り当てられた数字の要素を1に設定します。
for i in range(len(predict_char_list)):
  y[i, predict_char_list[i]] = 1


# ## one-hot-vectorの確認
# 
# 実際に想定通りone-hot-vectorが出来ているか確認してみます。`np.where`を利用してtrueとなっているインデックスを取得してみます。

# In[55]:


np.where(x[0][:-1] == 1)[1]


# In[52]:


np.where(y[0] == 1)


# となり、想定通りone-hot-vectorが出来ていることがわかりました。

# ## モデルの構築
# 
# LSTMのモデルを構築する関数を作成します。
# ここでは簡単にLSTMと全結合層で構成されたモデルを作成します。

# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

NUM_MIDDLE = 300

def build_lstm_model():
  lstm_model = Sequential()
  lstm_model.add(LSTM(NUM_MIDDLE, input_shape=(NUM_LSTM, len(all_chars))))
  lstm_model.add(Dense(len(all_chars), activation="softmax"))
  lstm_model.compile(loss='categorical_crossentropy', optimizer="adam")
  
  print(lstm_model.summary())
  
  return lstm_model

model = build_lstm_model()


# In[ ]:





# epoch終了後に実行させるコールバック関数を実行させます

# In[22]:


from tensorflow.keras.callbacks import LambdaCallback
 
def on_epoch_end(epoch, logs):
  print("エポック: ", epoch)

  beta = 5  # 確率分布を調整する定数
  prev_text = text[0: NUM_LSTM]  # 入力に使う文字
  created_text = prev_text  # 生成されるテキスト
  
  print("シード: ", created_text)

  for i in range(400):
    # 入力をone-hot表現に
    x_pred = np.zeros((1, NUM_LSTM, len(all_chars)))
    for j, char in enumerate(prev_text):
      x_pred[0, j, char_indices[char]] = 1
    
    # 予測を行い、次の文字を得る
    y = model.predict(x_pred)
    p_power = y[0] ** beta  # 確率分布の調整
    next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))    
    next_char = indices_char[next_index]

    created_text += next_char
    prev_text = prev_text[1:] + next_char

  print(created_text)

# エポック終了後に実行される関数を設定
epoch_end_callback= LambdaCallback(on_epoch_end=on_epoch_end)


# In[25]:


## とても時間がかかる

epochs = 10
batch_size = 100

history = model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[epoch_end_callback])


# In[ ]:





# In[ ]:





# In[ ]:




