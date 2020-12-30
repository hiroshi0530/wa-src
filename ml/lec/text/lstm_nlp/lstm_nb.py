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
# 文章を学習させるには、日本語の文字1文字1文字をスカラーかやベクトルとして表現するする必要があります。前回やったとおりword2vecを用いてベクトル表現を得る方法もありますが、ここでは、それぞれの文字にスカラーである数字を割り当てて学習させます。
# 
# まず、銀河鉄道の夜で利用されている文字をすべて取り出します。

# In[10]:


all_chars = sorted(list(set(all_sentence)))
all_chars[:10]


# 次に、文字をスカラーに変換するための辞書を作成します。

# In[11]:


char_num_dic = dict((c, i) for i, c in enumerate(all_chars))
num_char_dic = dict((i, c) for i, c in enumerate(all_chars))


# 文字列を受け取って、スカラーのリストを返す関数を作成します。

# In[12]:


def get_scalar_list(char_list):
  return [char_num_dic[c] for c in char_list]


# この関数を利用し、予想に利用する文字列と予想する文字をスカラーのリストに変換します。LSTMで予測するのに必要な時系列データの数を100とします。また、100個の文字列から、次の1文字を予測するモデルを作成します。

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





# In[ ]:



# インデックスと文字で辞書を作成
chars = sorted(list(set(text)))  # setで文字の重複をなくし、各文字をリストに格納する
print("文字数（重複無し）", len(chars))
char_indices = {}  # 文字がキーでインデックスが値
for i, char in enumerate(chars):
  char_indices[char] = i
indices_char = {}  # インデックスがキーで文字が値
for i, char in enumerate(chars):
  indices_char[i] = char
 
# 時系列データと、それから予測すべき文字を取り出します
time_chars = []
next_chars = []
for i in range(0, len(text) - n_rnn):
  time_chars.append(text[i: i + n_rnn])
  next_chars.append(text[i + n_rnn])
 
# 入力と正解をone-hot表現で表します
x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
for i, t_cs in enumerate(time_chars):
  t[i, char_indices[next_chars[i]]] = 1  # 正解をone-hot表現で表す
  for j, char in enumerate(t_cs):
    x[i, j, char_indices[char]] = 1  # 入力をone-hot表現で表す
    
print("xの形状", x.shape)
print("tの形状", t.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 減衰振動曲線
# 
# サンプル用のデータとして、以下の式からサンプリングを行います。
# 
# $$
# y = \exp\left(-\frac{x}{\tau}\right)\cos(x) 
# $$
# 
# 波を打ちながら、次第に収束していく、自然現象ではよくあるモデルになります。単純なRNNと比較するため、サンプルデータは同じ関数とします。

# In[ ]:


x = np.linspace(0, 5 * np.pi, 200)
y = np.exp(-x / 5) * (np.cos(x))


# ### データの確認
# 
# $x$と$y$のデータの詳細を見てみます。

# In[ ]:


print('shape : ', x.shape)
print('ndim : ', x.ndim)
print('data : ', x[:10])


# In[ ]:


print('shape : ', y.shape)
print('ndim : ', y.ndim)
print('data : ', y[:10])


# グラフを確認してみます。

# In[ ]:


plt.plot(x,y)
plt.grid()
plt.show()


# $\tau=5$として、綺麗な減衰曲線が得られました。

# ## ニューラルネットの構築
# 
# kerasに投入するためにデータの前処理を行い、再帰型のニューラルネットの構築を行います。
# 
# 構築が終了したら、compileメソッドを利用して、モデルをコンパイルします。compileの仕様は以下の様になっています。
# 
# ```bash
# compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
# ```

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

NUM_RNN = 20
NUM_MIDDLE = 40

# データの前処理
n = len(x) - NUM_RNN
r_x = np.zeros((n, NUM_RNN))
r_y = np.zeros((n, NUM_RNN))
for i in range(0, n):
  r_x[i] = y[i: i + NUM_RNN]
  r_y[i] = y[i + 1: i + NUM_RNN + 1]

r_x = r_x.reshape(n, NUM_RNN, 1)
r_y = r_y.reshape(n, NUM_RNN, 1)

# RNNニューラルネットの構築
rnn_model = Sequential()
rnn_model.add(SimpleRNN(NUM_MIDDLE, input_shape=(NUM_RNN, 1), return_sequences=True))
rnn_model.add(Dense(1, activation="linear"))
rnn_model.compile(loss="mean_squared_error", optimizer="sgd")

# LSTMニューラルネットの構築
lstm_model = Sequential()
lstm_model.add(LSTM(NUM_MIDDLE, input_shape=(NUM_RNN, 1), return_sequences=True))
lstm_model.add(Dense(1, activation="linear"))
lstm_model.compile(loss="mean_squared_error", optimizer="sgd")


# 投入するデータや、モデルの概要を確認します。

# In[ ]:


print(r_y.shape)
print(r_x.shape)


# 二つのモデルの比較を行います。LSTMの方がパラメタ数が多いことがわかります。学習するにもLSTMの方が時間がかかります。

# In[ ]:


print(rnn_model.summary())
print(lstm_model.summary())


# ## 学習
# 
# fitメソッドを利用して、学習を行います。
# fitメソッドの仕様は以下の通りになっています。[こちら](https://keras.io/ja/models/sequential/)を参照してください。
# 
# ```bash
# fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
# ```

# In[ ]:


batch_size = 10
epochs = 1000

# validation_split で最後の10％を検証用に利用します
rnn_history = rnn_model.fit(r_x, r_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

# validation_split で最後の10％を検証用に利用します
lstm_history = lstm_model.fit(r_x, r_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)


# ## 損失関数の可視化
# 
# 学習によって誤差が減少していく様子を可視化してみます。

# In[ ]:


rnn_loss = rnn_history.history['loss'] # 訓練データの損失関数
rnn_val_loss = rnn_history.history['val_loss'] #テストデータの損失関数

lstm_loss = lstm_history.history['loss'] # 訓練データの損失関数
lstm_val_loss = lstm_history.history['val_loss'] #テストデータの損失関数

plt.plot(np.arange(len(rnn_loss)), rnn_loss, label='rnn_loss')
plt.plot(np.arange(len(rnn_val_loss)), rnn_val_loss, label='rnn_val_loss')
plt.plot(np.arange(len(lstm_loss)), lstm_loss, label='lstm_loss')
plt.plot(np.arange(len(lstm_val_loss)), lstm_val_loss, label='lstm_val_loss')
plt.grid()
plt.legend()
plt.show()


# ## 結果の確認

# In[ ]:


# 初期の入力値
rnn_res = r_y[0].reshape(-1)
lstm_res = r_y[0].reshape(-1)

for i in range(0, n):
  _rnn_y = rnn_model.predict(rnn_res[- NUM_RNN:].reshape(1, NUM_RNN, 1))
  rnn_res = np.append(rnn_res, _rnn_y[0][NUM_RNN - 1][0])
  
  _lstm_y = lstm_model.predict(lstm_res[- NUM_RNN:].reshape(1, NUM_RNN, 1))
  lstm_res = np.append(lstm_res, _lstm_y[0][NUM_RNN - 1][0])
  
plt.plot(np.arange(len(y)), y, label=r"$\exp\left(-\frac{x}{\tau}\right) \cos x$")
plt.plot(np.arange(len(rnn_res)), rnn_res, label="RNN result")
plt.plot(np.arange(len(lstm_res)), lstm_res, label="LSTM result")
plt.legend()
plt.grid()
plt.show()


# 減衰振動曲線の場合、今回設定したパラメタでは、LSTMとRNNの差は出ていないようです。ただ、実務レベルでは、RNNよりLSTMの方がより使われており、結果も出ているように思います。今回はただの練習なので、ここで終わりにしようと思います。
