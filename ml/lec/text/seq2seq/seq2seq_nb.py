#!/usr/bin/env python
# coding: utf-8

# ## kerasとsequnece to sequence
# 
# 前回、LSTMによる実装を行いましたので、次はsquence to sequenceモデルを実装していこうと思います。今現在では、機械翻訳などの自然言語処理では、このsequnece to sequenceとAttentionを基本としたモデルがよく利用されています。BERTなどもAttentionモデルが基本となっています。
# 
# ここでは、復習もかねて、基本的なsequnece to sequenceを実装します。$y=\exp x$を$y=\log x$に翻訳するモデルの構築を行います。なお、モデルの詳細は検索すればいくらでも出てきますのでここでは割愛します。文献や教科書、技術者によっては、sequnece to sequenceモデルは、「Encoder-Decoderモデル」、「系列変換モデル」などと呼ばれることも多いようです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/seq2seq/seq2seq_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/seq2seq/seq2seq_nb.ipynb)
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

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)
print('gensim version : ', gensim.__version__)


# ## sequence to sequence モデルの入出力データ
# 
# 最初にsequence to sequence モデルのアルゴリズムの概要と、それをkerasで実行する場合、どのようなデータの入出力になるのかを簡単に説明しようと思います。
# 
# ### データの入出力のイメージ
# 
# ![svg](seq2seq_nb_files_local/seq2seq.svg)
# 
# sequence to sequence はencoderとdecoderという二つの部分で構成されています。それぞれの部分はRNNやLSTMなどのモデルで構築されます。
# このような特徴から時系列データの解析に強く、機械翻訳や音声認識などの分野で利用されているようです。
# 
# kerasでseq2seqを実装するには、encoderとdecoderそれぞれへの入力データ（図で言うdataset 1と2）と正解データ（dataset 3)が必要になります。
# 
# 正解データは、decoderへの入力セットから時系列的に一つずれていることがポイントになります。
# 
# 以下のサイトを参考にさせていただきました。
# 
# - https://blog.octopt.com/sequence-to-sequence/
# 
# 

# ## サンプルデータ
# 
# サンプル用のデータとして、以下の式を利用します。
# 
# $$
# encoder_y = \exp x
# $$
# 
# $$
# dencoder_y = \log x
# $$
# 

# In[4]:


x = np.linspace(0, 3, 200)
seq_in = np.exp(x)
seq_out = np.log(x)


# ### データの確認
# 
# $x$と$y$のデータの詳細を見てみます。

# In[5]:


print('shape : ', x.shape)
print('ndim : ', x.ndim)
print('data : ', x[:10])


# In[6]:


print('shape : ', seq_in.shape)
print('ndim : ', seq_in.ndim)
print('data : ', seq_in[:10])


# In[8]:


print('shape : ', seq_out.shape)
print('ndim : ', seq_out.ndim)
print('data : ', seq_out[:10])


# グラフを確認してみます。

# In[9]:


plt.plot(x, seq_in, label='$y=\exp x$')
plt.plot(x, seq_out, label='$y=\log x$')
plt.legend()
plt.grid()
plt.show()


# ### データの準備
# 
# kerasに入力するためのデータをnumpy配列に格納します。

# In[12]:


NUM_LSTM = 10

n = len(x) - NUM_LSTM
ex = np.zeros((n, NUM_LSTM))
dx = np.zeros((n, NUM_LSTM))
dy = np.zeros((n, NUM_LSTM))

for i in range(0, n):
  ex[i] = seq_in[i:i+NUM_LSTM]
  dx[i, 1:] = seq_out[i:i+NUM_LSTM-1]
  dy[i] = seq_out[i:i+NUM_LSTM]

ex = ex.reshape(n, NUM_LSTM, 1)
dx = dx.reshape(n, NUM_LSTM, 1)
dy = dy.reshape(n, NUM_LSTM, 1)


# ## モデルの構築
# 
# sequence to sequenceのモデルをkerasを用いて実装します。
# 単純なRNNやLSTMとは異なり、モデルが複数あり、それぞれから入力する必要があるため、Sequenceではなく、Modelを利用します。

# In[15]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

NUM_MID = 20

def build_seq2se2_model():
  e_input = Input(shape=(NUM_LSTM, 1))
  e_lstm = LSTM(NUM_MID, return_state=True)
  e_output, e_state_h, e_state_c = e_lstm(e_input)
  e_state = [e_state_h, e_state_c]
  
  d_input = Input(shape=(NUM_LSTM, 1))
  d_lstm = LSTM(NUM_MID, return_sequences=True, return_state=True)
  d_output, _, _ = d_lstm(d_input, initial_state=e_state)
  d_dense = Dense(1, activation='linear')
  d_output = d_dense(d_output)
  
  model = Model([e_input, d_input], d_output)
  model.compile(loss="mean_squared_error", optimizer="adam")
  print(model.summary())
  
  return model

seq2seq_model = build_seq2se2_model()


# ## モデルの学習

# In[19]:


# 学習用のパラメタを設定します
batch_size = 10
epochs = 20

history = seq2seq_model.fit([ex, dx], dy, epochs=epochs, batch_size=8)


# ## 損失関数
# 
# 損失関数が減少していく様子を可視化してみます。

# In[21]:


loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()


# In[ ]:





# ## 予測するためのencoderとdecoderのモデルを返す関数を作成します

# In[22]:


def build_predict_encoder_d_model():
  e_model = Model(e_input, e_state)
  
  d_input = Input(shape=(1, 1))
  d_state_in_h = Input(shape=(NUM_MID,))
  d_state_in_c = Input(shape=(NUM_MID,))
  d_state_in = [d_state_in_h, d_state_in_c]
  
  d_output, d_state_h, d_state_c = d_lstm(d_input, initial_state=d_state_in)
  d_state = [d_state_h, d_state_c]
  
  d_output = d_dense(d_output)
  d_model = Model([d_input] + d_state_in, [d_output] + d_state)
  
  return e_model, d_model


# データを変換するための関数を実装します。

# In[ ]:


def translate(input_data):
  state_value = encoder_model.predict(input_data)
  y_decoder = np.zeros((1, 1, 1))
  translated = []
  
  for i in range(0, n_rnn):  # 各時刻ごとに予測を行う
    y, h, c = decoder_model.predict([y_decoder] + state_value)
    y = y[0][0][0]
    translated.append(y)
    y_decoder[0][0][0] = y
    state_value = [h, c]

  return translated


# 結果の確認

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




