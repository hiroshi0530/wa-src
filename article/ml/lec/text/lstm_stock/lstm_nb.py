#!/usr/bin/env python
# coding: utf-8

# ## RNN, LSTMを使った株価予測
# 
# RNNやLSTMは時系列データの予測のために利用されます。時系列データには、ある場所の気温や、来客数、商品の価格など多岐にわたりますが、最もデータを入手しやすい株価をRNNとLSTMで予測を行ってみたいと思います。
# 
# ただし、ニューラルネットはあくまでも得られたデータの範囲内でしか予測する事が出来ず、想定外の状況になった場合、そのモデルはほぼ意味をなしません。例えば、コロナショック前の1年前のデータを用いても、コロナショックを予測する事は出来ません。
# 
# また、株価の形成はテクニカルな要素だけでなく、ファンダメンタルズ、実需や先物などの複雑な要素もあり、LSTMで未来を予測するのは難しいとは思います。とはいえ、面白そうなので、年末の時間を利用してLSTMに慣れるためにもやってみようと思います。
# 
# あくまでもRNNやLSTMに慣れる練習の一環ですので、この結果をもって株価が予測できるなどとは思わないでください。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_stock/lstm_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_stock/lstm_nb.ipynb)
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
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)


# ## データの取得
# 
# 今回は日経平均とアメリカのS&P500の株価のデータの予測を行います。データはそれぞれ以下のサイトからダウンロードしました。
# 
# ### 日経平均のデータ
# 
# - https://indexes.nikkei.co.jp/nkave/index?type=download
# 
# ### SP500のデータ
# 
# - https://kabuoji3.com/stock/download.php
# 
# 
# ## 日経平均の予測
# 
# ### データの確認
# まず最初に日経のデータを見てみます。

# In[4]:


get_ipython().system('ls ')


# In[5]:


get_ipython().run_cell_magic('bash', '', 'head nikkei.csv')


# 文字コードがshift-jisになっているので、utf-8に直します。

# In[6]:


get_ipython().run_cell_magic('bash', '', 'nkf --guess nikkei.csv')


# In[7]:


get_ipython().run_cell_magic('bash', '', 'nkf -w nikkei.csv > nikkei_utf8.csv')


# In[8]:


get_ipython().run_cell_magic('bash', '', 'head nikkei_utf8.csv')


# 問題ないようなので、pandasで読み込みます。

# In[9]:


df = pd.read_csv('nikkei_utf8.csv')


# In[10]:


df.head()


# In[11]:


df.tail()


# 最後の行に著作権に関する注意書きがありますが、これを削除します。複写や流布は行いません。

# In[12]:


df.drop(index=975, inplace=True)


# In[13]:


df.tail()


# データを可視化してみます。コロナショックで大きくへこんでいることがわかりますが、2020年の年末の時点では金融緩和の影響を受けて大幅に上がっています。

# ### データの整形
# 
# 最初のデータを基準に、その値からの変化率を計算し、そのリストに対して学習を行います。

# In[14]:


def shape_data(data_list):
  return [d / data_list[0] - 1 for d in data_list]

df['data_list'] = shape_data(df['終値'])


# In[15]:


ticks = 10
xticks = ticks * 5 

plt.plot(df['データ日付'][::ticks], df['終値'][::ticks], label='nikkei stock')
plt.grid()
plt.legend()
plt.xticks(df['データ日付'][::xticks], rotation=60)
plt.show()


# 比率に直したグラフも示しておきます。

# In[16]:


plt.plot(df.index.values[::ticks], df['data_list'][::ticks], label='nikkei stock')
plt.grid()
plt.legend()
plt.show()


# ### 定数の準備

# In[17]:


# データとしては約四年分あるが、今回はこれを8このパートに分けて、それぞれの領域で予想を行う
TERM_PART_LIST = [0, 120, 240, 360, 480, 600, 720, 840]

# 予測に利用するデータ数
# 90個のデータから後の30個のデータを予測する
NUM_LSTM = 90

# 中間層の数
NUM_MIDDLE = 200

# ニューラルネットのモデルの定数
batch_size = 100
epochs = 2000
validation_split = 0.25


# ### データの準備
# 
# kerasに投入するためにデータを整えます。

# In[18]:


def get_x_y_lx_ly(term_part):
  
  date = np.array(df['データ日付'][TERM_PART_LIST[term_part]: TERM_PART_LIST[term_part + 1]])
  x = np.array(df.index.values[TERM_PART_LIST[term_part]: TERM_PART_LIST[term_part + 1]])
  y = np.array(df['data_list'][TERM_PART_LIST[term_part]: TERM_PART_LIST[term_part + 1]])
  
  n = len(y) - NUM_LSTM
  l_x = np.zeros((n, NUM_LSTM))
  l_y = np.zeros((n, NUM_LSTM))
  
  for i in range(0, n):
    l_x[i] = y[i: i + NUM_LSTM]
    l_y[i] = y[i + 1: i + NUM_LSTM + 1]
  
  l_x = l_x.reshape(n, NUM_LSTM, 1)
  l_y = l_y.reshape(n, NUM_LSTM, 1)
  
  return n, date, x, y, l_x, l_y

n, date, x, y, l_x, l_y = get_x_y_lx_ly(0)


# In[19]:


print('shape : ', x.shape)
print('ndim : ', x.ndim)
print('data : ', x[:10])


# In[20]:


print('shape : ', y.shape)
print('ndim : ', y.ndim)
print('data : ', y[:10])


# In[21]:


print(l_y.shape)
print(l_x.shape)


# ### モデルの構築
# 
# モデルの構築を定義する関数です。デフォルトではRNNとします。

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU


def build_model(model_name='RNN'):
  # LSTMニューラルネットの構築
  model = Sequential()
  
  # RNN,LSTM、GRUを選択できるようにする
  if model_name == 'RNN':
    model.add(SimpleRNN(NUM_MIDDLE, input_shape=(NUM_LSTM, 1), return_sequences=True))
  
  if model_name == 'LSTM':
    model.add(LSTM(NUM_MIDDLE, input_shape=(NUM_LSTM, 1), return_sequences=True))
  
  if model_name == 'GRU':
    model.add(GRU(NUM_MIDDLE, input_shape=(NUM_LSTM, 1), return_sequences=True))
  
  model.add(Dense(1, activation="linear"))
  model.compile(loss="mean_squared_error", optimizer="sgd")
  
  return model


# ニューラルネットを深くした（今回は使わない）
def build_model_02(): 
  
  NUM_MIDDLE_01 = 100
  NUM_MIDDLE_02 = 120
  
  # LSTMニューラルネットの構築
  model = Sequential()
  model.add(LSTM(NUM_MIDDLE_01, input_shape = (NUM_LSTM, 1), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(NUM_MIDDLE_02, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.add(Activation("linear"))
  model.compile(loss="mean_squared_error", optimizer="sgd")
  # model.compile(loss="mse", optimizer='rmsprop')
    
  return model
  
model = build_model('RNN')


# ### モデルの詳細

# In[23]:


print(model.summary())


# In[24]:


# validation_split で最後の10％を検証用に利用します
history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)


# ### 損失関数の可視化
# 
# 学習によって誤差が減少していく様子を可視化してみます。今のエポック数で収束しているように見えます。

# In[25]:


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(np.arange(len(loss)), loss, label='loss')
plt.plot(np.arange(len(val_loss)), val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.show()


# ### RNNによる結果の確認
# 
# 薄いオレンジに塗りつぶされた期間が予測のために利用した期間です。その期間は、実際の推移と予測が一致しています。オレンジの実線が実際の株価推移、青が予測です。

# In[26]:


def plot_result():

  # 初期の入力値
  res = []
  res = np.append(res, l_x[0][0][0])
  res = np.append(res, l_y[0].reshape(-1))
  
  for i in range(0, n):
    _y = model.predict(res[- NUM_LSTM:].reshape(1, NUM_LSTM, 1))
    
    # 予測されたデータを次の予測のためのインプットデータとして利用
    res = np.append(res, _y[0][NUM_LSTM - 1][0])
  
  res = np.delete(res, -1)  
  
  plt.plot(date, y, label="stock price", color='coral')
  plt.plot(date, res, label="prediction result", color='blue')
  plt.xticks(date[::12], rotation=60)
  
  plt.legend()
  plt.grid()
  
  plt.axvspan(0, NUM_LSTM, color="coral", alpha=0.2)
  
  plt.show()
  
print('{} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
plot_result()


# 結果としてはどうでしょうか？まぁトレンドは大きく外していないかなという程度でしょうか笑

# ### 他の期間の予測
# 
# これまでの関数を使って、他の期間の予測もしてみます。

# In[27]:


for term in [1, 2, 3, 4, 5, 6]:
  n, date, x, y, l_x, l_y = get_x_y_lx_ly(term)
  model = build_model('RNN')
  history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
  print('予測期間 : {} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
  plot_result()


# ### LSTMによる予測

# In[28]:


for term in [0, 1]:
  n, date, x, y, l_x, l_y = get_x_y_lx_ly(term)
  model = build_model('LSTM')
  history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
  print('予測期間 : {} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
  plot_result()


# LSTMでは今回の行った簡単なモデルでは、ほとんど予測できませんでした。よってグラフも二つしか示していません。もう少し考察すれば良さそうですが、今回の目的からはそれるので辞めておきます。

# ### GRUによる予測

# In[29]:


for term in [0, 1]:
  n, date, x, y, l_x, l_y = get_x_y_lx_ly(term)
  model = build_model('GRU')
  history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
  print('予測期間 : {} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
  plot_result()


# GRUでも意味のある結果が得られませんでした。

# ## S&P500の予測
# 
# ### 2019年
# 同じようにアメリカの代表的な株価指数であるS&P500についても予測してみます。
# ファイルは上記のサイトからダウンロード出来ます。

# In[30]:


get_ipython().system('ls')


# ファイルの中身を簡単に見てみます。

# In[31]:


get_ipython().run_cell_magic('bash', '', 'head sp500_2019.csv')


# 文字コードがShift-JISのようなので、utf-8に置換します。

# In[32]:


get_ipython().run_cell_magic('bash', '', 'nkf -w sp500_2019.csv > sp500_2019_utf8.csv')


# さらに見てみると、1行目がpandasに入れるのに余計なので、削除します。

# In[33]:


get_ipython().run_cell_magic('bash', '', 'head sp500_2019_utf8.csv')


# In[34]:


get_ipython().run_cell_magic('bash', '', "sed -ie '1d' sp500_2019_utf8.csv ")


# In[35]:


get_ipython().run_cell_magic('bash', '', 'head sp500_2019_utf8.csv')


# 準備が整ったので、pandasに入れます。

# In[36]:


df = pd.read_csv('sp500_2019_utf8.csv')


# In[37]:


df.head()


# In[38]:


df.tail()


# 日経平均と同様に、終値を変化率に変換します。同じ関数を利用します。

# In[39]:


df['data_list'] = shape_data(df['終値'])


# また、先ほどの関数を再利用したいので、日付というカラム名をデータ日付と言うカラム名に変更します。

# In[40]:


df = df.rename(columns={'日付':'データ日付'})


# In[41]:


df.head()


# In[42]:


df.tail()


# 全体のグラフを俯瞰しています。

# In[43]:


plt.plot(df['データ日付'][::ticks], df['終値'][::ticks], label='sp500 2019')
plt.grid()
plt.legend()
plt.xticks(df['データ日付'][::xticks], rotation=60)
plt.show()


# 予測を行って、結果をグラウかしてみます。

# In[44]:


for term in [0, 1]:
  n, date, x, y, l_x, l_y = get_x_y_lx_ly(term)
  model = build_model('RNN')
  history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
  print('予測期間 : {} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
  plot_result()


# 日経平均と同様、トレンドに沿って予測しており、逆張り防止にはなるかもしれません笑

# ### 2020年
# 
# 次に2020年の株価について予測を行ってみます。データの前処理などは省略します。

# In[45]:


get_ipython().run_cell_magic('bash', '', "head sp500_2020_utf8.csv\nnkf -w sp500_2020.csv > sp500_2020_utf8.csv\nsed -ie '1d' sp500_2020_utf8.csv ")


# In[46]:


df = pd.read_csv('sp500_2020_utf8.csv')
df.head()


# In[47]:


df['data_list'] = shape_data(df['終値'])
df = df.rename(columns={'日付':'データ日付'})
df.head()


# In[48]:


df.tail()


# In[49]:


plt.plot(df['データ日付'][::ticks], df['終値'][::ticks], label='sp500 2020')
plt.grid()
plt.legend()
plt.xticks(df['データ日付'][::xticks], rotation=60)
plt.show()


# In[50]:


for term in [0, 1]:
  n, date, x, y, l_x, l_y = get_x_y_lx_ly(term)
  model = build_model('RNN')
  history = model.fit(l_x, l_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
  print('予測期間 : {} - {} の結果'.format(date[0], date[NUM_LSTM - 1]))
  plot_result()


# ## まとめ
# 
# 特徴量抽出、モデル検討、ハイパーパラメタの調整などやれることはたくさんあると思いますが、目的はkerasに慣れる事で、サービスインなどの予定はないので、ここで終わりにします。
# 株価を決定する要素は様々あるので、単純なNNでは予測するのはかなり難しいだろうと思っています。
