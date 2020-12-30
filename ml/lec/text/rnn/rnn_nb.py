#!/usr/bin/env python
# coding: utf-8

# ## kerasとRNNの基礎
# 
# 復習を兼ねてkerasを用いて再帰型ニューラルネットワーク（Recurrent Neural Network：以下、RNN）の実装を行ってみようと思います。何でもいいと思いますが、時系列データとして、減衰振動曲線を用意して、それをRNNを用いて学習させてみようと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/rnn/rnn_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/rnn/rnn_nb.ipynb)
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

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)


# ## 減衰振動曲線
# 
# サンプル用のデータとして、以下の式からサンプリングを行います。
# 
# $$
# y = \exp\left(-\frac{x}{\tau}\right)\cos(x) 
# $$
# 
# 波を打ちながら、次第に収束していく、自然現象ではよくあるモデルになります。

# In[4]:


x = np.linspace(0, 5 * np.pi, 200)
y = np.exp(-x / 5) * (np.cos(x))


# ### データの確認
# 
# $x$と$y$のデータの詳細を見てみます。

# In[5]:


print('shape : ', x.shape)
print('ndim : ', x.ndim)
print('data : ', x[:10])


# In[6]:


print('shape : ', y.shape)
print('ndim : ', y.ndim)
print('data : ', y[:10])


# グラフを確認してみます。

# In[7]:


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

# ニューラルネットの構築
model = Sequential()

model.add(SimpleRNN(NUM_MIDDLE, input_shape=(NUM_RNN, 1), return_sequences=True))
model.add(Dense(1, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="sgd")


# 投入するデータや、モデルの概要を確認します。

# In[9]:


print(r_y.shape)
print(r_x.shape)
print(model.summary())


# ## 学習
# 
# fitメソッドを利用して、学習を行います。
# fitメソッドの仕様は以下の通りになっています。[こちら](https://keras.io/ja/models/sequential/)を参照してください。
# 
# ```bash
# fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
# ```

# In[10]:


batch_size = 10
epochs = 500

# validation_split で最後の10％を検証用に利用します
history = model.fit(r_x, r_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)


# ## 損失関数の可視化
# 
# 学習によって誤差が減少していく様子を可視化してみます。

# In[11]:


loss = history.history['loss'] # 訓練データの損失関数
val_loss = history.history['val_loss'] #テストデータの損失関数

plt.plot(np.arange(len(loss)), loss, label='loss')
plt.plot(np.arange(len(val_loss)), val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.show()


# ## 結果の確認

# In[12]:


# 初期の入力値
res = r_y[0].reshape(-1)

for i in range(0, n):
  _y = model.predict(res[- NUM_RNN:].reshape(1, NUM_RNN, 1))
  res = np.append(res, _y[0][NUM_RNN - 1][0])
  
plt.plot(np.arange(len(y)), y, label=r"$\exp\left(-\frac{x}{\tau}\right) \cos x$")
plt.plot(np.arange(len(res)), res, label="RNN result")
plt.legend()
plt.grid()
plt.show()


# 単純なRNNだと少しずつずれが顕著になってきます。epochやモデルを改良すればもっと良い結果が出るかもしませんが、復習なのでここで一旦終わりとします。次はLSTMをやってみようと思います。
