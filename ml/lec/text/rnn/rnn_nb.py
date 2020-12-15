#!/usr/bin/env python
# coding: utf-8

# ## word2vec と doc2vec
# 
# 単語や文章を分散表現（意味が似たような単語や文章を似たようなベクトルとして表現）を取得します。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/scipy/template/template_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/scipy/template/template_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)


# In[ ]:





# In[4]:


x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

plt.plot(x, y)
plt.grid()
plt.show()


# In[ ]:





# In[5]:


n_rnn = 10
n_sample = len(x) - n_rnn
x1 = np.zeros((n_sample, n_rnn))
t1 = np.zeros((n_sample, n_rnn))

for i in range(0, n_sample):
  x1[i] = y[i : i + n_rnn]
  t1[i] = y[i + 1 : i + n_rnn + 1]

x1 = x1.reshape(n_sample, n_rnn, 1)
t1 = t1.reshape(n_sample, n_rnn, 1)

print(x1.shape)
print(t1.shape)


# ### RNNの構築
# 
# Kerasの中で最もシンプルなSimpleRNNを利用する

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

batch_size = 8
n_in = 1
n_mid = 20
n_out = 1


# In[7]:


model = Sequential()

model.add(SimpleRNN(n_mid, input_shape=(n_rnn, n_in), return_sequences=True))
model.add(Dense(n_out, activation='linear'))
model.compile(loss="mean_squared_error", optimizer='sgd')

model.summary()


# In[8]:


history = model.fit(x1, t1, epochs=200, batch_size=batch_size, validation_split=0.1)


# In[9]:


loss = history.history['loss']
val_loss = history.history['val_loss']


# In[10]:


plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(val_loss)), val_loss)
plt.grid()
plt.show()


# In[11]:


predicted = x1[0].reshape(-1)

for i in range(0, n_sample):
  y1 = model.predict(predicted[ -n_rnn:].reshape(1, n_rnn, 1))
  predicted = np.append(predicted, y1[0][n_rnn - 1][0])
  
plt.plot(np.arange(len(y)), y, label='training_data')
plt.plot(np.arange(len(predicted)), predicted, label='Predicted')

plt.legend()
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


len(x)


# In[13]:


len(np.linspace(-10,10, 100))


# In[14]:


x


# In[15]:


y


# In[2]:


from scipy.stats import norm

get_ipython().run_line_magic('pinfo', 'norm')


# In[6]:


a = [i for i in range(100)]


# In[ ]:




