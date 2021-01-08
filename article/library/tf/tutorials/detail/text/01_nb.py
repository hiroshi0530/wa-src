#!/usr/bin/env python
# coding: utf-8

# ## tensorflow tutorials メモ
# 
# tensorflowが2.0になってチュートリアルも新しくなりました。勉強がてら、すべてのチュートリアルを自分の環境で行ってみたいと思います。コードはほぼチュートリアルのコピーです。その中で気づいた部分や、注意すべき部分がこの記事の付加価値です。
# 
# 今回は単語の埋め込み（分散表現の学習）です。リンクこちらです。
# 
# - https://www.tensorflow.org/tutorials/text/word_embeddings?hl=ja

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


print('tf version     : ', tf.__version__)
print('keras version  : ', keras.__version__)
print('numpy version  : ',np.__version__)
print('pandas version : ',pd.__version__)
print('matlib version : ',matplotlib.__version__)


# ## 単語の埋め込み(Word embeddings)

# Embddingレイヤーの作成。埋め込みの重みはランダムに初期化。

# In[8]:


embedding_layer = layers.Embedding(1000, 5)


# In[26]:


result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()


# テキストやシーケンスの問題では、入力として`(samples, sequence_lenght)`の二次元整数テンソルとなる。シーケンスのバッチを入力すると、Embeddingレイヤーは`shape=(samples, sequence_length, embedding_dimensionality)`の三次元のFloaｔテンソルを返す。

# In[10]:


result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
result.shape


# ## 埋め込みを最初から学習する
# 
# IMDBの映画レビューから感情分析器を作成する。tfdsからデータをロードする。

# In[31]:


(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)


# エンコーダー(tfds.features.text.SubwordTextEncoder)を取得し、どのような単語が含まれているか見る。

# In[32]:


encoder = info.features['text'].encoder


# In[33]:


print(encoder.subwords[:20])


# 映画のレビューは長さが異なっている。padded_batchを用いてレビューの長さを標準化する。tf2.2からpadded_shapesは必須ではなくなったようです。デフォルトで自動で最も長い文章に合わせてパディングされるようです。

# In[13]:


train_data


# パディングして、データをバッチ化します。

# In[14]:


train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=([None],[]))
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=([None],[]))


# 2.2移行はこれで良さそうです。

# In[15]:


train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)


# In[16]:


train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()


# 大きさを見てみます。

# In[37]:


train_batch.numpy().shape


# ## 単純なモデルの構築
# 
# - 整数エンコードされた語彙を受け取り、埋め込みベクトルを返す。その結果次元は `(batch, sequence, embedding)`となる。
# - GlobalAveragePooling1Dはそれぞれのサンプルについて、固定長の出力ベクトルを返す
# - 16層の全結合隠れ層

# In[17]:


embedding_dim = 16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()


# ## モデルのコンパイル

# In[39]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20, verbose=0)


# In[24]:


history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.grid()
plt.show()


# ## 学習した埋め込みの取得
# 
# 訓練によって、学習され行列を取得します。shapeが`(vocab_size, embedding_dim)`になります。

# In[20]:


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


# ## ファイルへ出力

# In[21]:


import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
  vec = weights[num+1] # 0 はパディングのためスキップ
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()


# In[22]:


try:
  from google.colab import files
except ImportError:
   pass
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

