#!/usr/bin/env python
# coding: utf-8

# ## tensorflow tutorials RNN を使ったテキスト分類
# 
# tensorflowが2.0になってチュートリアルも新しくなりました。勉強がてら、すべてのチュートリアルを自分の環境で行ってみたいと思います。コードはほぼチュートリアルのコピーです。その中で気づいた部分や、注意すべき部分がこの記事の付加価値です。
# 
# - https://www.tensorflow.org/tutorials/text/text_classification_rnn?hl=ja

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import os, time

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


# ### シェイクスピアデータセットのダウンロード

# In[4]:


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# ### データの読み込み

# In[5]:


# 読み込んだのち、Python 2 との互換性のためにデコード
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# テキストの長さは含まれる文字数
print ('Length of text: {} characters'.format(len(text)))


# In[6]:


# テキストの最初の 250文字を参照
print(text[:250])


# In[7]:


# ファイル中のユニークな文字の数
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


# ## テキスト処理
# それぞれの文字を数値に変換します。文字を数値に変換するテーブル、数値を文字に変換するテーブルを作成します。

# In[8]:


# それぞれの文字からインデックスへの対応表を作成
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


# In[9]:


print('{')
for char,_ in zip(char2idx, range(20)):
  print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[10]:


# テキストの最初の 13 文字がどのように整数に変換されるかを見てみる
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# In[11]:


# ひとつの入力としたいシーケンスの文字数としての最大の長さ
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# 訓練用サンプルとターゲットを作る
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])


# In[ ]:





# In[12]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))


# In[13]:


def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)


# In[14]:


for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# In[15]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
  print("Step {:4d}".format(i))
  print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
  print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# ## 訓練用バッチの作成

# In[16]:


# バッチサイズ
BATCH_SIZE = 64

# データセットをシャッフルするためのバッファサイズ
# （TF data は可能性として無限長のシーケンスでも使えるように設計されています。
# このため、シーケンス全体をメモリ内でシャッフルしようとはしません。
# その代わりに、要素をシャッフルするためのバッファを保持しています）
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# ## モデルの構築

# In[17]:


# 文字数で表されるボキャブラリーの長さ
vocab_size = len(vocab)

# 埋め込みベクトルの次元
embedding_dim = 256

# RNN ユニットの数
rnn_units = 1024


# In[18]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


# In[19]:


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# ## モデルを試す

# In[ ]:


for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# In[ ]:


model.summary()


# In[ ]:





# In[ ]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# In[ ]:


sampled_indices


# In[ ]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# In[ ]:





# ## モデルの訓練

# In[ ]:


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# In[ ]:


model.compile(optimizer='adam', loss=loss)


# ## チェックポイントの構成

# In[ ]:


# チェックポイントが保存されるディレクトリ
checkpoint_dir = './training_checkpoints'
# チェックポイントファイルの名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# ## 訓練の実行

# In[ ]:


EPOCHS=10


# In[ ]:


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# In[ ]:





# ## テキストの生成

# In[ ]:


tf.train.latest_checkpoint(checkpoint_dir)


# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


model.summary()


# In[ ]:





# ## 予測ループ

# In[ ]:


def generate_text(model, start_string):
  # 評価ステップ（学習済みモデルを使ったテキスト生成）

  # 生成する文字数
  num_generate = 1000

  # 開始文字列を数値に変換（ベクトル化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 結果を保存する空文字列
  text_generated = []

  # 低い temperature　は、より予測しやすいテキストをもたらし
  # 高い temperature は、より意外なテキストをもたらす
  # 実験により最適な設定を見つけること
  temperature = 1.0

  # ここではバッチサイズ　== 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # バッチの次元を削除
      predictions = tf.squeeze(predictions, 0)

      # カテゴリー分布をつかってモデルから返された文字を予測 
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 過去の隠れ状態とともに予測された文字をモデルへのつぎの入力として渡す
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# In[ ]:


print(generate_text(model, start_string=u"ROMEO: "))


# In[ ]:





# ## 上級編： 訓練のカスタマイズ

# In[ ]:


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# In[ ]:


optimizer = tf.keras.optimizers.Adam()


# In[ ]:


@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss


# In[ ]:


# 訓練ステップ
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # 各エポックの最初に、隠れ状態を初期化する
  # 最初は隠れ状態は None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # 5エポックごとにモデル（のチェックポイント）を保存する
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))


# In[ ]:




