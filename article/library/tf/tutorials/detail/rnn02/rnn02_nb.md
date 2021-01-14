
## tensorflow tutorials RNN を使ったテキスト分類

tensorflowが2.0になってチュートリアルも新しくなりました。勉強がてら、すべてのチュートリアルを自分の環境で行ってみたいと思います。コードはほぼチュートリアルのコピーです。その中で気づいた部分や、注意すべき部分がこの記事の付加価値です。

- https://www.tensorflow.org/tutorials/text/text_classification_rnn?hl=ja


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G6020



```python
!python -V
```

    Python 3.7.3



```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

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
```

    tf version     :  2.4.0
    keras version  :  2.4.0
    numpy version  :  1.19.4
    pandas version :  1.0.3
    matlib version :  3.0.3


### シェイクスピアデータセットのダウンロード


```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
    1122304/1115394 [==============================] - 1s 1us/step


### データの読み込み


```python
# 読み込んだのち、Python 2 との互換性のためにデコード
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# テキストの長さは含まれる文字数
print ('Length of text: {} characters'.format(len(text)))
```

    Length of text: 1115394 characters



```python
# テキストの最初の 250文字を参照
print(text[:250])
```

    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    



```python
# ファイル中のユニークな文字の数
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
```

    65 unique characters


## テキスト処理
それぞれの文字を数値に変換します。文字を数値に変換するテーブル、数値を文字に変換するテーブルを作成します。


```python
# それぞれの文字からインデックスへの対応表を作成
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
```


```python
print('{')
for char,_ in zip(char2idx, range(20)):
  print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
```

    {
      '\n':   0,
      ' ' :   1,
      '!' :   2,
      '$' :   3,
      '&' :   4,
      "'" :   5,
      ',' :   6,
      '-' :   7,
      '.' :   8,
      '3' :   9,
      ':' :  10,
      ';' :  11,
      '?' :  12,
      'A' :  13,
      'B' :  14,
      'C' :  15,
      'D' :  16,
      'E' :  17,
      'F' :  18,
      'G' :  19,
      ...
    }



```python
# テキストの最初の 13 文字がどのように整数に変換されるかを見てみる
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
```

    'First Citizen' ---- characters mapped to int ---- > [18 47 56 57 58  1 15 47 58 47 64 43 52]



```python
# ひとつの入力としたいシーケンスの文字数としての最大の長さ
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# 訓練用サンプルとターゲットを作る
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
```

    F
    i
    r
    s
    t



```python

```


```python
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))
```

    'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '
    'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'
    "now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"
    "ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"
    'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'



```python
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)
```


```python
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
```

    Input data:  'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
    Target data: 'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '



```python
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
  print("Step {:4d}".format(i))
  print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
  print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
```

    Step    0
      input: 18 ('F')
      expected output: 47 ('i')
    Step    1
      input: 47 ('i')
      expected output: 56 ('r')
    Step    2
      input: 56 ('r')
      expected output: 57 ('s')
    Step    3
      input: 57 ('s')
      expected output: 58 ('t')
    Step    4
      input: 58 ('t')
      expected output: 1 (' ')


## 訓練用バッチの作成


```python
# バッチサイズ
BATCH_SIZE = 64

# データセットをシャッフルするためのバッファサイズ
# （TF data は可能性として無限長のシーケンスでも使えるように設計されています。
# このため、シーケンス全体をメモリ内でシャッフルしようとはしません。
# その代わりに、要素をシャッフルするためのバッファを保持しています）
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
```




    <BatchDataset shapes: ((64, 100), (64, 100)), types: (tf.int64, tf.int64)>



## モデルの構築


```python
# 文字数で表されるボキャブラリーの長さ
vocab_size = len(vocab)

# 埋め込みベクトルの次元
embedding_dim = 256

# RNN ユニットの数
rnn_units = 1024
```


```python
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
```


```python
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
```

## モデルを試す


```python
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
```


```python
model.summary()
```


```python

```


```python
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
```


```python
sampled_indices
```


```python
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
```


```python

```

## モデルの訓練


```python
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
```


```python
model.compile(optimizer='adam', loss=loss)
```

## チェックポイントの構成


```python
# チェックポイントが保存されるディレクトリ
checkpoint_dir = './training_checkpoints'
# チェックポイントファイルの名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
```

## 訓練の実行


```python
EPOCHS=10
```


```python
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
```


```python

```

## テキストの生成


```python
tf.train.latest_checkpoint(checkpoint_dir)
```


```python
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
```


```python
model.summary()
```


```python

```

## 予測ループ


```python
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
```


```python
print(generate_text(model, start_string=u"ROMEO: "))
```


```python

```

## 上級編： 訓練のカスタマイズ


```python
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
```


```python
optimizer = tf.keras.optimizers.Adam()
```


```python
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
```


```python
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
```


```python

```
