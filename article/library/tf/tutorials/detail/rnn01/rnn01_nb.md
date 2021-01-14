
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


## ヘルパー関数の作成
結果を描画するための関数


```python
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
```


```python
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']
```


```python
encoder = info.features['text'].encoder
```


```python
'Vocabulary size: {}'.format(encoder.vocab_size)
```




    'Vocabulary size: 8185'




```python
sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))
```

    Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]
    The original string: "Hello TensorFlow."



```python
assert original_string == sample_string
```


```python

```


```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64
```


```python
train_dataset = (train_examples
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(BATCH_SIZE, padded_shapes=([None],[])))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE,  padded_shapes=([None],[])))
```


```python
train_dataset = (train_examples
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(BATCH_SIZE))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE))
```


```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```


```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```


```python
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
```

        387/Unknown - 251s 640ms/step - loss: 0.6931 - accuracy: 0.5021

    ERROR:root:Internal Python error in the inspect module.
    Below is the traceback from this internal error.
    


    Traceback (most recent call last):
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3296, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-15-0556d3299cc6>", line 1, in <module>
        test_loss, test_acc = model.evaluate(test_dataset)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py", line 1389, in evaluate
        tmp_logs = self.test_function(iterator)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 828, in __call__
        result = self._call(*args, **kwds)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 855, in _call
        return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 2943, in __call__
        filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 1919, in _call_flat
        ctx, args, cancellation_manager=cancellation_manager))
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 560, in call
        ctx=ctx)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/execute.py", line 60, in quick_execute
        inputs, attrs, num_outputs)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 2033, in showtraceback
        stb = value._render_traceback_()
    AttributeError: 'KeyboardInterrupt' object has no attribute '_render_traceback_'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/IPython/core/ultratb.py", line 1095, in get_records
        return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/IPython/core/ultratb.py", line 313, in wrapped
        return f(*args, **kwargs)
      File "/Users/hiroshi/anaconda3/lib/python3.7/site-packages/IPython/core/ultratb.py", line 347, in _fixed_getinnerframes
        records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))
      File "/Users/hiroshi/anaconda3/lib/python3.7/inspect.py", line 1502, in getinnerframes
        frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)
      File "/Users/hiroshi/anaconda3/lib/python3.7/inspect.py", line 1460, in getframeinfo
        filename = getsourcefile(frame) or getfile(frame)
      File "/Users/hiroshi/anaconda3/lib/python3.7/inspect.py", line 696, in getsourcefile
        if getattr(getmodule(object, filename), '__loader__', None) is not None:
      File "/Users/hiroshi/anaconda3/lib/python3.7/inspect.py", line 739, in getmodule
        f = getabsfile(module)
      File "/Users/hiroshi/anaconda3/lib/python3.7/inspect.py", line 709, in getabsfile
        return os.path.normcase(os.path.abspath(_filename))
    KeyboardInterrupt



    ---------------------------------------------------------------------------



```python

```


```python
def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec
```


```python
def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)
```


```python

```


```python
# パディングなしのサンプルテキストの推論

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)
```


```python

```


```python
# パディングありのサンプルテキストの推論

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)
```


```python
plot_graphs(history, 'accuracy')
```


```python
plot_graphs(history, 'loss')
```


```python

```

## 2つ以上の LSTM レイヤー


```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
```


```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```


```python
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)
```


```python
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
```


```python

```


```python
# パディングなしのサンプルテキストの推論

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)
```


```python
# パディングありのサンプルテキストの推論

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)
```


```python
plot_graphs(history, 'accuracy')
```


```python
plot_graphs(history, 'loss')
```


```python

```


```python

```


```python

```


```python

```


```python

```
