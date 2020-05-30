
# tensorflow 2.0

tensorlowが1.x系から2.x系へバージョンアップされました。大きな変更点は、define and runからdefine by runに変更になったことだと思います。ここでは、自分がtensorflow 2.0を利用していく中で、注意した方が良いと思う点や、実務を効率的にこなすために覚えておこうと意識した点をまとめていきます。

## tf.data.Dataset
 
tf.data.Datasetはかなり便利です。教師あり学習の場合、ニューラルネットに投入するデータは通常、データとラベルがありますが、それを一喝してまとめてデータセットとして扱う際に力を発揮します。また、データのバッチ化やシャッフル機能、mapなどにも対応しており、tensorflowを利用するならば、必ず利用する機能かと思います。

### tf.data.from_tensors

まずはtensorflowの読み込みと、基本的なデータセットの作成からです。
データセットの作成は、

```python
tf.data.Dataset.from_tensors
```

を利用します。tensorflowのversionは以下の通りです。


```python

```


```python

```


```python

```
