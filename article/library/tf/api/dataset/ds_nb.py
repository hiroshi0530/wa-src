
# coding: utf-8

# # tensorflow 2.0
# 
# tensorlowが1.x系から2.x系へバージョンアップされました。大きな変更点は、define and runからdefine by runに変更になったことだと思います。ここでは、自分がtensorflow 2.0を利用していく中で、注意した方が良いと思う点や、実務を効率的にこなすために覚えておこうと意識した点をまとめていきます。
# 
# ## tf.data.Dataset
#  
# tf.data.Datasetはかなり便利です。教師あり学習の場合、ニューラルネットに投入するデータは通常、データとラベルがありますが、それを一喝してまとめてデータセットとして扱う際に力を発揮します。また、データのバッチ化やシャッフル機能、mapなどにも対応しており、tensorflowを利用するならば、必ず利用する機能かと思います。
# 
# ### tf.data.from_tensors
# 
# まずはtensorflowの読み込みと、基本的なデータセットの作成からです。
# データセットの作成は、
# 
# ```python
# tf.data.Dataset.from_tensors
# ```
# 
# を利用します。tensorflowのversionは以下の通りです。

# In[1]:


import tensorflow as tf

tf.__version__


# tensorflowの名前の元である、0階、1階、2階のtensorのDatasetは以下の様に作れます。引数はList型やtf.constなどでも大丈夫です。暗黙的にTensor型に変換してくれます。

# In[23]:


dataset0 = tf.data.Dataset.from_tensors(1)
print(dataset0)
dataset1 = tf.data.Dataset.from_tensors([1,2])
print(dataset1)
dataset2 = tf.data.Dataset.from_tensors([[1,2],[3,4]])
print(dataset2)


# shapesが(),(2,),(2,2)となっていて、それぞれの次元のtensorが出来ています。また、Datasetはジェネレータのため、値を参照するにはイテレータの様に呼び出す必要があります。また、Dataset型から取り出されたEagerTensor型はnumpy()メソッドを実装しており、明示的にnumpy形式に変換することが出来ます。

# In[43]:


_dataset0 = next(iter(dataset0))
_dataset1 = next(iter(dataset1))
_dataset2 = next(iter(dataset2))


# In[25]:


print('_dataset0 : \n{}'.format(_dataset0.numpy()))
print('_dataset1 : \n{}'.format(_dataset1.numpy()))
print('_dataset2 : \n{}'.format(_dataset2.numpy()))


# また、ジェネレータなのでforで取り出すことも可能です。

# In[28]:


for i in dataset0:
  print(i.numpy())
for i in dataset1:
  print(i.numpy())
for i in dataset2:
  print(i.numpy())


# ### tf.data.from_tensor_slices
# おそらくDatasetを作るときは、元々何らかの形でリスト型になっている物をDataset型に変換することが多いと思いますので、実際は`tf.data.from_tensors`よりこちらの方をよく使うと思います。
# まずは一次元リストを入れてみます。

# In[57]:


dataset_20 = tf.data.Dataset.from_tensor_slices([i for i in range(5)])


# In[58]:


for i in dataset_20:
  print(i)


# In[59]:


dataset_20


# となります。tensor_slicesという名前から予想されるとおり、リストから順番にスライスしてDatasetを作っているイメージですね。二次元のリストを入れると以下の通りです。

# In[60]:


dataset_21 = tf.data.Dataset.from_tensor_slices([[j for j in range(3)] for i in range(4)])


# In[61]:


for i in dataset_21:
  print(i)


# こちらも想像通り、一次元のリストのDataset型となっています。

# In[6]:


import seaborn as sns
import pandas as pd

df = pd.DataFrame([[1,2,3], [4,5,6]])

df.plot()


# ## tf.keras.preprocessing 
# 
# 自然言語処理処理では、配列を同じ長さに揃える必要があります。 
