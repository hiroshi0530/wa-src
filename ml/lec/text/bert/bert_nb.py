#!/usr/bin/env python
# coding: utf-8

# ## BERTの基礎
# 
# 今ではGoogleの検索アルゴリズムにも利用されているBertについて、手を動かして触ってみます。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/bert/bert_nb.ipynb)
# 
# ### google colaboratory
# <a href="https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/bert/bert_nb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

import tensorflow as tf
from tensorflow import keras

import torch

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)
print('torch version : ', torch.__version__)


# ## 必要なライブラリのインストール
# 
# 今回入門用にやってみる pytorch-transformersをインストールします。
# 
# ```bash
# pip install pytorch-transformers
# ```
# 
# BertModelの事前学習済みのモデルの詳細を確認します。

# In[5]:


from pytorch_transformers import BertModel

bert_model = BertModel.from_pretrained('bert-base-uncased')
print(bert_model)


# ## 設定の確認

# In[6]:


from pytorch_transformers import BertConfig

config = BertConfig.from_pretrained("bert-base-uncased")
print(config) 


# In[8]:


from transformers import BertForSequenceClassification

sc_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True)
print(sc_model.state_dict().keys())


# ## 最適化アルゴリズム
# 今回は、最適化アルゴリズムに`AdamW`を採用します。  
# `AdamW`は、オリジナルのAdamの重みの減衰に関する式を変更したものです。  
# https://arxiv.org/abs/1711.05101

# In[9]:


from transformers import AdamW

optimizer = AdamW(sc_model.parameters(), lr=1e-5)


# ## Tokenizerの設定
# `BertTokenizer`により文章を単語に分割し、idに変換します。  
# `BertForSequenceClassification`のモデルの訓練時には入力の他にAttention maskを渡す必要があるのですが、`BertTokenizer`によりこちらも得ることができます。

# In[10]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentences = ["I love baseball.", "I hate baseball."]
tokenized = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
print(tokenized)

x = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]


# ## ファインチューニング
# 事前に学習済みのモデルに対して、追加で訓練を行います。

# In[11]:


import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

sc_model.train()
t = torch.tensor([1,0])  # 文章の分類

weight_record = []  # 重みを記録

for i in range(100):
    y = sc_model(x, attention_mask=attention_mask)
    loss = F.cross_entropy(y.logits, t)
    loss.backward()
    optimizer.step()

    weight = sc_model.state_dict()["bert.encoder.layer.11.output.dense.weight"][0][0].item()
    weight_record.append(weight)

plt.plot(range(len(weight_record)), weight_record)
plt.show()


# 追加の訓練により、重みが調整されていく様子が確認できます。

# In[12]:


1 + 1


# In[13]:


for i in range(100):
  print(i)

