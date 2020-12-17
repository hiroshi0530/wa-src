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





# ### aa
# 
# 青空文庫からすべての作品をダウンロード
# 
# gitがかなり重いので、最新の履歴だけを取得します。
# 
# ```bash
# git clone --depth 1 https://github.com/aozorabunko/aozorabunko.git
# ```

# In[4]:


get_ipython().system('ls -a')


# 実際のファイルはcardsにzip形式として保存されているようです。

# In[6]:


ls ./aozorabunko/cards/*. | wc -l


# zipファイルだけzipsに移動させます。
# 
# ```bash
# find ./aozorabunko/cards/ -name *.zip | xargs -I{} cp {} -t ./zips/
# ```

# In[9]:


get_ipython().system('ls ./zips/ | head -n 5')


# In[10]:


get_ipython().system('ls ./zips/ | wc -l')


# となり、
# 
# ```bash
# for i in `ls`; do [[ ${i##*.} == zip ]] && unzip -o $i -d ../texts/; done
# ```

# In[ ]:





# In[ ]:





# In[90]:


import re

with open('wagahaiwa_nekodearu.txt', mode='r') as f:
  all_sentence = f.read()

all_sentence = all_sentence.replace(" ", "").replace("　","").replace("\n","").replace("|","")
all_sentence = re.sub("《[^》]+》", "", all_sentence)

sentence_list = all_sentence.split("。")

sentence_list = [ s + "。" for s in sentence_list]

sentence_list[:10]


# ### janomeによる形態素解析

# In[91]:


from janome.tokenizer import Tokenizer

t = Tokenizer()

word_list = []
word_per_sentence_list = []
for sentence in sentence_list:
  word_list.extend(list(t.tokenize(sentence, wakati=True)))
  word_per_sentence_list.append(list(t.tokenize(sentence, wakati=True)))
  

print(word_list[0])
print(word_per_sentence_list[0])


# ### 単語のカウント
# 
# 単語のカウントを行い、出現頻度の高いベスト10を抽出してみます。

# In[73]:


import collections

count = collections.Counter(word_list)
count.most_common()[:10]


# ### gensimに含まれるword2vecを用いた学習
# 
# word2vecを用いて、word_listの分散表現を取得します。

# In[74]:


from gensim.models import word2vec

model = word2vec.Word2Vec(word_list, size=100, min_count=5, window=5, iter=20, sg=0)


# #### 分散行列

# In[75]:


model.wv.vectors


# #### 分散行列の形状確認

# In[76]:


model.wv.vectors.shape


# In[77]:


model.wv.index2word[:10]


# In[78]:


model.wv.vectors[0]


# In[79]:


model.wv.__getitem__("の")


# ### cos類似度による単語抽出

# In[80]:


model.wv.most_similar("男")


# ### 単語ベクトルによる演算
# 
# 足し算するにはpositiveメソッドを引き算にはnegativeメソッドを利用します。
# 
# まず、猫＋人を計算します。

# In[81]:


model.wv.most_similar(positive=["猫", "人"])


# 次に猫＋人ー男を計算します。

# In[82]:


model.wv.most_similar(positive=["猫", "人"], negative=["男"])


# ## doc2vec
# 
# 文章毎にタグ付けされたTaggedDocumentを作成します。

# In[92]:


from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

tagged_doc_list = []

for i, sentence in enumerate(word_per_sentence_list):
  tagged_doc_list.append(TaggedDocument(sentence, [i]))

print(tagged_doc_list[0])


# In[93]:


model = Doc2Vec(documents=tagged_doc_list, vector_size=100, min_count=5, window=5, epochs=20, dm=0)


# In[94]:


word_per_sentence_list[0]


# In[95]:


model.docvecs[0]


# most_similarで類似度が高い文章のIDと類似度を取得することが出来ます。

# In[96]:


model.docvecs.most_similar(0)


# In[98]:


for p in model.docvecs.most_similar(0):
  print(word_per_sentence_list[p[0]])


# 感覚的ですが、似たような文章が抽出されています。

# In[ ]:





# In[ ]:




