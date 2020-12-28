#!/usr/bin/env python
# coding: utf-8

# ## word2vec と doc2vec
# 
# 単語や文章を分散表現（意味が似たような単語や文章を似たようなベクトルとして表現）を取得します。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/lec/text/w2v/w2v_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/lec/text/w2v/w2v_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。tensorflowとkerasuのversionも確認します。

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


# ### テキストデータの取得
# 
# 著作権の問題がない青空文庫からすべての作品をダウンロードしてきます。gitがかなり重いので、最新の履歴だけを取得します。
# 
# ```bash
# git clone --depth 1 https://github.com/aozorabunko/aozorabunko.git
# ```

# 実際のファイルはcardsにzip形式として保存されているようです。ディレクトリの個数を確認してみます。

# In[4]:


get_ipython().system('ls ./aozorabunko/cards/* | wc -l')


# zipファイルだけzipsに移動させます。
# 
# ```bash
# find ./aozorabunko/cards/ -name *.zip | xargs -I{} cp {} -t ./zips/
# ```

# In[5]:


get_ipython().system('ls ./zips/ | head -n 5')


# In[6]:


get_ipython().system('ls ./zips/ | wc -l')


# となり、16444個のzipファイルがある事が分かります。こちらをすべて解凍し、ディレクトリを移動させます。
# 
# ```bash
# for i in `ls`; do [[ ${i##*.} == zip ]] && unzip -o $i -d ../texts/; done
# ```
# 
# これで、textｓというディレクトリにすべての作品のテキストファイルがインストールされました。

# In[7]:


get_ipython().system('ls ./texts/ | grep miyazawa')


# In[8]:


get_ipython().system('ls ./texts/ | grep ginga_tetsudo')


# となり、宮沢賢治関連の作品も含まれていることが分かります。銀河鉄道の夜もあります。
# 
# ## 銀河鉄道の夜を使ったword2vec
# 
# 今回はすべてのテキストファイルを対象にするには時間がかかるので、同じ岩手県出身の、高校の先輩でもある宮沢賢治の作品を例に取りword2vecを試してみます。
# しかし、ファイルの中身を見てみると、

# In[9]:


get_ipython().system('head ./texts/ginga_tetsudono_yoru.txt')


# In[10]:


get_ipython().system('nkf --guess ./texts/ginga_tetsudono_yoru.txt')


# となりshift_jisで保存されていることが分かります。

# In[11]:


get_ipython().system('nkf -w ./texts/ginga_tetsudono_yoru.txt > ginga.txt')


# と、ディレクトリを変更し、ファイル名も変更します。

# In[12]:


get_ipython().system('cat ginga.txt | head -n 25')


# In[13]:


get_ipython().system('cat ginga.txt | tail -n 25')


# となり、ファイルの先頭と、末尾に参考情報が載っているほかは、ちゃんとテキストとしてデータが取れている模様です。
# 先ず、この辺の前処理を行います。

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


import re

with open('ginga.txt', mode='r') as f:
  all_sentence = f.read()


# 全角、半角の空白、改行コード、縦線(|)をすべて削除します。正規表現を利用します。

# In[15]:


all_sentence = all_sentence.replace(" ", "").replace("　","").replace("\n","").replace("|","")


# 《》で囲まれたルビの部分を削除します。正規表現を利用します。

# In[16]:


all_sentence = re.sub("《[^》]+》", "", all_sentence)


# ----------の部分で分割を行い、2番目の要素を取得します。

# In[17]:


all_sentence = re.split("\-{8,}", all_sentence)[2]


# 。で分割し、文ごとにリストに格納します。

# In[18]:


sentence_list = all_sentence.split("。")
sentence_list = [ s + "。" for s in sentence_list]
sentence_list[:5]


# 最初の文は不要なので削除します。

# In[19]:


sentence_list = sentence_list[1:]
sentence_list[:5]


# となり、不要な部分を削除し、一文ごとにリストに格納できました。前処理は終了です。

# ## janomeによる形態素解析
# 
# janomeは日本語の文章を形態素ごとに分解する事が出来るツールです。同じようなツールとして、MecabやGinzaなどがあります。一長一短があると思いますが、ここではjanomeを利用します。

# In[65]:


from janome.tokenizer import Tokenizer

t = Tokenizer()

word_list = []
# word_per_sentence_list = []
# for sentence in sentence_list:
#   word_list.extend(list(t.tokenize(sentence, wakati=True)))
#   word_per_sentence_list.append(list(t.tokenize(sentence, wakati=True)))

# テキストを引数として、形態素解析の結果、名詞・動詞・形容詞(原形)のみを配列で抽出する関数を定義 
def extract_words(text):
  tokens = t.tokenize(text)
  return [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in['名詞', '動詞']]
    

#  関数テスト
# ret = extract_words('三四郎は京都でちょっと用があって降りたついでに。')
# for word in ret:
#    print(word)

# 全体のテキストを句点('。')で区切った配列にする。 
# sentences = text.split('。')
# それぞれの文章を単語リストに変換(処理に数分かかります)
# word_list = [extract_words(sentence) for sentence in sentence_list] 
for sentence in sentence_list:
  word_list.extend(extract_words(sentence))
print(word_list[:10])
# print(word_per_sentence_list[:5])


# ## 単語のカウント
# 
# 単語のカウントを行い、出現頻度の高いベスト10を抽出してみます。名詞のみに限定した方が良かったかもしれません。

# In[56]:


import collections

count = collections.Counter(word_list)
count.most_common()[:10]
dict(count.most_common())['銀河']
dict(count.most_common())['ジョバンニ']


# ## gensimに含まれるword2vecを用いた学習
# 
# word2vecを用いて、word_listの分散表現を取得します。使い方はいくらでも検索できますので、ここでは割愛します。単語のリストを渡せば、ほぼ自動的に分散表現を作ってくれます。

# In[84]:


from gensim.models import word2vec

model = word2vec.Word2Vec(word_list, size=100, min_count=5, window=5, iter=1000, sg=0)


# ### 分散行列

# In[75]:


model.wv.vectors


# ### 分散行列の形状確認
# 
# 443個の単語について、100次元のベクトルが生成されました。

# In[76]:


model.wv.vectors.shape


# 全単語数は、

# In[77]:


len(set(word_list))


# ですが、word2vecのmin_countを5にしているので、その文単語数が少なくなっています。

# In[83]:


model.wv.index2word[:10]
print(model.__dict__['wv']['銀河'])


# In[79]:


model.wv.vectors[0]


# In[82]:


model.wv.__getitem__("銀河")


# ### cos類似度による単語抽出
# 
# ベクトルの内積を計算することにより、指定した単語に類似した単語をその$\cos$の値と一緒に抽出する事ができます。

# In[81]:


print(model.wv.most_similar("銀河"))
print(model.wv.most_similar("本"))
print(model.wv.most_similar("ジョバンニ"))


# ### 単語ベクトルによる演算
# 
# 足し算するにはpositiveメソッドを引き算にはnegativeメソッドを利用します。
# 
# まず、銀河＋男を計算します。

# In[ ]:


model.wv.most_similar(positive=["銀河", "ジョバンニ"])


# 次に銀河＋ジョバンニー家を計算します。

# In[ ]:


model.wv.most_similar(positive=["銀河", "ジョバンニ"], negative=["家"])


# ## doc2vec
# 
# 文章毎にタグ付けされたTaggedDocumentを作成します。

# In[ ]:


from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

tagged_doc_list = []

for i, sentence in enumerate(word_per_sentence_list):
  tagged_doc_list.append(TaggedDocument(sentence, [i]))

print(tagged_doc_list[0])


# In[ ]:


model = Doc2Vec(documents=tagged_doc_list, vector_size=100, min_count=5, window=5, epochs=20, dm=0)


# In[ ]:


word_per_sentence_list[0]


# In[ ]:


model.docvecs[0]


# most_similarで類似度が高い文章のIDと類似度を取得することが出来ます。

# In[ ]:


model.docvecs.most_similar(0)


# In[ ]:


for p in model.docvecs.most_similar(0):
  print(word_per_sentence_list[p[0]])


# 感覚的ですが、似たような文章が抽出されています。

# In[ ]:





# In[ ]:




