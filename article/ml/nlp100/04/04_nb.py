
# coding: utf-8

# ## [第4章 形態素解析](https://nlp100.github.io/ja/ch04.html)
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/ml/nlp100/02/02_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().system('bash --version')


# ### 共通部分

# In[4]:


import MeCab
import re
import collections

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib.pyplot as plt
import japanize_matplotlib


# ### データのダウンロード

# In[5]:


get_ipython().system('wget https://nlp100.github.io/data/neko.txt -O ./neko.txt')


# In[6]:


in_file_name = './neko.txt'
out_file_name = './neko.txt.mecab'


# In[7]:


mecab= MeCab.Tagger()

with open(in_file_name, mode='r') as f:
  n = mecab.parse(f.read())
  with open(out_file_name, mode='w') as f1:
    f1.write(n)


# ### 30問 形態素解析結果の読み込み
# 最終的に1分の形態素のリストのリストとなるようにします。

# In[8]:


get_ipython().system('head -n 5 $out_file_name')


# In[9]:


get_ipython().system('tail -n 5 $out_file_name')


# 最後の一行は除外します。

# In[10]:


sentence_list = []

with open(out_file_name, mode='r') as f:
  temp = []
  for s in f.readlines()[:-1]:
    s1 = s.split('\t')
    s2 = s1[1].split(',')
    _dic = {
      'surface': s1[0],
      'base': s2[6],
      'pos': s2[0],
      'pos1': s2[1],
    }
    
    temp.append(_dic)
    
    if s1[0] == '。':
      sentence_list.append(temp)
      temp = []


# ### 31問 動詞

# In[11]:


result = []
for sentence in sentence_list:
  for _dic in sentence:
    if _dic['pos'] == '動詞':
      result.append(_dic['surface'])

print(set(result))


# ### 32問 動詞の原形

# In[12]:


result = []
for sentence in sentence_list:
  for _dic in sentence:
    if _dic['pos'] == '動詞':
      result.append(_dic['base'])

print(set(result))


# ### 33問 「AのB」

# In[13]:


result = []
for sentence in sentence_list:
  if len(sentence) >= 3:
    for i, _dic in enumerate(sentence[1:-1]):
      if _dic['base'] == 'の' and          sentence[i-1+1]['pos'] == '名詞' and          sentence[i+1+1]['pos'] == '名詞' :
        result.append(sentence[i-1+1]['base'])
        result.append(sentence[i+1+1]['base'])

print(set(result))


# ### 34問 名詞の連接

# In[14]:


result = []
temp_list = []
for sentence in sentence_list:
  temp_01 = list(map(lambda x: x['surface'] if x['pos'] == '名詞' else "", sentence))
  temp_02 = list(map(lambda x: 'a' if x['pos'] == '名詞' else 'b', sentence))
  
  a = ''.join(temp_02)
  temp_03 = list(re.sub('(aa+)','c' * len('\\1'),a))
  
  b = []
  t = '' 
  for i,j in zip(temp_01, temp_03):
    if j == 'c':
      t += i
    else:
      b.append(t)
      t = '' 
  
  result.extend(list(filter(lambda x: x != '',b)))
 
print(result)


# ### 35問 単語の出現頻度
# 単語の品詞は何なのかわかりませんが、ここは、名詞、動詞、形容詞に限定します。多い10個の単語を表示しています。

# In[15]:


result_list = []
for sentence in sentence_list:
  for w in sentence:
    if w['pos'] in ['名詞','動詞','形容詞']:
      result_list.append(w['surface'])
    
for w in sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)[0:10]:
  print(w)


# ### 36問 頻度上位10語

# In[16]:


result_list = []
for sentence in sentence_list:
  for w in sentence:
    if w['pos'] in ['名詞','動詞','形容詞']:
      result_list.append(w['surface'])
    
x = list(map(lambda x:x[0],sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)[0:10]))
y = list(map(lambda x:x[1],sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)[0:10]))

plt.bar(x,y)


# ### 37問 「猫」と共起頻度の高い上位10語
# 猫を含む文章から、そこに含まれる単語を抽出します。ここでは名詞と限定しています。

# In[17]:


result_list = []
for sentence in sentence_list:
  if '猫' in list(map(lambda x:x['surface'], sentence)):
    for w in sentence:
      if w['pos'] in ['名詞'] and w['surface'] != '猫':
        result_list.append(w['surface'])
    
x = list(map(lambda x:x[0],sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)[0:10]))
y = list(map(lambda x:x[1],sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)[0:10]))

plt.bar(x,y)


# ### 38問 ヒストグラム
# 出現頻度の高い上位20までを表示しています。

# In[18]:


result_list = []
for sentence in sentence_list:
  for w in sentence:
    if w['pos'] in ['名詞','動詞','形容詞']:
      result_list.append(w['surface'])

hist_dic = collections.Counter(list(map(lambda x:x[1],sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True))))

x = list(map(lambda x:str(x[0]),hist_dic.items()))[0:20]
y = list(map(lambda x:x[1],hist_dic.items()))[0:20]

plt.bar(x,y)


# ### 39問 Zipfの法則

# In[19]:


result_list = []
for sentence in sentence_list:
  for w in sentence:
    if w['pos'] in ['名詞','動詞','形容詞']:
      result_list.append(w['surface'])

sorted_list = sorted(collections.Counter(result_list).items(), key=lambda x:x[1], reverse=True)

x = list(range(1,len(sorted_list) + 1))
y = list(map(lambda x:x[1],sorted_list))

plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.scatter(x,y)


# ## 関連記事
# - [第1章 準備運動](/ml/nlp100/01/)
# - [第2章 UNIXコマンド](/ml/nlp100/02/)
# - [第3章 正規表現](/ml/nlp100/03/)
# - [第4章 形態素解析](/ml/nlp100/04/)
# - [第5章 係り受け解析](/ml/nlp100/05/)
# - [第6章 機械学習](/ml/nlp100/06/)
# - [第7章 単語ベクトル](/ml/nlp100/07/)
# - [第8章 ニューラルネット](/ml/nlp100/08/)
# - [第9章 RNN,CNN](/ml/nlp100/09/)
# - [第10章 機械翻訳](/ml/nlp100/10/)
