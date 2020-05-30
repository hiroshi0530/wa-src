
# coding: utf-8

# ## [第1章 準備運動](https://nlp100.github.io/ja/ch01.html)
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/ml/nlp100/01/01_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## 解答

# ### 00問

# In[1]:


a = 'stressed'

a[::-1]


# ### 01問

# In[4]:


a = 'パタトクカシーー'

a[1::2]


# ### 02問

# In[5]:


a = 'パトカー'
b = 'タクシー'

''.join([ i + j for i,j in zip(a,b)])


# ### 03問

# In[6]:


a = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'

list(map(lambda x: len(x.replace(',','').replace('.','')), a.split(' ')))


# ### 04問

# In[7]:


a = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
b = [1,5,6,7,8,9,15,16,19]

ret = {}
for i, s in enumerate(map(lambda x: x.replace(',','').replace('.',''), a.split(' '))):
  if i + 1 in b:
    ret.update({
      s[0] : i + 1
    })
  else:
    ret.update({
      s[0:2] : i + 1
    })
    
ret


# ### 05問

# In[8]:


def n_gram(arg, n):
  return [arg[i: i + n] for i in range(len(arg)) if i + n <= len(arg)]

a = 'I am an NLPer'

print(n_gram(a.split(' '), 2))
print(n_gram(a, 2))


# ### 06問

# In[9]:


a = 'paraparaparadise'
b = 'paragraph'

X = set(n_gram(a,2))
Y = set(n_gram(b,2))

print('和集合 : ',X | Y)
print('積集合 : ', X ^ Y)
print('差集合 : ', X - Y)

if 'se' in X:
  print('Xにseあり')
else:
  print('Xにseなし')
    
if 'se' in Y:
  print('Yにseあり')
else:
  print('Yにseなし')


# ### 07問

# In[10]:


def template(x,y,z):
  return '{}時の{}は{}'.format(x,y,z)

template(12,'気温',22.4)


# ### 08問

# In[11]:


def cipher(arg):
  return ''.join([chr(219 - ord(i)) if i.islower() else i for i in arg])

orig = 'sdf234DSFsdf'
print('orig      : ',orig)
encrypt = cipher(orig)
print('encrypted : ',encrypt)
decrypt = cipher(encrypt)
print('decrypted : ',decrypt)
print('judgement : ', orig == decrypt)


# ### 09問

# In[12]:


import random 

a = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'

def _09(arg):
  return ' '.join(list(map(lambda x: x[0] + ''.join(random.sample(x[1:-1], len(x[1:-1]))) + x[-1] if len(x) > 4 else x , a.split(' '))))

_09(a)


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
