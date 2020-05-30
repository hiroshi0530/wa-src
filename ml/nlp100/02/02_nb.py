
# coding: utf-8

# ## [第2章 UNIXコマンド](https://nlp100.github.io/ja/ch02.html)
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


# テキストファイルをダウンロードします。

# In[4]:


get_ipython().system('wget https://nlp100.github.io/data/popular-names.txt -O ./popular-names.txt')


# ファイルは、「アメリカで生まれた赤ちゃんの「名前」「性別」「人数」「年」をタブ区切り形式で格納したファイルである」という事ですが、どんなファイルか見てみます。

# In[5]:


get_ipython().system('head -n 5 popular-names.txt')


# In[6]:


get_ipython().system('tail -n 5 popular-names.txt')


# ## 解答

# ### 共通部分

# In[7]:


file_name = './popular-names.txt'


# ### 10問

# In[8]:


with open(file_name,mode='r') as f:
  print("line number by python : ", len(f.readlines()))

get_ipython().system('echo "line number by unix   : "`cat $file_name | wc -l`')


# ### 11問

# In[9]:


with open(file_name,mode='r') as f:
  with open('11_python_out', mode='w') as p:
    for s in f.readlines():
      p.write(s.replace('\t',' '))

get_ipython().system('expand -t 1 $file_name > 11_unix_out')

get_ipython().system('md5 $file_name')
get_ipython().system('md5 11_python_out')
get_ipython().system('md5 11_unix_out')


# ### 12問

# In[10]:


with open(file_name,mode='r') as f:
  with open('col1.txt', mode='w') as c1:
    with open('col2.txt', mode='w') as c2:
      for s in f.readlines():
        c1.write(s.split('\t')[0] + '\n')
        c2.write(s.split('\t')[1] + '\n')
        
get_ipython().system('cut -f 1 $file_name > u_col1.txt')
get_ipython().system('cut -f 2 $file_name > u_col2.txt')

get_ipython().system('md5 col1.txt')
get_ipython().system('md5 u_col1.txt')
get_ipython().system('md5 col2.txt')
get_ipython().system('md5 u_col2.txt')


# ### 13問

# In[11]:


with open('col1.txt', mode='r') as c1:
  with open('col2.txt', mode='r') as c2:
    with open('13_merge.txt', mode='w') as w:
      for s1,s2 in zip(c1.readlines(), c2.readlines()):
        w.write(s1.replace('\n','') + '\t' + s2.replace('\n','') + '\n')
        
get_ipython().system('paste col1.txt col2.txt > u_13_merge.txt')

get_ipython().system('md5 13_merge.txt')
get_ipython().system('md5 u_13_merge.txt')


# ### 14問

# In[12]:


def print_n(n):
  with open(file_name,mode='r') as f:
    for i,s in enumerate(f.readlines()[:n]):
      print(s.replace('\n',''))

print('print by python')
print_n(4)
print()
print('print by unix')
get_ipython().system('head -n 4 $file_name')


# ### 15問

# In[13]:


def print_n(n):
  with open(file_name,mode='r') as f:
    for i in f.readlines()[-1 * n:]: 
      print(i.replace('\n',''))

print('print by python')
print_n(4)
print()
print('print by unix')
get_ipython().system('tail -n 4 $file_name')


# ### 16問

# In[14]:


def devide_n(n):
  with open(file_name,mode='r') as f:
    lines = f.readlines()
    num = int(len(lines) / n)
    
    for k in range(n + 1):
      with open('16_No_{:06d}.txt'.format(k), mode='w') as f:
        for i in lines[num * k: num * (k + 1)]:
          f.write(i)

devide_n(3)

def get_line(n):
  with open(file_name, mode='r') as f:
    return int(len(f.readlines()) / n)

get_ipython().system('split -l {get_line(3)} -a 4 $file_name ')

print('files by python')
get_ipython().system('ls | grep 16_No | xargs -I{} md5 {}')
print()
print('files by unix')
get_ipython().system('ls | grep xaa | xargs -I{} md5 {}')


# ### 17問

# In[15]:


with open(file_name,mode='r') as f:
  s_set = set([s.split('\t')[0] for s in f.readlines()])
  print('### python ###')
  print(sorted(list(s_set)))

print()
print('### unix ###')
get_ipython().system("cut -f 1 $file_name | sort | uniq | tr '\\n' ', '")


# ### 18問
# 全部表示すると長いので、先頭から10行だけを表示しています。

# In[16]:


with open(file_name,mode='r') as f:
  print("### python ###")
  for s in sorted([s.split('\t') for s in f.readlines()],key=lambda x:x[2],reverse=True)[0:10]:
    print(s)

print()
get_ipython().system('echo "### unix ###"')
get_ipython().system('cat $file_name | sort -k 3 -r | head -n 10')


# ### 19問
# 全部表示すると長いので、先頭から10行だけを表示しています。

# In[17]:


import collections

with open(file_name,mode='r') as f:
  s_list = [s.split('\t')[0] for s in f.readlines()]
  print('### python sort ###')
  for s in sorted(collections.Counter(s_list).items(),key=lambda x:x[1], reverse=True )[0:10]:
    print(str(s[1]) + ' ' + s[0])

print()
print('### unix sort ###')
get_ipython().system('cut -f 1 $file_name | sort | uniq -c | sort -r | head -n 10')


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
