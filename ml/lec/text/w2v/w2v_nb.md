## word2vec と doc2vec

単語や文章を分散表現（意味が似たような単語や文章を似たようなベクトルとして表現）を取得します。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/lec/text/w2v/w2v_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/lec/text/w2v/w2v_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G6032



```python
!python -V
```

    Python 3.8.5


基本的なライブラリをインポートしそのバージョンを確認しておきます。tensorflowとkerasuのversionも確認します。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

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
```

    matplotlib version : 3.3.2
    scipy version : 1.5.2
    numpy version : 1.18.5
    tensorflow version :  2.3.1
    keras version :  2.4.0


### テキストデータの取得

著作権の問題がない青空文庫からすべての作品をダウンロードしてきます。gitがかなり重いので、最新の履歴だけを取得します。

```bash
git clone --depth 1 https://github.com/aozorabunko/aozorabunko.git
```

実際のファイルはcardsにzip形式として保存されているようです。ディレクトリの個数を確認してみます。


```python
!ls ./aozorabunko/cards/* | wc -l
```

       19636


zipファイルだけzipsに移動させます。

```bash
find ./aozorabunko/cards/ -name *.zip | xargs -I{} cp {} -t ./zips/
```


```python
!ls ./zips/ | head -n 5
```

    1000_ruby_2956.zip
    1001_ruby_2229.zip
    1002_ruby_20989.zip
    1003_ruby_2008.zip
    1004_ruby_2053.zip



```python
!ls ./zips/ | wc -l
```

       16444


となり、16444個のzipファイルがある事が分かります。こちらをすべて解凍し、ディレクトリを移動させます。

```bash
for i in `ls`; do [[ ${i##*.} == zip ]] && unzip -o $i -d ../texts/; done
```

これで、textｓというディレクトリにすべての作品のテキストファイルがインストールされました。


```python
!ls ./texts/ | grep miyazawa
```

    miyazawa_kenji_zenshu.txt
    miyazawa_kenji_zenshuno_kankoni_saishite.txt
    miyazawa_kenjino_sekai.txt
    miyazawa_kenjino_shi.txt



```python
!ls ./texts/ | grep ginga_tetsudo
```

    ginga_tetsudono_yoru.txt


となり、宮沢賢治関連の作品も含まれていることが分かります。銀河鉄道の夜もあります。

## 銀河鉄道の夜を使ったword2vec

今回はすべてのテキストファイルを対象にするには時間がかかるので、同じ岩手県出身の、高校の先輩でもある宮沢賢治の作品を例に取りword2vecを試してみます。
しかし、ファイルの中身を見てみると、


```python
!head ./texts/ginga_tetsudono_yoru.txt
```

    
    
    
    
    
    
    
    
    
    



```python
!nkf --guess ./texts/ginga_tetsudono_yoru.txt
```

    Shift_JIS (CRLF)


となりshift_jisで保存されていることが分かります。


```python
!nkf -w ./texts/ginga_tetsudono_yoru.txt > ginga.txt
```

と、ディレクトリを変更し、ファイル名も変更します。


```python
!cat ginga.txt | head -n 25
```

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    cat: stdout: Broken pipe



```python
!cat ginga.txt | tail -n 25
```

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


となり、ファイルの先頭と、末尾に参考情報が載っているほかは、ちゃんとテキストとしてデータが取れている模様です。
先ず、この辺の前処理を行います。


```python

```


```python

```


```python

```


```python

```


```python
import re

with open('ginga.txt', mode='r') as f:
  all_sentence = f.read()
```

全角、半角の空白、改行コード、縦線(|)をすべて削除します。正規表現を利用します。


```python
all_sentence = all_sentence.replace(" ", "").replace("　","").replace("\n","").replace("|","")
```

《》で囲まれたルビの部分を削除します。正規表現を利用します。


```python
all_sentence = re.sub("《[^》]+》", "", all_sentence)
```

----------の部分で分割を行い、2番目の要素を取得します。


```python
all_sentence = re.split("\-{8,}", all_sentence)[2]
```

。で分割し、文ごとにリストに格納します。


```python
sentence_list = all_sentence.split("。")
sentence_list = [ s + "。" for s in sentence_list]
sentence_list[:5]
```




    ['［＃７字下げ］一、午后の授業［＃「一、午后の授業」は中見出し］「ではみなさんは、さういふふうに川だと云はれたり、乳の流れたあとだと云はれたりしてゐたこのぼんやりと白いものがほんたうは何かご承知ですか。',
     '」先生は、黒板に吊した大きな黒い星座の図の、上から下へ白くけぶった銀河帯のやうなところを指しながら、みんなに問をかけました。',
     'カムパネルラが手をあげました。',
     'それから四五人手をあげました。',
     'ジョバンニも手をあげやうとして、急いでそのまゝやめました。']



最初の文は不要なので削除します。


```python
sentence_list = sentence_list[1:]
sentence_list[:5]
```




    ['」先生は、黒板に吊した大きな黒い星座の図の、上から下へ白くけぶった銀河帯のやうなところを指しながら、みんなに問をかけました。',
     'カムパネルラが手をあげました。',
     'それから四五人手をあげました。',
     'ジョバンニも手をあげやうとして、急いでそのまゝやめました。',
     'たしかにあれがみんな星だと、いつか雑誌で読んだのでしたが、このごろはジョバンニはまるで毎日教室でもねむく、本を読むひまも読む本もないので、なんだかどんなこともよくわからないといふ気持ちがするのでした。']



となり、不要な部分を削除し、一文ごとにリストに格納できました。前処理は終了です。

## janomeによる形態素解析

janomeは日本語の文章を形態素ごとに分解する事が出来るツールです。同じようなツールとして、MecabやGinzaなどがあります。一長一短があると思いますが、ここではjanomeを利用します。


```python
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
```

    ['先生', '黒板', '吊す', '星座', '図', '上', '下', 'けぶる', '銀河', '帯']


## 単語のカウント

単語のカウントを行い、出現頻度の高いベスト10を抽出してみます。名詞のみに限定した方が良かったかもしれません。


```python
import collections

count = collections.Counter(word_list)
count.most_common()[:10]
dict(count.most_common())['銀河']
dict(count.most_common())['ジョバンニ']
```




    191



## gensimに含まれるword2vecを用いた学習

word2vecを用いて、word_listの分散表現を取得します。使い方はいくらでも検索できますので、ここでは割愛します。単語のリストを渡せば、ほぼ自動的に分散表現を作ってくれます。


```python
from gensim.models import word2vec

model = word2vec.Word2Vec(word_list, size=100, min_count=5, window=5, iter=1000, sg=0)
```




    ['先生',
     '黒板',
     '吊す',
     '星座',
     '図',
     '上',
     '下',
     'けぶる',
     '銀河',
     '帯',
     'やう',
     'ところ',
     '指す',
     'みんな',
     '問',
     'かける',
     'カムパネルラ',
     '手',
     'あげる',
     '四',
     '五',
     '人',
     '手',
     'あげる',
     'ジョバンニ',
     '手',
     'あげる',
     'やう',
     '急ぐ',
     'やめる',
     'あれ',
     'みんな',
     '星',
     'いつか',
     '雑誌',
     '読む',
     'の',
     'このごろ',
     'ジョバンニ',
     '毎日',
     '教室',
     '本',
     '読む',
     'ひま',
     '読む',
     '本',
     'こと',
     'わかる',
     '気持ち',
     'する',
     'の',
     '先生',
     'それ',
     '見る',
     '附ける',
     'の',
     'ジョバンニ',
     'さん',
     'あなた',
     'わかる',
     'ゐる',
     'する',
     'ジョバンニ',
     '勢',
     '立ちあがる',
     '立つ',
     '見る',
     'それ',
     '答',
     'へる',
     'こと',
     'できる',
     'の',
     'ザネリ',
     '前',
     '席',
     'ふり',
     'ジョバンニ',
     '見る',
     'く',
     'わら',
     'ひる',
     'ジョバンニ',
     'どぎまぎ',
     'する',
     '赤',
     'なる',
     'しまふ',
     '先生',
     '云',
     'ひる',
     '望遠鏡',
     '銀河',
     'よる',
     'くる',
     '調べる',
     '銀河',
     '大体',
     'する',
     '星',
     'ジョバンニ',
     '思ふ',
     'こんど',
     '答',
     'へる',
     'こと',
     'できる',
     '先生',
     '困る',
     'やう',
     'する',
     '眼',
     'カムパネルラ',
     '方',
     '向ける',
     'カムパネルラ',
     'さん',
     '名指す',
     '元気',
     '手',
     'あげる',
     'カムパネルラ',
     'ぢ',
     'ぢ',
     '立ち上る',
     '答',
     'できる',
     '先生',
     '意外',
     'やう',
     'ぢ',
     'カムパネルラ',
     '見る',
     'ゐる',
     '急ぐ',
     '云',
     'ひる',
     '自分',
     '星図',
     '指す',
     'ぼんやり',
     '銀河',
     '望遠鏡',
     '見る',
     'たくさん',
     '星',
     '見える',
     'の',
     'ジョバンニ',
     'さん',
     'する',
     'する',
     'ジョバンニ',
     '赤',
     'なる',
     'うなづく',
     'いつか',
     'ジョバンニ',
     '眼',
     'なか',
     '涙',
     'なる',
     'する',
     '僕',
     '知る',
     'ゐる',
     'の',
     'カムパネルラ',
     '知る',
     'ゐる',
     'それ',
     'いつか',
     'カムパネルラ',
     'お父さん',
     '博士',
     'うち',
     'カムパネルラ',
     'いっしょ',
     '読む',
     '雑誌',
     'なか',
     'ある',
     'の',
     'それ',
     'どこ',
     'カムパネルラ',
     '雑誌',
     '読む',
     'お父さん',
     '書',
     '斎',
     '巨',
     '本',
     'もつ',
     'くる',
     'ぎん',
     'ところ',
     'ひろげる',
     'まっ黒',
     '頁',
     'いっぱい',
     '点々',
     'ある',
     '写真',
     '二',
     '人',
     'いつ',
     '見る',
     'の',
     'それ',
     'カムパネルラ',
     '忘れる',
     '筈',
     '返事',
     'する',
     'の',
     'このごろ',
     'ぼく',
     '朝',
     '午后',
     '仕事',
     '学校',
     '出る',
     'みんな',
     '遊ぶ',
     'カムパネルラ',
     '物',
     '云',
     'やう',
     'なる',
     'カムパネルラ',
     'それ',
     '知る',
     '気の毒',
     'がる',
     '返事',
     'する',
     'の',
     '考へる',
     'ぶん',
     'カムパネルラ',
     'あはれ',
     'やう',
     '気',
     'する',
     'の',
     '先生',
     '云',
     'ひる',
     '天の川',
     'ほん',
     'うに',
     '川',
     '考へる',
     '一つ',
     '一つ',
     '星',
     'みんな',
     '川',
     'そこ',
     '砂',
     '砂利',
     '粒',
     'あたる',
     'わけ',
     'これ',
     '巨',
     '乳',
     '流れ',
     '考へる',
     '天の川',
     '似る',
     'ゐる',
     '星',
     'みな',
     '乳',
     'なか',
     '細か',
     'うかぶ',
     'ゐる',
     '脂',
     '油',
     '球',
     'あたる',
     'の',
     'そん',
     '何',
     '川',
     '水',
     'あたる',
     '云',
     'ひる',
     'それ',
     '真空',
     '光',
     'ある',
     'さ',
     '伝へる',
     'もの',
     '太陽',
     '地球',
     'なか',
     '浮ぶ',
     'ゐる',
     'の',
     'つまり',
     '私',
     'ども',
     '天の川',
     '水',
     'なか',
     '棲む',
     'ゐる',
     'わけ',
     '天の川',
     '水',
     'なか',
     '四方',
     '見る',
     'ちゃう',
     '水',
     '見える',
     'やう',
     '天の川',
     '底',
     'ところ',
     '星',
     'たくさん',
     '集う',
     '見え',
     'ぼんやり',
     '見える',
     'の',
     '模型',
     'ごらん',
     'なさる',
     '先生',
     '中',
     'たくさん',
     '光る',
     '砂',
     'つぶ',
     '入る',
     '両面',
     '凸レンズ',
     '指す',
     '天の川',
     '形',
     'ちゃう',
     'こんな',
     'の',
     'いちいち',
     '光る',
     'つぶ',
     'みんな',
     '私',
     'ども',
     '太陽',
     'やう',
     'ぶん',
     '光る',
     'ゐる',
     '星',
     '考へる',
     '私',
     'ども',
     '太陽',
     'ほる',
     '中ごろ',
     'ある',
     '地球',
     '近く',
     'ある',
     'する',
     'みなさん',
     '夜',
     'まん中',
     '立つ',
     'レンズ',
     '中',
     '見る',
     'はする',
     'ごらん',
     'なさる',
     'こっち',
     '方',
     'レンズ',
     'づく',
     '光る',
     '粒',
     '星',
     '見える',
     'する',
     'こっち',
     'こっち',
     '方',
     'ガラス',
     '光る',
     '粒',
     '星',
     'たくさん',
     '見える',
     'の',
     '見える',
     'これ',
     '今日',
     '銀河',
     '説',
     'の',
     'そん',
     'レンズ',
     'さ',
     'どれ',
     '位',
     'ある',
     '中',
     'さまざま',
     '星',
     '時間',
     '次',
     '理科',
     '時間',
     'お話',
     'する',
     '今日',
     '銀河',
     'お祭',
     'の',
     'みなさん',
     '外',
     'でる',
     'ごらん',
     'なさる',
     'こ',
     '本',
     'ノート',
     'おす',
     'まひ',
     'なさる',
     '教室',
     '中',
     '机',
     '蓋',
     'あける',
     'しめる',
     '本',
     '重ねる',
     'する',
     '音',
     'みんな',
     '立つ',
     '礼',
     'する',
     '教室',
     '出る',
     '＃',
     '７',
     '字',
     '下げ',
     '二',
     '活版',
     '所',
     '＃「〔',
     '二',
     '活版',
     '所',
     '見出し',
     'ジョバンニ',
     '学校',
     '門',
     '出る',
     'とき',
     '組',
     '七',
     '八',
     '人',
     '家',
     '帰る',
     'カムパネルラ',
     'まん中',
     'する',
     '校庭',
     '隅',
     '桜',
     '木',
     'ところ',
     '集まる',
     'ゐる',
     'それ',
     'こむ',
     '星祭',
     'あかり',
     'こしらえる',
     '川',
     '流す',
     '烏瓜',
     '取る',
     '行く',
     '相談',
     'の',
     'ジョバンニ',
     '手',
     '振る',
     '学校',
     '門',
     '出る',
     '来る',
     '町',
     '家々',
     'はこぶ',
     '銀河',
     '祭り',
     'いち',
     'ゐる',
     '葉',
     '玉',
     'つるす',
     'ひのき',
     '枝',
     'あかり',
     'つける',
     'いろいろ',
     '仕度',
     'する',
     'ゐる',
     'の',
     '家',
     '帰る',
     'ジョバンニ',
     '町',
     '三つ',
     '曲る',
     'ある',
     '活版',
     '処',
     'いう',
     '入口',
     '計算',
     '台',
     '居る',
     'だぶだぶ',
     'シャツ',
     '着る',
     '人',
     'おじぎ',
     'する',
     'ジョバンニ',
     '靴',
     'ぬぐ',
     '上る',
     '突き当り',
     '扉',
     'あける',
     '中',
     '昼',
     '電',
     '燈',
     'つく',
     'たくさん',
     '輪',
     '転',
     '器',
     'ば',
     'とむ',
     'はる',
     'きれる',
     '頭',
     'しばる',
     'ラムプ',
     '［＃「',
     'ラムプ',
     '傍線',
     'シェード',
     'かける',
     'する',
     '人',
     'たち',
     '何',
     '歌',
     'ふる',
     'やう',
     '読む',
     '数',
     'する',
     'たくさん',
     '働く',
     '居る',
     'ジョバンニ',
     '入口',
     '三',
     '番目',
     '卓子',
     '座る',
     '人',
     '所',
     '行く',
     'おじぎ',
     'する',
     '人',
     '棚',
     'さがす',
     'これ',
     '拾う',
     '行ける',
     '云',
     'ひる',
     '一',
     '枚',
     '紙切れ',
     '渡す',
     'ジョバンニ',
     '人',
     '卓子',
     '足もと',
     '一つ',
     '函',
     'とりだす',
     'ふる',
     'の',
     '電',
     '燈',
     'たくさん',
     'つく',
     'たてかける',
     'ある',
     '壁',
     '隅',
     '所',
     'しゃがむ',
     '込む',
     'ピンセット',
     '粟粒',
     'ぐらゐの',
     '活字',
     '次',
     '次',
     '拾',
     'ひる',
     'はじめる',
     '胸',
     'あて',
     'する',
     '人',
     'ジョバンニ',
     'うし',
     'ろ',
     '通る',
     'よう',
     '虫めがね',
     '君',
     '云',
     'ひる',
     '近く',
     '四',
     '五',
     '人',
     '人',
     'たち',
     '声',
     'たてる',
     'こっち',
     '向く',
     '冷',
     'く',
     'わら',
     'ひる',
     'ジョバンニ',
     '何',
     'ん',
     '眼',
     '拭',
     'ひる',
     '活字',
     'ひろ',
     'ひる',
     '六',
     '時',
     'うる',
     'たつ',
     'ころ',
     'ジョバンニ',
     '拾う',
     '活字',
     '入れる',
     '箱',
     'いちど',
     '手',
     'もつ',
     '紙きれ',
     '引き合せる',
     'さっき',
     '卓子',
     '人',
     '持つ',
     '来る',
     '人',
     '黙る',
     'それ',
     '受け取る',
     '微か',
     'うなづく',
     'ジョバンニ',
     'おじぎ',
     'する',
     '扉',
     'あける',
     'さっき',
     '計算',
     '台',
     'ところ',
     '来る',
     'さっき',
     '白',
     '服',
     '着る',
     '人',
     'だまる',
     '銀貨',
     '一つ',
     'ジョバンニ',
     '渡す',
     'ジョバンニ',
     '俄',
     '顔',
     'いる',
     'なる',
     '威勢',
     'おじぎ',
     'する',
     '台の下',
     '置く',
     '鞄',
     'もつ',
     'もてる',
     '飛びだす',
     '元気',
     '口笛',
     '吹く',
     'パン',
     '屋',
     '寄る',
     'パン',
     '塊',
     '一つ',
     '角砂糖',
     '一',
     '袋',
     '買',
     'ひる',
     '一目散',
     '走る',
     'だす',
     '＃',
     '７',
     '字',
     '下げ',
     '三',
     '家',
     '［＃「',
     '三',
     '家',
     '見出し',
     'ジョバンニ',
     '勢',
     '帰る',
     '来る',
     'の',
     '裏町',
     '家',
     '三つ',
     'ならぶ',
     '入口',
     '一番',
     '左側',
     '空',
     '箱',
     '紫いろ',
     'ケール',
     'アスパラガス',
     '植える',
     'ある',
     '二つ',
     '窓',
     '日覆',
     'ひる',
     'たま',
     'なる',
     'ゐる',
     'お母さん',
     'いま',
     '帰る',
     '工合',
     'ジョバンニ',
     '靴',
     'ぬぐ',
     '云',
     'ひる',
     'ジョバンニ',
     '仕事',
     '今日',
     'わたし',
     'はず',
     'うつ',
     '工合',
     'がい',
     'ジョバンニ',
     '玄関',
     '上る',
     '行く',
     'ジョバンニ',
     'お母さん',
     '入口',
     '室',
     '巾',
     '被る',
     '寝る',
     'ゐる',
     'の',
     'ジョバンニ',
     '窓',
     'あける',
     'お母さん',
     '今日',
     '角砂糖',
     '買う',
     'くる',
     '牛乳',
     '入れる',
     'あげる',
     'やう',
     '思う',
     'お前',
     'さき',
     'あがる',
     'あたし',
     'ん',
     'お母さん',
     '姉さん',
     'いつ',
     '帰る',
     '三',
     '時',
     'ころ',
     '帰る',
     'みんな',
     'そこら',
     'する',
     'くれる',
     'お母さん',
     '牛乳',
     '来る',
     'ゐる',
     'ん',
     '来る',
     'ぼく',
     '行く',
     'とる',
     '来る',
     'やう',
     'あたし',
     'でる',
     'いる',
     'ん',
     'お前',
     'さき',
     'あがる',
     '姉さん',
     'トマト',
     '何',
     'こしらえる',
     'そこ',
     '置く',
     '行く',
     'ぼく',
     'たべる',
     'やう',
     'ジョバンニ',
     '窓',
     'ところ',
     'トマト',
     '皿',
     'とる',
     'パン',
     'いっしょ',
     'たべる',
     'お母さん',
     'ぼく',
     'お父さん',
     '帰る',
     'くる',
     '思ふ',
     'あたし',
     'する',
     '思ふ',
     '思ふ',
     '今朝',
     '新聞',
     '今年',
     '北の方',
     '漁',
     'へん',
     '書く',
     'ある',
     'お父さん',
     '漁',
     '出る',
     'ゐる',
     'しれる',
     '出る',
     'ゐる',
     'お父さん',
     '監獄',
     '入る',
     'やう',
     'こと',
     'する',
     '筈',
     'ん',
     '前',
     'お父さん',
     '持つ',
     'くる',
     '学校',
     '寄贈',
     'する',
     '巨',
     '蟹',
     '甲',
     'ら',
     'の',
     'なか',
     'ひる',
     '角',
     '今',
     'みんな',
     '標本',
     '室',
     'ある',
     'ん',
     '六',
     '年生',
     '授業',
     'とき',
     '先生',
     'はる',
     'はる',
     '教室',
     '持つ',
     '行く',
     '一昨年',
     '修学旅行',
     '以下',
     '数',
     '文字',
     '分',
     '空白',
     'お父さん',
     '次',
     'ラッコ',
     '上着',
     'もつ',
     'くる',
     'みんな',
     'ぼく',
     'それ',
     '云',
     'ふよ',
     'ひやかす',
     'やう',
     '云',
     'ふむ',
     '悪口',
     '云',
     'ふる',
     'カムパネルラ',
     '云',
     'カムパネルラ',
     'みんな',
     'こと',
     '云',
     '気の毒',
     'さ',
     'うに',
     'する',
     'ゐる',
     '人',
     'うち',
     'お父さん',
     'ちる',
     'ゃうどおまへたちのやうに',
     'とき',
     '友達',
     'お父さん',
     'ぼく',
     'つれる',
     'カムパネルラ',
     'うち',
     'もつれる',
     '行く',
     'ころ',
     ...]



### 分散行列


```python
model.wv.vectors
```




    array([[ 0.6690906 , -1.8134489 , -0.50446075, ...,  0.22950223,
            -0.24067923, -0.45016605],
           [-0.0081544 ,  0.88565207, -0.6879916 , ...,  0.37250426,
            -0.37231675, -0.23907655],
           [-0.06978781, -0.4953329 , -0.1721944 , ..., -0.34273872,
            -0.676891  , -0.7721713 ],
           ...,
           [ 0.09245484, -0.6152532 , -0.20881364, ...,  0.04918382,
             0.10831165,  0.15404673],
           [ 0.6117021 , -0.9071201 ,  0.8482464 , ...,  0.27837202,
             0.4135082 ,  0.03481499],
           [-3.1874008 , -0.96890706,  1.3699456 , ..., -1.5262604 ,
            -0.79284537, -0.08142332]], dtype=float32)



### 分散行列の形状確認

443個の単語について、100次元のベクトルが生成されました。


```python
model.wv.vectors.shape
```




    (408, 100)



全単語数は、


```python
len(set(word_list))
```




    2019



ですが、word2vecのmin_countを5にしているので、その文単語数が少なくなっています。


```python
model.wv.index2word[:10]
print(model.__dict__['wv']['銀河'])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-83-eb818cbee18c> in <module>
          1 model.wv.index2word[:10]
    ----> 2 print(model.__dict__['wv']['銀河'])
    

    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in __getitem__(self, entities)
        351         if isinstance(entities, string_types):
        352             # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
    --> 353             return self.get_vector(entities)
        354 
        355         return vstack([self.get_vector(entity) for entity in entities])


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in get_vector(self, word)
        469 
        470     def get_vector(self, word):
    --> 471         return self.word_vec(word)
        472 
        473     def words_closer_than(self, w1, w2):


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in word_vec(self, word, use_norm)
        466             return result
        467         else:
    --> 468             raise KeyError("word '%s' not in vocabulary" % word)
        469 
        470     def get_vector(self, word):


    KeyError: "word '銀河' not in vocabulary"



```python
model.wv.vectors[0]
```




    array([ 0.6690906 , -1.8134489 , -0.50446075,  1.0823753 ,  1.0012556 ,
            0.17782725,  0.39993393, -0.7516008 ,  1.2903652 , -0.79781705,
            0.54625   ,  0.18831149, -0.10043006, -0.68015933, -0.22242035,
            0.72337776, -0.31982303, -0.9627689 , -0.22933453, -0.04989067,
            0.16735213, -0.02974823, -1.3249292 , -0.27127397,  0.42874482,
            0.01675199,  0.8601299 , -0.85613954, -0.79393893,  0.12290027,
            0.6677782 ,  0.4430345 , -0.15914361,  0.92404836,  0.7163351 ,
            0.27910623, -0.09720881,  0.68278235,  1.1329095 , -0.7275171 ,
           -0.99282736,  0.09739671,  1.4512872 , -0.29004535,  1.0013556 ,
           -0.78484267, -0.44537067, -0.17693432,  0.00596993, -0.2871559 ,
           -1.0671868 ,  0.35299167,  0.6387847 , -1.3476065 , -0.51196575,
           -0.09386528,  0.45643848,  0.6014701 , -0.29185364, -0.6555386 ,
            0.3910473 , -0.324209  , -0.5417036 ,  0.08710421, -1.1519334 ,
            0.08187845,  0.7924016 , -0.00519154, -0.2600619 ,  0.96227944,
           -0.12906776,  0.5477753 , -1.1792823 ,  0.20154633, -0.7700448 ,
            1.0795287 ,  0.538111  ,  0.24918164,  0.48424342, -0.22555429,
           -0.46567798,  0.2812898 ,  0.6985383 ,  1.2283741 ,  1.0126857 ,
            0.4486654 , -1.0553776 ,  0.07277382, -0.3959616 , -0.9007682 ,
            0.2317583 ,  0.82350373,  0.42838004,  1.0937254 , -0.36720416,
            1.062953  ,  0.7355613 ,  0.22950223, -0.24067923, -0.45016605],
          dtype=float32)




```python
model.wv.__getitem__("銀河")
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-82-7276877cfefc> in <module>
    ----> 1 model.wv.__getitem__("銀河")
    

    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in __getitem__(self, entities)
        351         if isinstance(entities, string_types):
        352             # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
    --> 353             return self.get_vector(entities)
        354 
        355         return vstack([self.get_vector(entity) for entity in entities])


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in get_vector(self, word)
        469 
        470     def get_vector(self, word):
    --> 471         return self.word_vec(word)
        472 
        473     def words_closer_than(self, w1, w2):


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in word_vec(self, word, use_norm)
        466             return result
        467         else:
    --> 468             raise KeyError("word '%s' not in vocabulary" % word)
        469 
        470     def get_vector(self, word):


    KeyError: "word '銀河' not in vocabulary"


### cos類似度による単語抽出

ベクトルの内積を計算することにより、指定した単語に類似した単語をその$\cos$の値と一緒に抽出する事ができます。


```python
print(model.wv.most_similar("銀河"))
print(model.wv.most_similar("本"))
print(model.wv.most_similar("ジョバンニ"))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-81-081816b8f0b5> in <module>
    ----> 1 print(model.wv.most_similar("銀河"))
          2 print(model.wv.most_similar("本"))
          3 print(model.wv.most_similar("ジョバンニ"))


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in most_similar(self, positive, negative, topn, restrict_vocab, indexer)
        551                 mean.append(weight * word)
        552             else:
    --> 553                 mean.append(weight * self.word_vec(word, use_norm=True))
        554                 if word in self.vocab:
        555                     all_words.add(self.vocab[word].index)


    ~/anaconda3/lib/python3.8/site-packages/gensim/models/keyedvectors.py in word_vec(self, word, use_norm)
        466             return result
        467         else:
    --> 468             raise KeyError("word '%s' not in vocabulary" % word)
        469 
        470     def get_vector(self, word):


    KeyError: "word '銀河' not in vocabulary"


### 単語ベクトルによる演算

足し算するにはpositiveメソッドを引き算にはnegativeメソッドを利用します。

まず、銀河＋男を計算します。


```python
model.wv.most_similar(positive=["銀河", "ジョバンニ"])
```

次に銀河＋ジョバンニー家を計算します。


```python
model.wv.most_similar(positive=["銀河", "ジョバンニ"], negative=["家"])
```

## doc2vec

文章毎にタグ付けされたTaggedDocumentを作成します。


```python
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

tagged_doc_list = []

for i, sentence in enumerate(word_per_sentence_list):
  tagged_doc_list.append(TaggedDocument(sentence, [i]))

print(tagged_doc_list[0])
```


```python
model = Doc2Vec(documents=tagged_doc_list, vector_size=100, min_count=5, window=5, epochs=20, dm=0)
```


```python
word_per_sentence_list[0]
```


```python
model.docvecs[0]
```

most_similarで類似度が高い文章のIDと類似度を取得することが出来ます。


```python
model.docvecs.most_similar(0)
```


```python
for p in model.docvecs.most_similar(0):
  print(word_per_sentence_list[p[0]])
```

感覚的ですが、似たような文章が抽出されています。


```python

```


```python

```
