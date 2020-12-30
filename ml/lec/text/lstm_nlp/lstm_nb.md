
## kerasとLSTMを用いた文章の生成

LSTMを用いて文章を生成することが出来ます。文章を時系列データとして訓練データとして学習し、文章を入力し、次の文字列を予測するようなっモデルを生成します。今回は前回青空文庫からダウンロードした、宮沢賢治の銀河鉄道の夜を学習データとして採用し、LSTMによって、宮沢賢治風の文章を作成してみようと思います。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_nlp/lstm_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/lec/text/lstm_nlp/lstm_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


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


基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

import tensorflow as tf
from tensorflow import keras
import gensim
import gensim

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)
print('gensim version : ', gensim.__version__)
```

    matplotlib version : 3.0.3
    scipy version : 1.4.1
    numpy version : 1.19.4
    tensorflow version :  2.4.0
    keras version :  2.4.0
    gensim version :  3.8.3


## テキストファイルの前処理

題材として、宮沢賢治の銀河鉄道の夜を利用します。既に著作権フリーなので、自由に利用できます。ちなみに、宮沢賢治は同郷で高校の先輩ですが、日本語が全く出来ない私は一度も読んだことはないです。ですので、LSTMによる文章が自然なものなのか、宮沢賢治風なのか、不明です。。

銀河鉄道の夜は以前、word2vecを利用した[分散表現の作成の記事](/ml/lec/text/w2v/)で利用しました。
テキストの前処理などは重複する部分があるかと思います。

まずはテキストの中身を見てみます。


```bash
%%bash
cat ginga.txt | head -n 25
```

    銀河鉄道の夜
    宮沢賢治
    
    -------------------------------------------------------
    【テキスト中に現れる記号について】
    
    《》：ルビ
    （例）北十字《きたじふじ》
    
    ［＃］：入力者注　主に外字の説明や、傍点の位置の指定
    　　　（数字は、JIS X 0213の面区点番号またはUnicode、底本のページと行数）
    （例）※［＃小書き片仮名ヰ、155-15］
    
    　［＃（…）］：訓点送り仮名
    　（例）僕［＃（ん）］とこ
    -------------------------------------------------------
    
    ［＃７字下げ］一、午后の授業［＃「一、午后の授業」は中見出し］
    
    「ではみなさんは、さういふふうに川だと云はれたり、乳の流れたあとだと云はれたりしてゐたこのぼんやりと白いものがほんたうは何かご承知ですか。」先生は、黒板に吊した大きな黒い星座の図の、上から下へ白くけぶった銀河帯のやうなところを指しながら、みんなに問をかけました。
    カムパネルラが手をあげました。それから四五人手をあげました。ジョバンニも手をあげやうとして、急いでそのまゝやめました。たしかにあれがみんな星だと、いつか雑誌で読んだのでしたが、このごろはジョバンニはまるで毎日教室でもねむく、本を読むひまも読む本もないので、なんだかどんなこともよくわからないといふ気持ちがするのでした。
    ところが先生は早くもそれを見附けたのでした。
    「ジョバンニさん。あなたはわかってゐるのでせう。」
    ジョバンニは勢よく立ちあがりましたが、立って見るともうはっきりとそれを答へることができないのでした。ザネリが前の席からふりかへって、ジョバンニを見てくすっとわらひました。ジョバンニはもうどぎまぎしてまっ赤になってしまひました。先生がまた云ひました。
    「大きな望遠鏡で銀河をよっく調べると銀河は大体何でせう。」



```bash
%%bash
cat ginga.txt | tail -n 25
```

    ジョバンニはそのカムパネルラはもうあの銀河のはづれにしかゐないといふやうな気がしてしかたなかったのです。
    けれどもみんなはまだ、どこかの波の間から、
    「ぼくずゐぶん泳いだぞ。」と云ひながらカムパネルラが出て来るか或ひはカムパネルラがどこかの人の知らない洲にでも着いて立ってゐて誰かの来るのを待ってゐるかといふやうな気がして仕方ないらしいのでした。けれども俄かにカムパネルラのお父さんがきっぱり云ひました。
    「もう駄目です。落ちてから四十五分たちましたから。」
    ジョバンニは思はずか〔け〕よって博士の前に立って、ぼくはカムパネルラの行った方を知ってゐますぼくはカムパネルラといっしょに歩いてゐたのですと云はうとしましたがもうのどがつまって何とも云へませんでした。すると博士はジョバンニが挨拶に来たとでも思ったものですか　しばらくしげしげジョバンニを見てゐましたが
    「あなたはジョバンニさんでしたね。どうも今晩はありがたう。」と叮ねいに云ひました。
    　ジョバンニは何も云へずにたゞおじぎをしました。
    「あなたのお父さんはもう帰ってゐますか。」博士は堅く時計を握ったまゝまたきゝました。
    「いゝえ。」ジョバンニはかすかに頭をふりました。
    「どうしたのかなあ、ぼくには一昨日大へん元気な便りがあったんだが。今日あ〔〕たりもう着くころなんだが。船が遅れたんだな。ジョバンニさん。あした放課后みなさんとうちへ遊びに来てくださいね。」
    さう云ひながら博士は〔〕また川下の銀河のいっぱいにうつった方へじっと眼を送りました。ジョバンニはもういろいろなことで胸がいっぱいでなんにも云へずに博士の前をはなれて早くお母さんに牛乳を持って行ってお父さんの帰ることを知らせやうと思ふともう一目散に河原を街の方へ走りました。
    
    
    
    底本：「【新】校本宮澤賢治全集　第十一巻　童話※［＃ローマ数字4、1-13-24］　本文篇」筑摩書房
    　　　1996（平成8）年1月25日初版第1刷発行
    ※底本のテキストは、著者草稿によります。
    ※底本では校訂及び編者による説明を「〔　〕」、削除を「〔〕」で表示しています。
    ※「カムパネルラ」と「カンパネルラ」の混在は、底本通りです。
    ※底本は新字旧仮名づかいです。なお拗音、促音の小書きは、底本通りです。
    入力：砂場清隆
    校正：北川松生
    2016年6月10日作成
    青空文庫作成ファイル：
    このファイルは、インターネットの図書館、青空文庫（http://www.aozora.gr.jp/）で作られました。入力、校正、制作にあたったのは、ボランティアの皆さんです。


となり、ファイルの先頭と、末尾に参考情報が載っているほかは、ちゃんとテキストとしてデータが取れている模様です。
先ず、この辺の前処理を行います。


```python
import re

with open('ginga.txt', mode='r') as f:
  all_sentence = f.read()
```


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
all_sentence[:100]
```




    '［＃７字下げ］一、午后の授業［＃「一、午后の授業」は中見出し］「ではみなさんは、さういふふうに川だと云はれたり、乳の流れたあとだと云はれたりしてゐたこのぼんやりと白いものがほんたうは何かご承知ですか。'



となり、不要な部分を削除し、必要な部分をall_sentenceに格納しました。

## one hot vectorの作成

文章を学習させるには、日本語の文字1文字1文字をベクトルとして表現する必要があります。前回やったとおりword2vecを用いてベクトル表現を得る方法もありますが、ここでは、それぞれの文字に対して、`[0,0,1,0,0]`などのone-hot-vectorを付与します。ですので、ベクトルの次元数としては、文字数分だけあり、学習にかなりの時間を要します。

まず、銀河鉄道の夜で利用されている文字をすべて取り出します。


```python
all_chars = sorted(list(set(all_sentence)))
all_chars[:10]
```




    ['-', '.', '/', '0', '1', '2', '3', '4', '5', '6']



次に、文字に対して数字を対応させます。上記の`all_chars`に格納された順番の数字を付与します。


```python
char_num_dic = dict((c, i) for i, c in enumerate(all_chars))
num_char_dic = dict((i, c) for i, c in enumerate(all_chars))
```

後の処理を簡単にするために、文字列を受け取って、対応する数字のリストを返す関数を作成します。


```python
def get_scalar_list(char_list):
  return [char_num_dic[c] for c in char_list]
```

この関数を利用し、予想に利用する文字列と予想する文字を数字のリストに変換します。

また、LSTMで予測するのに必要な時系列データの数を100とします。
100個の文字列から、次の1文字を予測するモデルを作成します。


```python
NUM_LSTM = 100

train_chars_list = []
predict_char_list = []
for c in range(0, len(all_sentence) - NUM_LSTM):
  train_chars_list.append(get_scalar_list(all_sentence[c: c + NUM_LSTM]))
  predict_char_list.append(char_num_dic[all_sentence[c + NUM_LSTM]])
```


```python
print(train_chars_list[0])
```

    [1117, 1110, 1114, 403, 190, 50, 1118, 183, 26, 295, 327, 78, 538, 628, 1117, 1110, 29, 183, 26, 295, 327, 78, 538, 628, 30, 79, 195, 942, 261, 55, 1118, 29, 71, 79, 94, 74, 53, 112, 79, 26, 53, 39, 38, 85, 85, 39, 75, 440, 64, 72, 204, 79, 107, 63, 105, 26, 201, 78, 682, 107, 63, 36, 72, 64, 72, 204, 79, 107, 63, 105, 55, 70, 110, 63, 51, 78, 91, 112, 99, 105, 72, 769, 38, 97, 78, 44, 90, 112, 63, 39, 79, 225, 43, 52, 519, 790, 71, 57, 43, 27]



```python
print(predict_char_list[0])
```

    30


train_chars[0]からpredict_char[0]を予測するようなモデルを作成します。

これらの数字をone hot vectorで表現します。

表現するベクトルのサイズは`len(all_chars)`となります。また、kerasに投入することを前提に、入力するテンソルの形状として

`(サンプル数、予測に利用する時系列データの数、one-hot-vectorの次元)`となります。


```python
# xを入力するデータ
# yを正解データ
# one-hot-vectorを入力するため、最初にゼロベクトルを作成します。

x = np.zeros((len(train_chars_list), NUM_LSTM, len(all_chars)), dtype=np.bool)
y = np.zeros((len(predict_char_list), len(all_chars)), dtype=np.bool)
```

必要な部分だけ1に修正します。


```python
# 入力データに割り当てられた数字の要素を1に設定します。
for i in range(len(train_chars_list)):
  for j in range(NUM_LSTM):
    x[i, j, train_chars_list[i][j]] = 1

# 正解データに割り当てられた数字の要素を1に設定します。
for i in range(len(predict_char_list)):
  y[i, predict_char_list[i]] = 1
```

## one-hot-vectorの確認

実際に想定通りone-hot-vectorが出来ているか確認してみます。`np.where`を利用してtrueとなっているインデックスを取得してみます。


```python
np.where(x[0][:-1] == 1)[1]
```




    array([1117, 1110, 1114,  403,  190,   50, 1118,  183,   26,  295,  327,
             78,  538,  628, 1117, 1110,   29,  183,   26,  295,  327,   78,
            538,  628,   30,   79,  195,  942,  261,   55, 1118,   29,   71,
             79,   94,   74,   53,  112,   79,   26,   53,   39,   38,   85,
             85,   39,   75,  440,   64,   72,  204,   79,  107,   63,  105,
             26,  201,   78,  682,  107,   63,   36,   72,   64,   72,  204,
             79,  107,   63,  105,   55,   70,  110,   63,   51,   78,   91,
            112,   99,  105,   72,  769,   38,   97,   78,   44,   90,  112,
             63,   39,   79,  225,   43,   52,  519,  790,   71,   57,   43])




```python
np.where(y[0] == 1)
```




    array([30])



となり、想定通りone-hot-vectorが出来ていることがわかりました。

## モデルの構築

LSTMのモデルを構築する関数を作成します。
ここでは簡単にLSTMと全結合層で構成されたモデルを作成します。


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

NUM_MIDDLE = 300

def build_lstm_model():
  lstm_model = Sequential()
  lstm_model.add(LSTM(NUM_MIDDLE, input_shape=(NUM_LSTM, len(all_chars))))
  lstm_model.add(Dense(len(all_chars), activation="softmax"))
  lstm_model.compile(loss='categorical_crossentropy', optimizer="adam")
  
  print(lstm_model.summary())
  
  return lstm_model

model = build_lstm_model()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 300)               1704000   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1119)              336819    
    =================================================================
    Total params: 2,040,819
    Trainable params: 2,040,819
    Non-trainable params: 0
    _________________________________________________________________
    None



```python

```

epoch終了後に実行させるコールバック関数を実行させます


```python
from tensorflow.keras.callbacks import LambdaCallback
 
def on_epoch_end(epoch, logs):
  print("エポック: ", epoch)

  beta = 5  # 確率分布を調整する定数
  prev_text = text[0: NUM_LSTM]  # 入力に使う文字
  created_text = prev_text  # 生成されるテキスト
  
  print("シード: ", created_text)

  for i in range(400):
    # 入力をone-hot表現に
    x_pred = np.zeros((1, NUM_LSTM, len(all_chars)))
    for j, char in enumerate(prev_text):
      x_pred[0, j, char_indices[char]] = 1
    
    # 予測を行い、次の文字を得る
    y = model.predict(x_pred)
    p_power = y[0] ** beta  # 確率分布の調整
    next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))    
    next_char = indices_char[next_index]

    created_text += next_char
    prev_text = prev_text[1:] + next_char

  print(created_text)

# エポック終了後に実行される関数を設定
epoch_end_callback= LambdaCallback(on_epoch_end=on_epoch_end)
```


```python
## とても時間がかかる

epochs = 10
batch_size = 100

history = model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[epoch_end_callback])
```

    Epoch 1/10
     34/391 [=>............................] - ETA: 15:11 - loss: 6.2912


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-25-837be0bb79ea> in <module>
          2 batch_size = 100
          3 
    ----> 4 history = model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[epoch_end_callback])
    

    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1098                 _r=1):
       1099               callbacks.on_train_batch_begin(step)
    -> 1100               tmp_logs = self.train_function(iterator)
       1101               if data_handler.should_sync:
       1102                 context.async_wait()


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        826     tracing_count = self.experimental_get_tracing_count()
        827     with trace.Trace(self._name) as tm:
    --> 828       result = self._call(*args, **kwds)
        829       compiler = "xla" if self._experimental_compile else "nonXla"
        830       new_tracing_count = self.experimental_get_tracing_count()


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        853       # In this case we have created variables on the first call, so we run the
        854       # defunned version which is guaranteed to never create variables.
    --> 855       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        856     elif self._stateful_fn is not None:
        857       # Release the lock early so that multiple threads can perform the call


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py in __call__(self, *args, **kwargs)
       2941        filtered_flat_args) = self._maybe_define_function(args, kwargs)
       2942     return graph_function._call_flat(
    -> 2943         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
       2944 
       2945   @property


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1917       # No tape is watching; skip to running the function.
       1918       return self._build_call_outputs(self._inference_function.call(
    -> 1919           ctx, args, cancellation_manager=cancellation_manager))
       1920     forward_backward = self._select_forward_and_backward_functions(
       1921         args,


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        558               inputs=args,
        559               attrs=attrs,
    --> 560               ctx=ctx)
        561         else:
        562           outputs = execute.execute_with_cancellation(


    ~/anaconda3/lib/python3.7/site-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         58     ctx.ensure_initialized()
         59     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
    ---> 60                                         inputs, attrs, num_outputs)
         61   except core._NotOkStatusException as e:
         62     if name is not None:


    KeyboardInterrupt: 



```python

```


```python

```


```python

```
