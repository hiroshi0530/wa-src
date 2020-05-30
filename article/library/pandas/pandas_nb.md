
## pandasとデータ分析
pandasはデータ分析では必ず利用する重要なツールです。この使い方を知るか知らないか、もしくは、やりたいことをグーグル検索しなくてもすぐに手を動かせるかどうかは、エンジニアとしての力量に直結します。ここでは、具体的なデータを元に私の経験から重要と思われるメソッドや使い方を説明します。他に重要な使い方に遭遇したらどんどん追記していきます。

また、jupyter形式のファイルは[github](https://github.com/hiroshi0530/wa/blob/master/src/pandas/pandas_nb.ipynb)に置いておきます。

## 頻出のコマンド一覧
概要として、よく利用するコマンドを以下に載せます。
概要として、よく利用するコマンドを以下に載せます。

### ファイルのIO

#### CSVファイルの読み込み
```python
df.read_csv()
```

#### EXCELファイルの読み込み
```python
df.read_excel()
```

#### 先頭の5行を表示
```python
df.head()
```

#### 最後の5行を表示
```python
df.tail()
```

#### インデックスの確認
```python
df.index
```

#### サイズの確認
```python
df.shape
```

#### カラム名の確認
```python
df.columns
```

#### データ形式の確認
```python
df.dtypes
```

#### 
```python
df.loc[]
```

#### 
```python
df.iloc[]
```

#### 
```python
df.query()
```

#### 
```python
df.unique()
```

#### 
```python
df.drop_duplicates()
```

#### 
```python
df.describe()
```

#### 
```python
df.set_index()
```

#### 
```python
df.rename()
```

#### 
```python
df.sort_values()
```

#### 
```python
df.to_datetime()
```

#### 
```python
df.sort_index()
```

#### 
```python
df.apply()
```

#### 
```python
pd.cut()
```

#### 
```python
df.isnull()
```

#### 
```python
df.any()
```

#### 
```python
df.fillna()
```

#### 
```python
df.dropna()
```

#### 
```python
df.replace()
```

#### 
```python
df.mask()
```

#### 
```python
df.drop()
```

#### 
```python
df.value_counts()
```

#### 
```python
df.groupby()
```

#### 
```python
df.diff()
```

#### 
```python
df.rolling()
```

#### 
```python
df.pct_change()
```

#### 
```python
df.plot()
```

#### 
```python
df.pivot()
```

#### 
```python
pd.get_dummies()
```

#### 
```python
df.to_csv()
```

#### 
```python
pd.options.display.max_columns = None
```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```

#### 
```python

```



## 具体例

以下実際のデータを用いて、上記のコマンドの利用例を説明します。

### 環境
最初に、私の実行環境のOSとterminalの環境です。


```python
!sw_vers
```


```python
!uname -a | awk '{c="";for(i=3;i<=NF;i++) c=c $i" "; print c}'
```

    18.7.0 Darwin Kernel Version 18.7.0: Tue Aug 20 16:57:14 PDT 2019; root:xnu-4903.271.2~2/RELEASE_X86_64 x86_64 


### importとバージョン確認


```python
import pandas as pd

pd.__version__
```




    '0.23.1'



### データの読み込み
データの例として、Googleのtensorflowのページでも利用されている[Auto MPG](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/) のデータセットを利用します。wgetでデータをダウンロードします。-O オプションで上書きします。


```python
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data -O ./auto-mpg.data   
```

    --2020-04-16 00:10:12--  https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
    archive.ics.uci.edu (archive.ics.uci.edu) をDNSに問いあわせています... 128.195.10.252
    archive.ics.uci.edu (archive.ics.uci.edu)|128.195.10.252|:443 に接続しています... 接続しました。
    HTTP による接続要求を送信しました、応答を待っています... 200 OK
    長さ: 30286 (30K) [application/x-httpd-php]
    `./auto-mpg.data' に保存中
    
    ./auto-mpg.data     100%[===================>]  29.58K  --.-KB/s 時間 0.1s       
    
    2020-04-16 00:10:13 (198 KB/s) - `./auto-mpg.data' へ保存完了 [30286/30286]
    


データ属性は[本家のホームページ](https://archive.ics.uci.edu/ml/datasets/auto+mpg)によると、

1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)

となっています。詳細は[本家のホームページ](https://archive.ics.uci.edu/ml/datasets/auto+mpg)を参照してください。

データの概要を見てみます。


```python
!head -n 5 auto-mpg.data
```

    18.0   8   307.0      130.0      3504.      12.0   70  1	"chevrolet chevelle malibu"
    15.0   8   350.0      165.0      3693.      11.5   70  1	"buick skylark 320"
    18.0   8   318.0      150.0      3436.      11.0   70  1	"plymouth satellite"
    16.0   8   304.0      150.0      3433.      12.0   70  1	"amc rebel sst"
    17.0   8   302.0      140.0      3449.      10.5   70  1	"ford torino"


9個のカラムがあります。また、データの区切り形式を確認するため、タブを可視化するコマンドを実行します。catのtオプションになります。私の環境はmacOSですので、linux環境の方はmanで調べてください。


```python
!head -n 5 auto-mpg.data | cat -evt
```

    18.0   8   307.0      130.0      3504.      12.0   70  1^I"chevrolet chevelle malibu"$
    15.0   8   350.0      165.0      3693.      11.5   70  1^I"buick skylark 320"$
    18.0   8   318.0      150.0      3436.      11.0   70  1^I"plymouth satellite"$
    16.0   8   304.0      150.0      3433.      12.0   70  1^I"amc rebel sst"$
    17.0   8   302.0      140.0      3449.      10.5   70  1^I"ford torino"$


これより、最後のカラムの前にタブがあるのがわかります。少々わかりにくいですが、^I がタブの目印になります。

これだと、区切り文字が空白とタブが混在しているので、タブを空白に置換します。出来れば、sedでタブを置換したいのですが、sedの挙動がmacOSとlinuxで異なるので、やや冗長ですが、一度中間ファイルを作成します。実際のタブの置換はtrを利用します。


```python
!cat auto-mpg.data | tr '\t' ' ' >> temp.data
!mv temp.data auto-mpg.data && rm -f temp.data
!head -n 5 auto-mpg.data | cat -evt
```

    18.0   8   307.0      130.0      3504.      12.0   70  1 "chevrolet chevelle malibu"$
    15.0   8   350.0      165.0      3693.      11.5   70  1 "buick skylark 320"$
    18.0   8   318.0      150.0      3436.      11.0   70  1 "plymouth satellite"$
    16.0   8   304.0      150.0      3433.      12.0   70  1 "amc rebel sst"$
    17.0   8   302.0      140.0      3449.      10.5   70  1 "ford torino"$


最後のコマンドでタブの有無を確認すると、確かにタブが消えています。ここでようやく準備が整いました。このファイルをpandasを用いて読み込みます。その際、column名を指定します。

ここまでの流れは面倒と感じるかもしれませんが、データ分析の仕事をしているとデータ分析コンテストのように整然としたデータがそろっていることの方が珍しいです。データを整える前処理も重要な仕事です。それらにはlinuxのコマンドを使いこなすことが重要です。


```python
column_names = ['mpg','cylinders','displacement','horsepower','weight',
                'acceleration', 'model year', 'origin', 'car name'] 

df = pd.read_csv('./auto-mpg.data', 
                 names=column_names,
                 sep=' ',
                 skipinitialspace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import seaborn as sns

iris = sns.load_dataset("iris")
sns.pairplot(iris)
```




    <seaborn.axisgrid.PairGrid at 0x11655a0f0>




![svg](pandas_nb_files/pandas_nb_16_1.svg)



```python
sns.pairplot(iris, hue="species")
```




    <seaborn.axisgrid.PairGrid at 0x125a9f2e8>




![svg](pandas_nb_files/pandas_nb_17_1.svg)


## よく使う関数

最後のまとめとして、良く使う関数をまとめておきます。

#### インデックスの変更(既存のカラム名に変更)

```python
df.set_index('xxxx')
```

#### カラム名の変更

```python
df.rename(columns={'before': 'after'}, inplace=True)
```

#### あるカラムでソートする

```python
df.sort_values(by='xxx', ascending=True)
```

#### インデックスでソートする

```python
df.sort_index()
```

#### datetime型の型変換
```python
df.to_datetime()
```

#### NaNのカラムごとの個数
```python
df.isnull().sum()
```




## 参考文献
- [チートシート](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
- [read_csvの全引数について解説してくれてます](https://own-search-and-study.xyz/2015/09/03/pandas%E3%81%AEread_csv%E3%81%AE%E5%85%A8%E5%BC%95%E6%95%B0%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99/)
