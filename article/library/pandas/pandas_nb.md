
## pandasとデータ分析
pandasはデータ分析では必ず利用する重要なツールです。この使い方を知るか知らないか、もしくは、やりたいことをグーグル検索しなくてもすぐに手を動かせるかどうかは、エンジニアとしての力量に直結します。ここでは、具体的なデータを元に私の経験から重要と思われるメソッドや使い方を説明します。他に重要な使い方に遭遇したらどんどん追記していきます。


### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/pandas/pandas_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/pandas/pandas_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G95



```python
!python -V
```

    Python 3.5.5 :: Anaconda, Inc.


基本的なライブラリをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
```

    matplotlib version : 2.2.2
    scipy version : 1.4.1
    numpy version : 1.18.1


### importとバージョン確認


```python
import pandas as pd

print('pandas version :', pd.__version__)
```

    pandas version : 0.24.2


## 基本操作

### データの読み込みと表示

利用させてもらうデータは[danielさんのgithub](https://github.com/chendaniely/pandas_for_everyone)になります。pandasの使い方の本を書いておられる有名な方のリポジトリです。[Pythonデータ分析／機械学習のための基本コーディング！ pandasライブラリ活用入門](https://www.amazon.co.jp/dp/B07NZP6V29/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)です。僕も持っています。とても勉強になると思います。

データはエボラ出血の発生数(Case)と死者数(Death)だと思います。

`read_csv`を利用して、CSVを読み込み、先頭の5行目を表示してみます。


```python
import pandas as pd

df = pd.read_csv('./country_timeseries.csv', sep=',')
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
      <th>Date</th>
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/5/2015</td>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/4/2015</td>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2015</td>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/2/2015</td>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12/31/2014</td>
      <td>284</td>
      <td>2730.0</td>
      <td>8115.0</td>
      <td>9633.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1739.0</td>
      <td>3471.0</td>
      <td>2827.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



末尾の5データを表示します。


```python
df.tail()
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
      <th>Date</th>
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>3/27/2014</td>
      <td>5</td>
      <td>103.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>118</th>
      <td>3/26/2014</td>
      <td>4</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119</th>
      <td>3/25/2014</td>
      <td>3</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>120</th>
      <td>3/24/2014</td>
      <td>2</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>59.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>121</th>
      <td>3/22/2014</td>
      <td>0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### データの確認

#### データの型などの情報を取得


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 122 entries, 0 to 121
    Data columns (total 18 columns):
    Date                   122 non-null object
    Day                    122 non-null int64
    Cases_Guinea           93 non-null float64
    Cases_Liberia          83 non-null float64
    Cases_SierraLeone      87 non-null float64
    Cases_Nigeria          38 non-null float64
    Cases_Senegal          25 non-null float64
    Cases_UnitedStates     18 non-null float64
    Cases_Spain            16 non-null float64
    Cases_Mali             12 non-null float64
    Deaths_Guinea          92 non-null float64
    Deaths_Liberia         81 non-null float64
    Deaths_SierraLeone     87 non-null float64
    Deaths_Nigeria         38 non-null float64
    Deaths_Senegal         22 non-null float64
    Deaths_UnitedStates    18 non-null float64
    Deaths_Spain           16 non-null float64
    Deaths_Mali            12 non-null float64
    dtypes: float64(16), int64(1), object(1)
    memory usage: 17.2+ KB


#### 大きさ（行数と列数）の確認


```python
df.shape
```




    (122, 18)



#### インデックスの確認


```python
df.index
```




    RangeIndex(start=0, stop=122, step=1)



#### カラム名の確認


```python
df.columns
```




    Index(['Date', 'Day', 'Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone',
           'Cases_Nigeria', 'Cases_Senegal', 'Cases_UnitedStates', 'Cases_Spain',
           'Cases_Mali', 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone',
           'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates',
           'Deaths_Spain', 'Deaths_Mali'],
          dtype='object')



#### 任意の列名のデータの取得

カラム名を指定して、任意のカラムだけ表示させます。


```python
df[['Cases_UnitedStates','Deaths_UnitedStates']].head()
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
      <th>Cases_UnitedStates</th>
      <th>Deaths_UnitedStates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 行数や列数を指定してデータを取得


```python
df.iloc[[6,7],[0,3]]
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
      <th>Date</th>
      <th>Cases_Liberia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>12/27/2014</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12/24/2014</td>
      <td>7977.0</td>
    </tr>
  </tbody>
</table>
</div>



#### ある条件を満たしたデータを取得


```python
df[df['Deaths_Liberia'] > 3000][['Deaths_Liberia']]
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
      <th>Deaths_Liberia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>3496.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3496.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3471.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3423.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3413.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3384.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3376.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3290.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3177.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3145.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3016.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 統計量の取得

describe()を利用して、列ごとの統計量を取得することが出来ます。ぱっと見、概要を得たいときに有力です。


```python
df.describe()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>122.000000</td>
      <td>93.000000</td>
      <td>83.000000</td>
      <td>87.000000</td>
      <td>38.000000</td>
      <td>25.00</td>
      <td>18.000000</td>
      <td>16.0</td>
      <td>12.000000</td>
      <td>92.000000</td>
      <td>81.000000</td>
      <td>87.000000</td>
      <td>38.000000</td>
      <td>22.0</td>
      <td>18.000000</td>
      <td>16.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>144.778689</td>
      <td>911.064516</td>
      <td>2335.337349</td>
      <td>2427.367816</td>
      <td>16.736842</td>
      <td>1.08</td>
      <td>3.277778</td>
      <td>1.0</td>
      <td>3.500000</td>
      <td>563.239130</td>
      <td>1101.209877</td>
      <td>693.701149</td>
      <td>6.131579</td>
      <td>0.0</td>
      <td>0.833333</td>
      <td>0.187500</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>89.316460</td>
      <td>849.108801</td>
      <td>2987.966721</td>
      <td>3184.803996</td>
      <td>5.998577</td>
      <td>0.40</td>
      <td>1.178511</td>
      <td>0.0</td>
      <td>2.746899</td>
      <td>508.511345</td>
      <td>1297.208568</td>
      <td>869.947073</td>
      <td>2.781901</td>
      <td>0.0</td>
      <td>0.383482</td>
      <td>0.403113</td>
      <td>2.405801</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>66.250000</td>
      <td>236.000000</td>
      <td>25.500000</td>
      <td>64.500000</td>
      <td>15.000000</td>
      <td>1.00</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>157.750000</td>
      <td>12.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>150.000000</td>
      <td>495.000000</td>
      <td>516.000000</td>
      <td>783.000000</td>
      <td>20.000000</td>
      <td>1.00</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>2.500000</td>
      <td>360.500000</td>
      <td>294.000000</td>
      <td>334.000000</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>219.500000</td>
      <td>1519.000000</td>
      <td>4162.500000</td>
      <td>3801.000000</td>
      <td>20.000000</td>
      <td>1.00</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>6.250000</td>
      <td>847.750000</td>
      <td>2413.000000</td>
      <td>1176.000000</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>289.000000</td>
      <td>2776.000000</td>
      <td>8166.000000</td>
      <td>10030.000000</td>
      <td>22.000000</td>
      <td>3.00</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>7.000000</td>
      <td>1786.000000</td>
      <td>3496.000000</td>
      <td>2977.000000</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



## インデックスをdatetime型に変更

インデックスをDateに変更し、上書きします。


```python
df.set_index('Date', inplace=True)
df.index
```




    Index(['1/5/2015', '1/4/2015', '1/3/2015', '1/2/2015', '12/31/2014',
           '12/28/2014', '12/27/2014', '12/24/2014', '12/21/2014', '12/20/2014',
           ...
           '4/4/2014', '4/1/2014', '3/31/2014', '3/29/2014', '3/28/2014',
           '3/27/2014', '3/26/2014', '3/25/2014', '3/24/2014', '3/22/2014'],
          dtype='object', name='Date', length=122)




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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1/5/2015</th>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/4/2015</th>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/3/2015</th>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/2/2015</th>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12/31/2014</th>
      <td>284</td>
      <td>2730.0</td>
      <td>8115.0</td>
      <td>9633.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1739.0</td>
      <td>3471.0</td>
      <td>2827.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



ついでにDateというインデックス名も変更します。rename関数を利用します。


```python
df.rename(index={'Date':'YYYYMMDD'}, inplace=True)
```


```python
df.columns
# df.sort_values(by="YYYYMMDD", ascending=True).head()
```




    Index(['Day', 'Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone',
           'Cases_Nigeria', 'Cases_Senegal', 'Cases_UnitedStates', 'Cases_Spain',
           'Cases_Mali', 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone',
           'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates',
           'Deaths_Spain', 'Deaths_Mali'],
          dtype='object')



インデックスでソートします。ただ、日付が文字列のオブジェクトになっているので、目論見通りのソートになっていません。


```python
df.sort_index(ascending=True).head()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1/2/2015</th>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/3/2015</th>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/4/2015</th>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1/5/2015</th>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10/1/2014</th>
      <td>193</td>
      <td>1199.0</td>
      <td>3834.0</td>
      <td>2437.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>739.0</td>
      <td>2069.0</td>
      <td>623.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



インデックスをdatetime型に変更します。


```python
df.index
```




    Index(['1/5/2015', '1/4/2015', '1/3/2015', '1/2/2015', '12/31/2014',
           '12/28/2014', '12/27/2014', '12/24/2014', '12/21/2014', '12/20/2014',
           ...
           '4/4/2014', '4/1/2014', '3/31/2014', '3/29/2014', '3/28/2014',
           '3/27/2014', '3/26/2014', '3/25/2014', '3/24/2014', '3/22/2014'],
          dtype='object', name='Date', length=122)




```python
df.index = pd.to_datetime(df.index, format='%m/%d/%Y')
df.index
```




    DatetimeIndex(['2015-01-05', '2015-01-04', '2015-01-03', '2015-01-02',
                   '2014-12-31', '2014-12-28', '2014-12-27', '2014-12-24',
                   '2014-12-21', '2014-12-20',
                   ...
                   '2014-04-04', '2014-04-01', '2014-03-31', '2014-03-29',
                   '2014-03-28', '2014-03-27', '2014-03-26', '2014-03-25',
                   '2014-03-24', '2014-03-22'],
                  dtype='datetime64[ns]', name='Date', length=122, freq=None)



となり、dtype='object'からobject='datetime64'とdatetime型に変更されていることが分かります。そこでソートしてみます。


```python
df.sort_index(ascending=True).head(10)
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-03-22</th>
      <td>0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-24</th>
      <td>2</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>59.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-25</th>
      <td>3</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-26</th>
      <td>4</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-27</th>
      <td>5</td>
      <td>103.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-28</th>
      <td>6</td>
      <td>112.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-29</th>
      <td>7</td>
      <td>112.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-03-31</th>
      <td>9</td>
      <td>122.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-04-01</th>
      <td>10</td>
      <td>127.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>83.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-04-04</th>
      <td>13</td>
      <td>143.0</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(ascending=True).tail(10)
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-20</th>
      <td>272</td>
      <td>2571.0</td>
      <td>7862.0</td>
      <td>8939.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586.0</td>
      <td>3384.0</td>
      <td>2556.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-21</th>
      <td>273</td>
      <td>2597.0</td>
      <td>NaN</td>
      <td>9004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1607.0</td>
      <td>NaN</td>
      <td>2582.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-24</th>
      <td>277</td>
      <td>2630.0</td>
      <td>7977.0</td>
      <td>9203.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3413.0</td>
      <td>2655.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-27</th>
      <td>280</td>
      <td>2695.0</td>
      <td>NaN</td>
      <td>9409.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1697.0</td>
      <td>NaN</td>
      <td>2732.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-28</th>
      <td>281</td>
      <td>2706.0</td>
      <td>8018.0</td>
      <td>9446.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1708.0</td>
      <td>3423.0</td>
      <td>2758.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>284</td>
      <td>2730.0</td>
      <td>8115.0</td>
      <td>9633.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1739.0</td>
      <td>3471.0</td>
      <td>2827.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-03</th>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-04</th>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



となり、想定通りのソートになっている事が分かります。

また、datetime型がインデックスに設定されたので、日付を扱いのが容易になっています。
例えば、2015年のデータを取得するのに、以下の様になります。


```python
df['2015']
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-05</th>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-04</th>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-03</th>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['2014-12'].sort_index(ascending=True)
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-03</th>
      <td>256</td>
      <td>NaN</td>
      <td>7719.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3177.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-07</th>
      <td>260</td>
      <td>2292.0</td>
      <td>NaN</td>
      <td>7897.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1428.0</td>
      <td>NaN</td>
      <td>1768.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2014-12-09</th>
      <td>262</td>
      <td>NaN</td>
      <td>7797.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3290.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-14</th>
      <td>267</td>
      <td>2416.0</td>
      <td>NaN</td>
      <td>8356.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1525.0</td>
      <td>NaN</td>
      <td>2085.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-18</th>
      <td>271</td>
      <td>NaN</td>
      <td>7830.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3376.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-20</th>
      <td>272</td>
      <td>2571.0</td>
      <td>7862.0</td>
      <td>8939.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586.0</td>
      <td>3384.0</td>
      <td>2556.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-21</th>
      <td>273</td>
      <td>2597.0</td>
      <td>NaN</td>
      <td>9004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1607.0</td>
      <td>NaN</td>
      <td>2582.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-24</th>
      <td>277</td>
      <td>2630.0</td>
      <td>7977.0</td>
      <td>9203.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3413.0</td>
      <td>2655.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-27</th>
      <td>280</td>
      <td>2695.0</td>
      <td>NaN</td>
      <td>9409.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1697.0</td>
      <td>NaN</td>
      <td>2732.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-28</th>
      <td>281</td>
      <td>2706.0</td>
      <td>8018.0</td>
      <td>9446.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1708.0</td>
      <td>3423.0</td>
      <td>2758.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>284</td>
      <td>2730.0</td>
      <td>8115.0</td>
      <td>9633.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1739.0</td>
      <td>3471.0</td>
      <td>2827.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



さらに、平均や合計値などの統計値を、年や月単位で簡単に取得することができます。


```python
df.resample('Y').mean()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-31</th>
      <td>139.940678</td>
      <td>848.988889</td>
      <td>2191.481481</td>
      <td>2162.488095</td>
      <td>16.736842</td>
      <td>1.08</td>
      <td>3.277778</td>
      <td>1.0</td>
      <td>3.5</td>
      <td>522.292135</td>
      <td>1040.582278</td>
      <td>613.297619</td>
      <td>6.131579</td>
      <td>0.0</td>
      <td>0.833333</td>
      <td>0.1875</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>287.500000</td>
      <td>2773.333333</td>
      <td>8161.500000</td>
      <td>9844.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1778.000000</td>
      <td>3496.000000</td>
      <td>2945.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.resample('M').mean()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-03-31</th>
      <td>4.500000</td>
      <td>94.500000</td>
      <td>6.500000</td>
      <td>3.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.000000</td>
      <td>3.750000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-04-30</th>
      <td>24.333333</td>
      <td>177.818182</td>
      <td>24.555556</td>
      <td>2.200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>113.636364</td>
      <td>9.625000</td>
      <td>1.111111</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-05-31</th>
      <td>51.888889</td>
      <td>248.777778</td>
      <td>12.555556</td>
      <td>7.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>166.666667</td>
      <td>11.111111</td>
      <td>1.222222</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-06-30</th>
      <td>84.636364</td>
      <td>373.428571</td>
      <td>35.500000</td>
      <td>125.571429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>250.428571</td>
      <td>28.000000</td>
      <td>29.375000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-07-31</th>
      <td>115.700000</td>
      <td>423.000000</td>
      <td>212.300000</td>
      <td>420.500000</td>
      <td>1.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>316.300000</td>
      <td>121.300000</td>
      <td>189.500000</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-08-31</th>
      <td>145.090909</td>
      <td>559.818182</td>
      <td>868.818182</td>
      <td>844.000000</td>
      <td>13.363636</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>394.363636</td>
      <td>468.454545</td>
      <td>353.000000</td>
      <td>3.545455</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-09-30</th>
      <td>177.500000</td>
      <td>967.888889</td>
      <td>2815.625000</td>
      <td>1726.000000</td>
      <td>20.714286</td>
      <td>1.285714</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>607.000000</td>
      <td>1508.000000</td>
      <td>565.777778</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-10-31</th>
      <td>207.470588</td>
      <td>1500.444444</td>
      <td>4758.750000</td>
      <td>3668.111111</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>2.555556</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>870.555556</td>
      <td>2419.000000</td>
      <td>1151.666667</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>0.666667</td>
      <td>0.428571</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>2014-11-30</th>
      <td>237.214286</td>
      <td>1950.500000</td>
      <td>7039.000000</td>
      <td>5843.625000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1174.500000</td>
      <td>2928.857143</td>
      <td>1256.750000</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.625</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>271.181818</td>
      <td>2579.625000</td>
      <td>7902.571429</td>
      <td>8985.875000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1612.857143</td>
      <td>3362.000000</td>
      <td>2495.375000</td>
      <td>8.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000</td>
    </tr>
    <tr>
      <th>2015-01-31</th>
      <td>287.500000</td>
      <td>2773.333333</td>
      <td>8161.500000</td>
      <td>9844.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1778.000000</td>
      <td>3496.000000</td>
      <td>2945.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.resample('Y').sum()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-31</th>
      <td>16513</td>
      <td>76409.0</td>
      <td>177510.0</td>
      <td>181649.0</td>
      <td>636.0</td>
      <td>27.0</td>
      <td>59.0</td>
      <td>16.0</td>
      <td>42.0</td>
      <td>46484.0</td>
      <td>82206.0</td>
      <td>51517.0</td>
      <td>233.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>1150</td>
      <td>8320.0</td>
      <td>16323.0</td>
      <td>29532.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5334.0</td>
      <td>6992.0</td>
      <td>8835.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.resample('M').sum()
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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-03-31</th>
      <td>36</td>
      <td>756.0</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-04-30</th>
      <td>365</td>
      <td>1956.0</td>
      <td>221.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1250.0</td>
      <td>77.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-05-31</th>
      <td>467</td>
      <td>2239.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1500.0</td>
      <td>100.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-06-30</th>
      <td>931</td>
      <td>2614.0</td>
      <td>284.0</td>
      <td>879.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1753.0</td>
      <td>196.0</td>
      <td>235.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-07-31</th>
      <td>1157</td>
      <td>4230.0</td>
      <td>2123.0</td>
      <td>4205.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3163.0</td>
      <td>1213.0</td>
      <td>1895.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-08-31</th>
      <td>1596</td>
      <td>6158.0</td>
      <td>9557.0</td>
      <td>9284.0</td>
      <td>147.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4338.0</td>
      <td>5153.0</td>
      <td>3883.0</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-09-30</th>
      <td>2130</td>
      <td>8711.0</td>
      <td>22525.0</td>
      <td>15534.0</td>
      <td>145.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5463.0</td>
      <td>12064.0</td>
      <td>5092.0</td>
      <td>56.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-10-31</th>
      <td>3527</td>
      <td>13504.0</td>
      <td>38070.0</td>
      <td>33013.0</td>
      <td>160.0</td>
      <td>8.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>7835.0</td>
      <td>19352.0</td>
      <td>10365.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2014-11-30</th>
      <td>3321</td>
      <td>15604.0</td>
      <td>49273.0</td>
      <td>46749.0</td>
      <td>160.0</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>9396.0</td>
      <td>20502.0</td>
      <td>10054.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>2983</td>
      <td>20637.0</td>
      <td>55318.0</td>
      <td>71887.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>11290.0</td>
      <td>23534.0</td>
      <td>19963.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2015-01-31</th>
      <td>1150</td>
      <td>8320.0</td>
      <td>16323.0</td>
      <td>29532.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5334.0</td>
      <td>6992.0</td>
      <td>8835.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



とても便利です。さらに、datetime型のもう一つの利点として、`.year`や`.month`などのメソッドを利用して、年月日を取得することが出来ます。


```python
df.index.year
```




    Int64Index([2015, 2015, 2015, 2015, 2014, 2014, 2014, 2014, 2014, 2014,
                ...
                2014, 2014, 2014, 2014, 2014, 2014, 2014, 2014, 2014, 2014],
               dtype='int64', name='Date', length=122)




```python
df.index.month
```




    DatetimeIndex(['2015-01-05', '2015-01-04', '2015-01-03', '2015-01-02',
                   '2014-12-31', '2014-12-28', '2014-12-27', '2014-12-24',
                   '2014-12-21', '2014-12-20',
                   ...
                   '2014-04-04', '2014-04-01', '2014-03-31', '2014-03-29',
                   '2014-03-28', '2014-03-27', '2014-03-26', '2014-03-25',
                   '2014-03-24', '2014-03-22'],
                  dtype='datetime64[ns]', name='Date', length=122, freq=None)




```python
df.index.day
```




    Int64Index([ 5,  4,  3,  2, 31, 28, 27, 24, 21, 20,
                ...
                 4,  1, 31, 29, 28, 27, 26, 25, 24, 22],
               dtype='int64', name='Date', length=122)



## cut処理（ヒストグラムの作成）


```python
labels = ['上旬', '中旬', '下旬']
df['period'] = pd.cut(list(df.index.day),  bins=[0,10,20,31], labels=labels, right=True) # 0<day≦10, 10<day≦20, 20<day≦31

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
      <th>Day</th>
      <th>Cases_Guinea</th>
      <th>Cases_Liberia</th>
      <th>Cases_SierraLeone</th>
      <th>Cases_Nigeria</th>
      <th>Cases_Senegal</th>
      <th>Cases_UnitedStates</th>
      <th>Cases_Spain</th>
      <th>Cases_Mali</th>
      <th>Deaths_Guinea</th>
      <th>Deaths_Liberia</th>
      <th>Deaths_SierraLeone</th>
      <th>Deaths_Nigeria</th>
      <th>Deaths_Senegal</th>
      <th>Deaths_UnitedStates</th>
      <th>Deaths_Spain</th>
      <th>Deaths_Mali</th>
      <th>period</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-05</th>
      <td>289</td>
      <td>2776.0</td>
      <td>NaN</td>
      <td>10030.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1786.0</td>
      <td>NaN</td>
      <td>2977.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>上旬</td>
    </tr>
    <tr>
      <th>2015-01-04</th>
      <td>288</td>
      <td>2775.0</td>
      <td>NaN</td>
      <td>9780.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1781.0</td>
      <td>NaN</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>上旬</td>
    </tr>
    <tr>
      <th>2015-01-03</th>
      <td>287</td>
      <td>2769.0</td>
      <td>8166.0</td>
      <td>9722.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1767.0</td>
      <td>3496.0</td>
      <td>2915.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>上旬</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>286</td>
      <td>NaN</td>
      <td>8157.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>上旬</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>284</td>
      <td>2730.0</td>
      <td>8115.0</td>
      <td>9633.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1739.0</td>
      <td>3471.0</td>
      <td>2827.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>下旬</td>
    </tr>
  </tbody>
</table>
</div>



## queryとwhereの使い方 (ソートも)

## nullの使い方

## 列名やインデックス名の変更
上で既に出てきていますが、列名やインデックスの名前を変更したい場合はよくあります。renameを使います。


```python
df.rename(columns={'before': 'after'}, inplace=True)
df.rename(index={'before': 'after'}, inplace=True)
```

## get_dummiesの使い方

## CSVへ出力

## 頻出のコマンド一覧
概要として、よく利用するコマンドを以下に載せます。

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


## よく使う関数

最後のまとめとして、良く使う関数をまとめておきます。個人的なsnipetみたいなものです。

#### インデックスの変更(既存のカラム名に変更)

```python
df.set_index('xxxx')
```

#### カラム名の変更

```python
df.rename(columns={'before': 'after'}, inplace=True)
df.rename(index={'before': 'after'}, inplace=True)
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
