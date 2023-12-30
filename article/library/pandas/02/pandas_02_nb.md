## pandasでリストとして格納されている各要素をカラムとして設定し、ワンホットエンコードとして展開する


データ分析をしていて、pandasの要素にリストが格納されており、そのリストに対してワンホットエンコードした状態のDataFrameを作りたいという機会があり、結構苦労したのでメモしておきます。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/pandas/02/pandas_02_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/pandas/02/pandas_02_nb.ipynb)

### 実行環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:		macOS
    ProductVersion:		13.5.1
    BuildVersion:		22G90



```python
!python -V
```

    Python 3.9.17


基本的なライブラリをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline

import pandas as pd

print('pandas version :', pd.__version__)
```

    pandas version : 2.0.3


### サンプルデータの準備


```python
df = pd.DataFrame(
    {
        "user_id": ["A", "B", "C"],
        "item_id": [["PC", "Book", "Water"], ["Book", "Table"], ["Desk", "CD"]],
    }
)

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
      <th>user_id</th>
      <th>item_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>[PC, Book, Water]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>[Book, Table]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>[Desk, CD]</td>
    </tr>
  </tbody>
</table>
</div>



### MultiLabelBinarizerの利用

結論から述べると、`MultiLabelBinarizer` というscikit-learnのライブラリを利用します。

以下のように、 `fit_transform`を利用する事で、ワンホットエンコードを簡単に実現できます。また、それに対応するカラム名も簡単に取得できます。


```python
from sklearn.preprocessing import MultiLabelBinarizer


mlb = MultiLabelBinarizer()
mlb.fit_transform(df.item_id)
```




    array([[1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0],
           [0, 1, 1, 0, 0, 0]])




```python
mlb.classes_
```




    array(['Book', 'CD', 'Desk', 'PC', 'Table', 'Water'], dtype=object)



あとはこれを組み合わせるだけです。popでdfから取りだして、最後にjoinで結合します。


```python
out_df = df.join(pd.DataFrame(mlb.fit_transform(df.pop("item_id")), columns=mlb.classes_))

out_df
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
      <th>user_id</th>
      <th>Book</th>
      <th>CD</th>
      <th>Desk</th>
      <th>PC</th>
      <th>Table</th>
      <th>Water</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 参考サイト

- https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
