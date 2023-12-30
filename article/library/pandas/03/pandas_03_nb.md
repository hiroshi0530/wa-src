## pandasでリストとして格納されている各要素を展開する

pandasに長さが等しいリストが格納されており、そのリストを展開して、それぞれ独立したカラムする方法をメモしておきます。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/scipy/pandas/pandas_03_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/scipy/pandas/pandas_03_nb.ipynb)

### 筆者の環境
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
        "item_id": [["PC", "Book"], ["Book", "Table"], ["Desk", "CD"]],
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
      <td>[PC, Book]</td>
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



### 結論

展開したいカラムを`pd.Series`を引数にapplyを適用します。
`item_id`カラムに対して適用し、適当にカラム名を設定します。


```python
df.item_id.apply(pd.Series).set_axis(["col1", "col2"], axis=1)
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC</td>
      <td>Book</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Book</td>
      <td>Table</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Desk</td>
      <td>CD</td>
    </tr>
  </tbody>
</table>
</div>



必要なカラムだけに適用したいので、`pop`と`join`を利用します。


```python
df.join(df.pop("item_id").apply(pd.Series).set_axis(["col1", "col2"], axis=1))
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>PC</td>
      <td>Book</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>Book</td>
      <td>Table</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>Desk</td>
      <td>CD</td>
    </tr>
  </tbody>
</table>
</div>



想定通りのDataFrameができました。


