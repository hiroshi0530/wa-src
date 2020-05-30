
## Numpy個人的tips

numpyもデータ分析や数値計算には欠かせないツールの一つです。機械学習などを実装していると必ず必要とされるライブラリです。個人的な備忘録としてメモを残しておきます。詳細は以下の公式ページを参照してください。
- [公式ページ](https://docs.scipy.org/doc/numpy/reference/)

### 目次
- [1. 基本的な演算](/article/library/numpy/base/)
- [2. 三角関数](/article/library/numpy/trigonometric/)
- [3. 指数・対数](/article/library/numpy/explog/) <= 今ここ
- [4. 統計関数](/article/library/numpy/statistics/)
- [5. 線形代数](/article/library/numpy/matrix/)
- [6. サンプリング](/article/library/numpy/sampling/)
- [7. その他](/article/library/numpy/misc/)

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/explog/explog_nb.ipynb)

### 筆者の環境
筆者の環境とimportの方法は以下の通りです。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G2022



```python
!python -V
```

    Python 3.7.3



```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import numpy as np

np.__version__
```




    '1.16.2'



## 指数と対数

### np.exp(x)
$\exp{x}$を計算します。


```python
print(np.exp(0))
print(np.exp(1))
print(np.exp(2))
```

    1.0
    2.718281828459045
    7.38905609893065


### np.expm1(x)
$\exp{x}-1$を計算します。


```python
print(np.expm1(0))
print(np.expm1(1))
print(np.expm1(2))
```

    0.0
    1.7182818284590453
    6.38905609893065


### np.exp2(x)
$2^{x}$を計算します。


```python
print(np.exp2(0))
print(np.exp2(1))
print(np.exp2(2))
```

    1.0
    2.0
    4.0


### np.log(x)
$\log{x}$を計算します。底は自然対数になります。


```python
print(np.log(1))
print(np.log(2))
print(np.log(np.e))
```

    0.0
    0.6931471805599453
    1.0


### np.log10(x)
$\log_{10}{x}$を計算します。


```python
print(np.log10(1))
print(np.log10(2))
print(np.log10(10))
```

    0.0
    0.3010299956639812
    1.0


### np.log2(x)
$\log_{2}{x}$を計算します。


```python
print(np.log2(1))
print(np.log2(2))
print(np.log2(10))
```

    0.0
    1.0
    3.321928094887362


### np.log1p(x)
$\log{(x + 1)}$を計算します。底は自然対数になります。これはデータ分析においてよく利用します。元の数字が0になる場合、1を足して対数をとり、分類器にかけることがしばしば発生します。いわゆるloglossを計算するときです。


```python
print(np.log1p(0))
print(np.log1p(1))
print(np.log1p(-1 + np.e))
```

    0.0
    0.6931471805599453
    1.0

