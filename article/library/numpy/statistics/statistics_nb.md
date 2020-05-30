
## Numpy個人的tips

numpyもデータ分析や数値計算には欠かせないツールの一つです。機械学習などを実装していると必ず必要とされるライブラリです。個人的な備忘録としてメモを残しておきます。詳細は以下の公式ページを参照してください。
- [公式ページ](https://docs.scipy.org/doc/numpy/reference/)

### 目次
- [1. 基本的な演算](/article/library/numpy/base/)
- [2. 三角関数](/article/library/numpy/trigonometric/)
- [3. 指数・対数](/article/library/numpy/explog/)
- [4. 統計関数](/article/library/numpy/statistics/) <= 今ここ
- [5. 線形代数](/article/library/numpy/matrix/)
- [6. サンプリング](/article/library/numpy/sampling/)
- [7. その他](/article/library/numpy/misc/)

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/numpy/statistics/statistics_nb.ipynb)

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



## 統計情報の取得

### np.max(x)
配列の最大値を返します。

2階のテンソルとして$a$を定義します。


```python
a = np.array([
    [1,8,3],
    [6,5,4],
    [7,2,9]
  ]
)
```

3階のテンソルとして$b$を定義します。


```python
b = np.array([
  [
    [1,8,3],
    [6,5,4],
    [7,2,9]
  ],
  [
    [1,9,4],
    [7,2,5],
    [6,8,3]
  ]
])
```


```python
print('-' * 20)
print('a   : \n',a)
print()
print('np.max(a) : \n',np.max(a))
print()
print('np.max(a, axis=0) : \n',np.max(a, axis=0))
print()
print('np.max(a, axis=1) : \n',np.max(a, axis=1))
print()

print('-' * 20)
print('b   : \n',b)
print()
print('np.max(b) : \n',np.max(b))
print()
print('np.max(b, axis=0) : \n',np.max(b, axis=0))

print()
print('np.max(b, axis=1) : \n',np.max(b, axis=1))

print()
print('np.max(b, axis=2 : \n',np.max(b, axis=2))
```

    --------------------
    a   : 
     [[1 8 3]
     [6 5 4]
     [7 2 9]]
    
    np.max(a) : 
     9
    
    np.max(a, axis=0) : 
     [7 8 9]
    
    np.max(a, axis=1) : 
     [8 6 9]
    
    --------------------
    b   : 
     [[[1 8 3]
      [6 5 4]
      [7 2 9]]
    
     [[1 9 4]
      [7 2 5]
      [6 8 3]]]
    
    np.max(b) : 
     9
    
    np.max(b, axis=0) : 
     [[1 9 4]
     [7 5 5]
     [7 8 9]]
    
    np.max(b, axis=1) : 
     [[7 8 9]
     [7 9 5]]
    
    np.max(b, axis=2 : 
     [[8 6 9]
     [9 7 8]]



```python
print('-' * 20)
print('a   : \n',a)
print()
print('np.argmax(a) : \n',np.argmax(a))
print()
print('np.argmax(a, axis=0) : \n',np.argmax(a, axis=0))
print()
print('np.argmax(a, axis=1) : \n',np.argmax(a, axis=1))
print()

print('-' * 20)
print('b   : \n',b)
print()
print('np.argmax(b) : \n',np.argmax(b))
print()
print('np.argmax(b, axis=0) : \n',np.argmax(b, axis=0))

print()
print('np.argmax(b, axis=1) : \n',np.argmax(b, axis=1))

print()
print('np.argmax(b, axis=2 : \n',np.argmax(b, axis=2))
```

    --------------------
    a   : 
     [[1 8 3]
     [6 5 4]
     [7 2 9]]
    
    np.argmax(a) : 
     8
    
    np.argmax(a, axis=0) : 
     [2 0 2]
    
    np.argmax(a, axis=1) : 
     [1 0 2]
    
    --------------------
    b   : 
     [[[1 8 3]
      [6 5 4]
      [7 2 9]]
    
     [[1 9 4]
      [7 2 5]
      [6 8 3]]]
    
    np.argmax(b) : 
     8
    
    np.argmax(b, axis=0) : 
     [[0 1 1]
     [1 0 1]
     [0 1 0]]
    
    np.argmax(b, axis=1) : 
     [[2 0 2]
     [1 0 1]]
    
    np.argmax(b, axis=2 : 
     [[1 0 2]
     [1 0 1]]


### np.argmax(x)
配列の最大値の位置を返します。


```python
a = np.random.randint(100,size=10)

print('a            : ',a)
print('max position : ',np.argmax(a))
```

    a            :  [53 35 94  2  3 14 21 55 17  6]
    max position :  2


### np.min(x)
配列の最小値を返します。


```python
a = np.random.randint(100,size=10)

print('a   : ',a)
print('min : ',np.min(a))
```

    a   :  [36 42  6 71 92 23 44 92 36 79]
    min :  6


### np.argmax(x)
配列の最小値の位置を返します。


```python
a = np.random.randint(100,size=10)

print('a            : ',a)
print('min position : ',np.argmin(a))
```

    a            :  [51 76 59 12 28 50 21 61 49 37]
    min position :  3


### np.maximum(x,y)
二つの配列を比較し、大きい値を選択し新たなndarrayを作ります。


```python
a = np.random.randint(100,size=10)
b = np.random.randint(100,size=10)

print('a   : ',a)
print('b   : ',b)
print('max : ',np.maximum(a,b))
```

    a   :  [25 78 95 45 79 33 72 33 38 81]
    b   :  [41 91 64  7 60 54 29 25 99 88]
    max :  [41 91 95 45 79 54 72 33 99 88]


### np.minimum(x,y)
二つの配列を比較し、小さい値を選択し新たなndarrayを作ります。


```python
a = np.random.randint(100,size=10)
b = np.random.randint(100,size=10)

print('a   : ',a)
print('b   : ',b)
print('min : ',np.minimum(a,b))
```

    a   :  [80 81 40 80 47 81 17 86 91 63]
    b   :  [84 51  7  4 62 66 83 85 21 66]
    min :  [80 51  7  4 47 66 17 85 21 63]


### np.sum(a, axis=None, dtype=None, out=None, keepdims=[no value], initial=[no value], where=[no value])


```python
a = np.arange(10)
np.sum(a)
```




    45



axisを指定して計算してみます。


```python
a = np.arange(12).reshape(3,4)

print('a : ')
print(a)
print('sum axis=0 : ', np.sum(a, axis=0))
print('sum axis=1 : ', np.sum(a, axis=1))
```

    a : 
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    sum axis=0 :  [12 15 18 21]
    sum axis=1 :  [ 6 22 38]


### np.average(a, axis=None, weights=None, returned=False)
平均を求めます。重み付きの平均も求める事が出来ます。

単純に配列の平均です。


```python
a = np.arange(10)
np.average(a)
```




    4.5



axisを指定した平均です。


```python
a = np.arange(12).reshape(3,4)

print('a : ', a)
print('average axis = 0 : ',np.average(a, axis=0))
print('average axis = 1 : ',np.average(a, axis=1))
```

    a :  [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    average axis = 0 :  [4. 5. 6. 7.]
    average axis = 1 :  [1.5 5.5 9.5]


重みを指定します。


```python
a = np.arange(5)

# 適当に重みを設定
w = np.array([0.1,0.2,0.5,0.15,0.05])

np.average(a,weights=w)
```




    1.7619047619047616



### np.mean(a, axis=None, dtype=None, out=None, keepdims=[no value])
平均を求めます。こちらは重み付きの平均を求める事が出来ません。しかし、計算時の型を指定することが出来ます。


```python
x = np.arange(10)
np.mean(x)
```




    4.5



整数型を指定して計算する。


```python
x = np.arange(10)
np.mean(x, dtype='int8')
```




    array([4], dtype=int8)



### np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=[no value])
標準偏差を求めます。


```python
x = np.arange(10)
np.std(x)
```




    2.8722813232690143



### np.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=[no value])
分散を求めます。


```python
x = np.arange(10)
np.var(x)
```




    8.25



### np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)


```python
x = np.arange(10)
print(x)
print('median x : ',np.median(x))
print()

x = np.arange(11)
print(x)
print('median x : ',np.median(x))
```

    [0 1 2 3 4 5 6 7 8 9]
    median x :  4.5
    
    [ 0  1  2  3  4  5  6  7  8  9 10]
    median x :  5.0


### np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)

bias=Trueで標本分散を求める。
yで追加の配列を指定可能。


```python
a = np.random.randint(10,size=9).reshape(3,3)
b = np.arange(3)

print('a : ')
print(a)
print()

print('不偏分散での共分散行列')
print(np.cov(a))
print()

print('標本分散での共分散行列')
print(np.cov(a, bias=True))
print()

print('それぞれの成分の標本分散 : 共分散行列の対角成分と一致')
print('var a[0] = ', np.var(a[0]))
print('var a[1] = ', np.var(a[1]))
print('var a[2] = ', np.var(a[2]))
print()

print('bを追加')
print('b : ')
print(b)
print(np.cov(a,b, bias=True))
```

    a : 
    [[2 2 1]
     [0 1 6]
     [0 9 3]]
    
    不偏分散での共分散行列
    [[ 0.33333333 -1.83333333  0.5       ]
     [-1.83333333 10.33333333 -0.5       ]
     [ 0.5        -0.5        21.        ]]
    
    標本分散での共分散行列
    [[ 0.22222222 -1.22222222  0.33333333]
     [-1.22222222  6.88888889 -0.33333333]
     [ 0.33333333 -0.33333333 14.        ]]
    
    それぞれの成分の標本分散 : 共分散行列の対角成分と一致
    var a[0] =  0.2222222222222222
    var a[1] =  6.888888888888888
    var a[2] =  14.0
    bを追加
    b : 
    [0 1 2]
    [[ 0.22222222 -1.22222222  0.33333333 -0.33333333]
     [-1.22222222  6.88888889 -0.33333333  2.        ]
     [ 0.33333333 -0.33333333 14.          1.        ]
     [-0.33333333  2.          1.          0.66666667]]


### np.corrcoef(x, y=None, rowvar=True, bias=[no value], ddof=[no value])


```python
a = np.random.randint(10,size=9).reshape(3,3)
np.corrcoef(a)
```




    array([[ 1.        ,  0.24019223, -0.75592895],
           [ 0.24019223,  1.        , -0.81705717],
           [-0.75592895, -0.81705717,  1.        ]])




