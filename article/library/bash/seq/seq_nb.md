
## seq
連番を作成します。

```bash
NAME
     cat -- concatenate and print files

SYNOPSIS
     cat [-benstuv] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/seq/seq_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/seq/seq_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

実際に動かす際は先頭の！や先頭行の%%bashは無視してください。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G2022



```python
!bash --version
```

    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin18)
    Copyright (C) 2007 Free Software Foundation, Inc.


### 通常の使い方


```python
!for i in `seq 10`; do echo $i; done
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


seqを利用しなくても、{..}でいけます。0埋めなど知る必要がなければこれで十分です。


```python
!for i in {0..10}; do echo $i; done
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


{..}は配列を作る演算子です。


```python
!echo {1..10}
```

    1 2 3 4 5 6 7 8 9 10


### 連番の0埋め


```python
!for i in `seq -w 3 10`; do echo $i; done
```

    03
    04
    05
    06
    07
    08
    09
    10


### 埋める0の量を変更 


```python
!for i in `seq -f %03g 3 10`; do echo $i; done
```

    003
    004
    005
    006
    007
    008
    009
    010

