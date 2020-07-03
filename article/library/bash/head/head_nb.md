
## head
ファイルの先頭、先頭から指定された行を表示します。

```bash
NAME
     head -- display first lines of a file

SYNOPSIS
     head [-n count | -c bytes] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/head/head_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/head/head_nb.ipynb)

### 環境
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


## 使用例

オプションなしのデフォルト設定では最初の10行を表示します。


```bash
%%bash
echo "テスト用のテキストの作成"
echo -e "1 \n2 \n3 \n4 \n5 \n6 \n7 \n8 \n9 \n10 \n11 \n12" > temp
head temp
echo "先頭の10行しか表示されない"
```

    テスト用のテキストの作成
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
    先頭の10行しか表示されない


## 代表的なオプション
- c : 先頭から指定したバイト数を表示
- n : 先頭から指定した行数を表示⇒最も良く使うオプション

### n オプション


```bash
%%bash
echo -e "test \ntest \ntest \ntest \ntest \ntest \n" > test
echo "先頭3行目を表示する"
head -n 3 test
```

    先頭3行目を表示する
    test 
    test 
    test 

