
##  cut
テキストファイルを読み取り、特定の文字列で分割します。タブで分割されたファイルなどから特定の列の値を抽出します。

```text
NAME
     cut -- cut out selected portions of each line of a file

SYNOPSIS
     cut -b list [-n] [file ...]
     cut -c list [file ...]
     cut -f list [-d delim] [-s] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/article/library/bash/cut/cut_nb.ipynb)

### 環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

実際に動かす際は先頭の！や先頭行の%%bashは無視してください。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G95



```python
!bash --version
```

    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin18)
    Copyright (C) 2007 Free Software Foundation, Inc.


## 使用例

### 代表的なオプション
- d : 区切り文字を指定します
- f : 区切りから何文字目の文字を抽出するか選択します

テスト用のファイルを作成します。タブ区切りで作成します。


```bash
%%bash
echo -e "a\tb\nc\td" > temp1.txt
cat -t temp1.txt
```

    a^Ib
    c^Id


1列目を表示します。


```bash
%%bash
cut -f 1 temp1.txt
```

    a
    c


2列目を表示します。


```bash
%%bash
cut -f 2 temp1.txt
```

    b
    d


スペースで区切られたファイルを作成します。


```bash
%%bash
echo -e "a b\nc d" > temp2.txt
cat temp2.txt
```

    a b
    c d



```bash
%%bash
cut -d ' ' -f 1 temp2.txt
```

    a
    c



```bash
%%bash
cut -d ' ' -f 2 temp2.txt
```

    b
    d

