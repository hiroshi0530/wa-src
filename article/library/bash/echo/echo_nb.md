
## echo
文字列を標準出力に表示します。

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/echo/echo_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/echo/echo_nb.ipynb)

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


```bash
%%bash
echo "echo is basic option."
```

    echo is basic option.


## 代表的なオプション
- n : 改行コードを付与しない
- e : エスケープ文字を有効にして表示する

### n オプション


```bash
%%bash
echo -n "linux"
echo " Linux"
```

    linux Linux


### e オプション


```bash
%%bash
echo -e "this is a pen.\nthis is a pen."
```

    this is a pen.
    this is a pen.

