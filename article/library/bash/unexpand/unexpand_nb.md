
## unexpend
与えられたファイルのスペースをタブに変換します。結果を標準出力に表示します。

```bash
NAME
     expand, unexpand -- expand tabs to spaces, and vice versa

SYNOPSIS
     expand [-t tab1,tab2,...,tabn] [file ...]
     unexpand [-a | -t tab1,tab2,...,tabn] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)


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
echo -e "a   b\nc   d" > temp1.txt
cat temp1.txt
```

    a   b
    c   d



```bash
%%bash
unexpand -a -t 3 temp1.txt > temp2.txt
cat -te temp2.txt
```

    a^I b$
    c^I d$

