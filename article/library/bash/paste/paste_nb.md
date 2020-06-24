
## paste
テキストファイルを列方向に結合します。catの列板です。

```text
NAME
     paste -- merge corresponding or subsequent lines of files

SYNOPSIS
     paste [-s] [-d list] file ...
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/paste/paste_nb.ipynb)

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

結合する二つのファイルを作成します。


```bash
%%bash
echo -e "a\nb\nc\nd\ne\nf" > temp1.txt
echo -e " 1\n 2\n 3\n 4\n 5\n 6" > temp2.txt
```


```bash
%%bash
paste temp1.txt temp2.txt
```

    a	 1
    b	 2
    c	 3
    d	 4
    e	 5
    f	 6


３つ以上のファイルも連結できます。


```bash
%%bash
echo -e "a\nb\nc\nd\ne\nf" > temp3.txt
echo -e " 1\n 2\n 3\n 4\n 5\n 6" > temp4.txt
paste temp1.txt temp2.txt temp3.txt temp4.txt
```

    a	 1	a	 1
    b	 2	b	 2
    c	 3	c	 3
    d	 4	d	 4
    e	 5	e	 5
    f	 6	f	 6


### 代表的なオプション
- d : 結合文字を指定します。デフォルトはタブです。
- s : 行と列を反転させます。 


```bash
%%bash
paste -d _ temp1.txt temp2.txt
```

    a_ 1
    b_ 2
    c_ 3
    d_ 4
    e_ 5
    f_ 6



```bash
%%bash
paste -s temp1.txt temp2.txt
```

    a	b	c	d	e	f
     1	 2	 3	 4	 5	 6

