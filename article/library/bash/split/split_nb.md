
## split
ファイルの分割を行います。

```bash
NAME
     split -- split a file into pieces

SYNOPSIS
     split [-a suffix_length] [-b byte_count[k|m]] [-l line_count]
           [-p pattern] [file [name]]

```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)


### 環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

linux環境では-nオプションで分割数が指定出来るのですが、manを見てもわかるとおり、FreeBSD（macOSの元となるOS）ではそういうオプションはありません。

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

通常、以下の様な代表的なオプションと共に利用します。

### 代表的なオプション
- b : 分割するバイト数
- l : 分割する行数
- a : prefixに利用する文字数



```bash
%%bash
echo -e "1\n2\n3\n4\n5\n6\n7\n8\n9\n10" > temp
cat temp

split -l 2 -a 3 temp prefix_
echo -e "\n<file list>"
ls | grep -v split

echo -e "\n<prefix_aaa file content>"
cat prefix_aaa
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
    
    <file list>
    prefix_aaa
    prefix_aab
    prefix_aac
    prefix_aad
    prefix_aae
    temp
    
    <prefix_aaa file content>
    1
    2

