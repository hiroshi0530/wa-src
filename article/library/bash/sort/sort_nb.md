
## sort
ファイルを読み込み、降順、昇順に並び替えをします。

```text
NAME
     sort -- sort or merge records (lines) of text and binary files

SYNOPSIS
     sort [-bcCdfghiRMmnrsuVz] [-k field1[,field2]] [-S memsize] [-T dir]
          [-t char] [-o output] [file ...]
     sort --help
     sort --version
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/sort/sort_nb.ipynb)

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
echo -e "b\nc\na\nz\ny" > temp1.txt
cat temp1.txt
sort -r temp1.txt > temp2.txt
echo -e "\nsorted"
cat temp2.txt
```

    b
    c
    a
    z
    y
    
    sorted
    z
    y
    c
    b
    a

