
## tail
ファイルの末尾を表示します。headオプションの逆です。

```bash
TAIL(1)                   BSD General Commands Manual 

NAME
     tail -- display the last part of a file

SYNOPSIS
     tail [-F | -f | -r] [-q] [-b number | -c number | -n number] [file ...]

```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/tail/tail_nb.ipynb)

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


```bash
%%bash
echo "ファイルの準備"
echo -e "1\n2 \n3 \n4 \n5 \n6" > temp
cat temp

echo -e "\n<ファイルの末尾3行を表示>"
tail -n 3 temp
```

    ファイルの準備
    1
    2 
    3 
    4 
    5 
    6
    
    <ファイルの末尾3行を表示>
    4 
    5 
    6


## 代表的なオプション
- n : 表示する行数
