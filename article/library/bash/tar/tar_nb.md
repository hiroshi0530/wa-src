
## tar
ファイルを結合し、アーカイブを作成します。また、アーカイブからファイルを展開します。

```bash
BSDTAR(1)                 BSD General Commands Manual 

NAME
     tar -- manipulate tape archives

SYNOPSIS
     tar [bundled-flags <args>] [<file> | <pattern> ...]
     tar {-c} [options] [files | directories]
     tar {-r | -u} -f archive-file [options] [files | directories]
     tar {-t | -x} [options] [patterns]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/tar/tar_nb.ipynb)

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

### ファイルの作成


```bash
%%bash
echo "ファイルの準備"
echo "1234567890" > temp1
echo "0987654321" > temp2

tar -czf temp.tgz temp1 temp2
echo -e "\n<ls>"
ls | grep temp

```

    ファイルの準備
    
    <ls>
    temp.tgz
    temp1
    temp2


### ファイルの解凍


```bash
%%bash
tar -xzf temp.tgz
```

## 代表的なオプション
長年、4つのコマンドを組み合わせた以下の二つのコマンドを利用しています。
vはverboseで必要な場合利用します。
今までこの二つで困ったことはありません。

### ファイルの解凍
- xzvf

### ファイルの作成
- czvf
